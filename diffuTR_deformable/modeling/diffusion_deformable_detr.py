# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import random
from collections import namedtuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.modeling.poolers import ROIPooler
from .diffusion_modules import SinusoidalPositionEmbeddings

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# TODO: Refine the crops, currently easy to crop out of the image (no error though)
def generate_random_boxes_cxcywh(num_boxes):
    # xyxy
    boxes = torch.rand(num_boxes, 4)
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    boxes = boxes.clamp(0, 1)
    # cxcywh
    boxes = box_xyxy_to_cxcywh(boxes)  # may have negative numbers?
    assert (boxes >= 0).all(), "boxes should be non-negative"
    return boxes

def get_area_of_boxes_xyxy(boxes):
    assert (boxes >= 0).all(), "boxes should be non-negative" 
    assert (boxes[:, 2:] >= boxes[:, :2]).all(), "boxes should be in (x1, y1, x2, y2) format"

    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # if some values in area is too small, set it to 1e-6
    area = torch.where(area < 1e-6, torch.ones_like(area) * 1e-6, area)
    return area

def move_tensors_to_device(container, device):
    if isinstance(container, dict):
        for key in container:
            if isinstance(container[key], torch.Tensor):
                container[key] = container[key].to(device)
    elif isinstance(container, list):
        for i in range(len(container)):
            if isinstance(container[i], torch.Tensor):
                container[i] = container[i].to(device)
    elif isinstance(container, torch.Tensor):
        container = container.to(device)
    else:
        raise NotImplementedError
    return container


@META_ARCH_REGISTRY.register()
class DiffuTR(nn.Module):
    """Implements the Deformable DETR model.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        backbone (nn.Module): the backbone module.
        position_embedding (nn.Module): the position embedding module.
        neck (nn.Module): the neck module.
        transformer (nn.Module): the transformer module.
        embed_dim (int): the dimension of the embedding.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): whether to use auxiliary loss. Default: True.
        with_box_refine (bool): whether to use box refinement. Default: False.
        as_two_stage (bool): whether to use two-stage. Default: False.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.

    """

    def __init__(
        self,
        backbone,
        position_embedding,
        neck,
        transformer,
        embed_dim,
        num_classes,
        num_queries,
        criterion,
        pixel_mean,
        pixel_std,
        aux_loss=True,
        with_box_refine=False,
        as_two_stage=False,
        select_box_nums_for_evaluation=100,
        device="cuda",

        # Diffusion
        latent_dim=256,
        with_diffusion=True,
        snr_scale=2.0,
        sample_step=1,  # TODO: Check default
        normalize_latent=False,
        scale_latent=1.,
    ):
        super().__init__()
        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # define neck module
        self.neck = neck

        # define learnable query embedding
        self.num_queries = num_queries
        if not as_two_stage:
            self.query_embedding = nn.Embedding(num_queries, embed_dim * 2)

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        self.num_classes = num_classes
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # define contoller for box refinement and two-stage variants
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage

        # init parameters for heads
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # If two-stage, the last class_embed and bbox_embed is for region proposal generation
        # Decoder layers share the same heads without box refinement, while use the different
        # heads when box refinement is used.
        num_pred = (
            (transformer.decoder.num_layers + 1) if as_two_stage else transformer.decoder.num_layers
        )
        if with_box_refine:
            self.class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for i in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for i in range(num_pred)]
            )
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        # hack implementation for two-stage. The last class_embed and bbox_embed is for region proposal generation
        if as_two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        #######################################################################
        # Add diffusion below
        self.with_diffusion = with_diffusion
        assert self.with_diffusion is True

        self.size_divisibility = self.backbone.size_divisibility
    
        # build diffusion
        timesteps = 1000
        sampling_timesteps = sample_step
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = snr_scale
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        

        # DiffuTR: other configs
        self.latent_dim = latent_dim
        cfg=None # TODO: roi_configs
        self.box_pooler = self._init_box_pooler(cfg, roi_input_shape=self.backbone.output_shape())
        
        # Not the same as diffusionDet
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.roi_conv = nn.Conv2d(
            latent_dim, latent_dim, kernel_size=7
        )
        self.label_embedder = nn.Embedding(80, latent_dim)  # Hard_coded to 80
        self.box_mlp = nn.Sequential(
            nn.Linear(4, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # fuse roi, box, label -> latent
        self.fuse_mlp = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.normalize_latent = normalize_latent
        self.scale_latent = scale_latent



    @staticmethod
    def _init_box_pooler(cfg, input_shape):
        
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler    

    def get_img_masks(self, images, batched_inputs):
        if self.training:
            bs, _, H, W  = images.tensor.shape
            img_masks = images.tensor.new_ones(bs, H, W)
            for img_id in range(bs):
                # mask padding regions in batched images
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            bs, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(bs, H, W)
        return img_masks

    def forward(self, batched_inputs):
        images, images_whwh = self.preprocess_image(batched_inputs)
        img_masks = self.get_img_masks(images, batched_inputs)
        
        # Add in DiffuTR
        # 1.Feature Extraction.
        backbone_feats = self.backbone(images.tensor)  # output backbone feature dict
        
        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in deformable DETR
        mlvl_feats = self.neck(backbone_feats)
        mlvl_masks = []
        mlvl_pos_embs = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            mlvl_pos_embs.append(self.position_embedding(mlvl_masks[-1]))
        
        # initialize object query embeddings
        # query_embeds = None
        # if not self.as_two_stage:
        #     query_embeds = self.query_embedding.weight

        feature_dict = dict(
            multi_level_feats=mlvl_feats, 
            multi_level_masks=mlvl_masks, 
            multi_level_pos_embeds=mlvl_pos_embs,
            query_embed=None,  # use ins_latents later
        )

        # 2.Forward test/train
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets, latent_dict = self.prepare_targets_and_noisy_inputs(gt_instances, mlvl_feats)  # TODO: No diffusion mode
            # t = t.squeeze(-1)
            # x_boxes = x_boxes * images_whwh[:, None, :]

            # TODO: replace query with diffused latents
            diffused_latents = latent_dict["diffused_ins_latents"]  # [bs, num_ins, latent_dim]
            feature_dict["query_embed"] = diffused_latents

            output = self.forward_detr_head(feature_dict)

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            results = self.test_with_ddim_sampling(batched_inputs, feature_dict, images_whwh, images)
            return results


    def forward_detr_head(self, feature_dict):
        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(**feature_dict)
        
        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.as_two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            output["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return output


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # Select top-k confidence boxes for inference
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results
    
    def prepare_targets_no_diffusion(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def prepare_targets_and_noisy_inputs(self, targets, mlvl_feats):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)

            target["labels"] = gt_classes
            target["boxes"] = gt_boxes  # centered, normalized(0-1)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor  # xyxy, un-normalized
            target["image_size_xyxy"] = image_size_xyxy
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt
            target["area"] = targets_per_image.gt_boxes.area()
            target = move_tensors_to_device(target, self.device)
            
            # pad noise_gt for one-to-one training
            padded_target = {}
            n_boxes = gt_boxes.size(0)
            assert n_boxes <= self.num_queries
            if n_boxes < self.num_queries:
                n_pad_boxes = self.num_queries - n_boxes
                pad_labels = -1 * torch.ones(n_pad_boxes)  # -1 for negative samples
                padded_target["labels"] = torch.cat([gt_classes, pad_labels], dim=0)

                pad_boxes = generate_random_boxes_cxcywh(n_pad_boxes)
                padded_target["boxes"] = torch.cat([gt_boxes, pad_boxes], dim=0)
                # assert target["padded_boxes"].size(0) == self.num_queries  # centered, normalized(0-1)
                # TODO: move the assert to sanity check function

                padded_target["boxes_xyxy"] = box_cxcywh_to_xyxy(target["padded_boxes"]) * image_size_xyxy

                # padded_target["image_size_xyxy"] = target["image_size_xyxy"]
                padded_target["image_size_xyxy_tgt"] = image_size_xyxy.unsqueeze(0).repeat(self.num_queries, 1)
                padded_target["area"] = get_area_of_boxes_xyxy(padded_target["boxes_xyxy"])
                padded_target = move_tensors_to_device(padded_target, self.device)

                USE_PADDED_TARGET = True
                if USE_PADDED_TARGET:
                    target.update(padded_target)

            new_targets.append(target)
        

        # generate diffusion related inputs
        crop_boxes = torch.stack([t['boxes'] for t in new_targets])
        crop_labels = torch.stack([t['labels'] for t in new_targets])
        latent_dict = self.instance_latent_coding(mlvl_feats, crop_boxes, crop_labels)

        return new_targets, latent_dict

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_queries, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        if num_gt < self.num_queries:
            box_placeholder = torch.randn(self.num_queries - num_gt, 4,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_queries:
            select_mask = [True] * self.num_queries + [False] * (num_gt - self.num_queries)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t



    def get_padded_boxes_and_labels(self, gt_bboxes, gt_labels):
        """
        :param gt_boxes: [nr_boxes, 4] (cx, cy, w, h), normalized
        :param gt_labels: [nr_boxes]
        """
        # step1: pad labels
        num_gt = gt_labels.size(0)
        if num_gt < self.num_queries:
            pad_labels = torch.full((self.num_queries - num_gt,), -1.).to(gt_labels)  # pad bg labels: -1
            padded_labels = torch.cat([gt_labels, pad_labels])

            pad_boxes = torch.randn(self.num_queries - num_gt, 4, device=self.device) / 6. + 0.5
            pad_boxes[:, 2:] = torch.clip(pad_boxes[:, 2:], min=1e-4)
            pad_boxes = box_cxcywh_to_xyxy(pad_boxes)
            
            
            
            


        elif num_gt > self.num_queries:
            pass

        else:
            pass            




        padded_dict = dict()

        return padded_dict

    def filter_out_close_boxes(self, gt_boxes, padded_boxes):
        pass

    # forward diffusion process
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def model_predictions(self, backbone_feats, images_whwh, x, t,):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_coord = self.head(backbone_feats, x_boxes, t, None)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord


    # A simple version as starting point
    def instance_latent_coding(self, mlvl_feats, crop_boxes, crop_labels):
        latent_dict = dict()

        # rois (clean)
        # -- crop_boxes: [N, nr_boxes, 4] xyxy
        assert crop_boxes.dim() == 3 and crop_labels.dim() == 2
        N, nr_crops = crop_boxes.shape[:2]
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(crop_boxes[b]))
        roi_features = self.box_pooler(mlvl_feats, proposal_boxes)  # [8000, 256, 7, 7], 8000 = 16 * 500
        roi_features = self.roi_conv(roi_features).squeeze(-1).squeeze(-1)  # [8000, 256]
        roi_features = roi_features.view(N, nr_crops, -1)  # [N, nr_boxes, 256]
        
        # box coding (clean)
        # -- crop_boxes: [N, nr_boxes, 4] xyxy
        # -- crop_labels: [N, nr_boxes]
        box_embeddings = self.box_mlp(crop_boxes)  # [N, nr_boxes, 256]
        label_embeddings = self.label_embedder(crop_labels)  # [N, nr_boxes, 256]

        # ins coding (clean) = fuse(box_coding, rois), serve as gt for decoder
        ins_latents = torch.cat([box_embeddings, roi_features, label_embeddings], dim=-1)  # [N, nr_boxes, 768]
        ins_latents = self.fuse_mlp(ins_latents)  # [N, nr_boxes, 256]
        
        # (optional) normalize ins latents
        if self.normalize_latent:
            ins_latents = F.normalize(ins_latents, dim=-1) * self.scale_latent

        # time_embedding - fix: for a batch
        t = torch.randint(0, self.num_timesteps, (N,), device=self.device).long()
        time_emb = self.time_mlp(t)  # [N, 256]

        # noise - fix: for a batch
        noise = torch.randn_like(ins_latents)

        # ins coding (noisy)
        diffused_ins_latents = self.q_sample(ins_latents, t, noise)

        
        latent_dict['t'] = t
        latent_dict['time_emb'] = time_emb
        latent_dict['noise'] = noise
        latent_dict['gt_ins_latents'] = ins_latents
        latent_dict['diffused_ins_latents'] = diffused_ins_latents

        return latent_dict

    def instance_latent_coding_old(self, mlvl_feats, padded_bboxes, padded_labels, images_whwh):
        # TODO: latent with no gt_bbox and gt_label
        
        latent_dict = dict()
        # TODO:
        # Generate gt_proposal_feat
        # Concat gt_proposal_feat with gt_bboxes and gt_labels
        # Option: 
        # latent_pre_norm (before_decoding)
        # latent_post_norm (after_decoding, only for loss_function)
        # save un_normed_latents and normed_latents together.
        
        # padded_bboxes:    [N, nr_boxes(300), 4], xyxy, Un-normalized
        # padded_labels:    [N, nr_boxes] (-1, 0-79), -1 for background
        # instance_latents: [N, nr_boxes, D]

        N, nr_boxes = padded_bboxes.shape[:2]
        proposal_boxes = list()
        for i in range(N):
            proposal_boxes.append(Boxes(padded_bboxes[i]))
        roi_features = self.box_pooler(mlvl_feats, proposal_boxes)  # [8000, 256, 7, 7], 8000 = 16 * 500
        roi_features = roi_features.view(N, nr_boxes, -1)  # [16, 500, 256*7*7]
        roi_features = self.roi_linear(roi_features)
        roi_features = self.activation(self.norm1(roi_features))  # [16, 500, 256]
        
        scaled_padded_bboxes = padded_bboxes / images_whwh[:, None, :]  # [16, 500, 4]
        one_hot_label = F.one_hot(padded_labels + 1, self.num_classes)  # [16, 500, 81], index 0 for bg
        
        ins_latents = torch.cat([roi_features, scaled_padded_bboxes, one_hot_label], dim=-1)  # [16, 500, 341]
        ins_latents = self.latent_linear(ins_latents)
        ins_latents = self.activation(self.norm2(ins_latents)) # [16, 500, 256]
        
        latent_dict['clean_ins_latents'] = ins_latents

        # Get diffused(noisy) ins latents


        latent_dict['diffused_ins_latents'] = diffused_ins_latents  # TODO


        return latent_dict

    @torch.no_grad()
    def test_with_ddim_sampling(self, batched_inputs, features, images_whwh, images, clip_denoised=True, do_postprocess=True):
        mlvl_feats=features['mlvl_feats']
        mlvl_masks=features['mlvl_masks']
        mlvl_pos_embs=features['mlvl_pos_embs']
        
        batch = images_whwh.shape[0]
        shape = (batch, self.num_queries, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)
        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                         self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                img = torch.cat((img, torch.randn(1, self.num_queries - num_remain, 4, device=img.device)), dim=1)
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        images.image_sizes)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.use_ensemble and self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result = Instances(images.image_sizes[0])
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results