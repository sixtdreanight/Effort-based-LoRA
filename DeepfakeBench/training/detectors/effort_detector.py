import os
import math
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

import loralib as lora
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Method 1: Original Effort (SVD with residual learning)
# =============================================================================
@DETECTOR.register_module(module_name='effort')
class EffortDetector(nn.Module):
    def __init__(self, config=None):
        super(EffortDetector, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(1024, 2)
        self.loss_func = nn.CrossEntropyLoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        # Load CLIP model from config
        clip_path = config.get('clip_path', "../models--openai--clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained(clip_path)
        
        # Get SVD configuration from method_config
        method_config = config.get('method_config', {})
        if method_config.get('type') == 'svd':
            svd_config = method_config.get('svd', {})
            r = svd_config.get('svd_r', 1023)
        else:
            r = 1023  # default
        
        # Apply SVD to self_attn layers
        clip_model.vision_model = apply_svd_residual_to_self_attn(clip_model.vision_model, r=r)
        
        return clip_model.vision_model

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
    
        loss_cls = self.loss_func(pred, label)
    
        config = self.config.get('method_config', {})
        svd_config = config.get('svd', {})
        lambda_orth = svd_config.get('lambda_orth', 1.0)
        lambda_ksv = svd_config.get('lambda_ksv', 1.0)
    
        loss_orth, loss_ksv = 0.0, 0.0
        num_svd_modules = 0
    
        for module in self.backbone.modules():
            if isinstance(module, SVDResidualLinear):
                loss_orth += module.compute_orthogonal_loss()
                loss_ksv += module.compute_keepsv_loss()
                num_svd_modules += 1
    
        if num_svd_modules > 0:
            loss_orth = loss_orth / num_svd_modules
            loss_ksv = loss_ksv / num_svd_modules
            loss_reg = lambda_orth * loss_orth + lambda_ksv * loss_ksv
        else:
            loss_reg = 0.0
    
        loss_total = loss_cls + loss_reg
    
        mask_real = label == 0
        mask_fake = label == 1

        loss_real = self.loss_func(pred[mask_real], label[mask_real]) if mask_real.sum() > 0 else torch.tensor(0.0, device=pred.device)
        loss_fake = self.loss_func(pred[mask_fake], label[mask_fake]) if mask_fake.sum() > 0 else torch.tensor(0.0, device=pred.device)

        return {
            'overall': loss_total,
            'cls_loss': loss_cls,
            'orth_loss': loss_orth,
            'ksv_loss': loss_ksv,
            'real_loss': loss_real,
            'fake_loss': loss_fake,
        }

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': features}


# =============================================================================
# Method 2: SVD + LoRA (Initialize with SVD, train with LoRA only)
# =============================================================================
@DETECTOR.register_module(module_name='effort_svd_lora')
class EffortSVDLoRADetector(nn.Module):
    def __init__(self, config=None):
        super(EffortSVDLoRADetector, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(1024, 2)
        self.loss_func = nn.CrossEntropyLoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        # Load CLIP model from config
        clip_path = config.get('clip_path', "../models--openai--clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained(clip_path)
        
        # Get SVD+LoRA configuration from method_config
        method_config = config.get('method_config', {})
        if method_config.get('type') == 'lora':
            svd_lora_config = method_config.get('svd_lora', {})
            r = svd_lora_config.get('svd_r', 1023)
            lora_rank = svd_lora_config.get('lora_rank', 4)
            lora_alpha = svd_lora_config.get('lora_alpha', 12)
            lora_dropout = svd_lora_config.get('lora_dropout', 0.1)
        else:
            r = 1023
            lora_rank = 4
            lora_alpha = 12
            lora_dropout = 0.1
        
        # Apply SVD initialization and add LoRA
        clip_model.vision_model = apply_svd_lora_to_self_attn(
            clip_model.vision_model, 
            r=r,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        return clip_model.vision_model

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)

        # Separate real/fake losses
        mask_real = label == 0
        mask_fake = label == 1

        if mask_real.sum() > 0:
            pred_real = pred[mask_real]
            label_real = label[mask_real]
            loss_real = self.loss_func(pred_real, label_real)
        else:
            loss_real = torch.tensor(0.0, device=pred.device)

        if mask_fake.sum() > 0:
            pred_fake = pred[mask_fake]
            label_fake = label[mask_fake]
            loss_fake = self.loss_func(pred_fake, label_fake)
        else:
            loss_fake = torch.tensor(0.0, device=pred.device)

        return {
            'overall': loss,
            'real_loss': loss_real,
            'fake_loss': loss_fake,
        }

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': features}


# =============================================================================
# Method 3: Pure LoRA
# =============================================================================
@DETECTOR.register_module(module_name='effort_lora')
class EffortLoRADetector(nn.Module):
    def __init__(self, config=None):
        super(EffortLoRADetector, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(1024, 2)
        self.loss_func = nn.CrossEntropyLoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        # Load CLIP model from config
        clip_path = config.get('clip_path', "../models--openai--clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained(clip_path)
        
        # Get LoRA configuration from method_config
        method_config = config.get('method_config', {})
        if method_config.get('type') == 'lora':
            lora_config = method_config.get('lora', {})
            lora_rank = lora_config.get('lora_rank', 4)
            lora_alpha = lora_config.get('lora_alpha', 12)
            lora_dropout = lora_config.get('lora_dropout', 0.1)
        else:
            lora_rank = 4
            lora_alpha = 12
            lora_dropout = 0.1
        
        # Apply LoRA
        clip_model.vision_model = apply_lora_to_self_attn(
            clip_model.vision_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        return clip_model.vision_model

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)

        # Separate real/fake losses
        mask_real = label == 0
        mask_fake = label == 1

        if mask_real.sum() > 0:
            pred_real = pred[mask_real]
            label_real = label[mask_real]
            loss_real = self.loss_func(pred_real, label_real)
        else:
            loss_real = torch.tensor(0.0, device=pred.device)

        if mask_fake.sum() > 0:
            pred_fake = pred[mask_fake]
            label_fake = label[mask_fake]
            loss_fake = self.loss_func(pred_fake, label_fake)
        else:
            loss_fake = torch.tensor(0.0, device=pred.device)

        return {
            'overall': loss,
            'real_loss': loss_real,
            'fake_loss': loss_fake,
        }

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': features}


# =============================================================================
# Custom Modules
# =============================================================================
class SVDResidualLinear(nn.Module):
    """Original Effort: SVD with trainable residual components"""
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        # Original weights (fixed - principal components)
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def compute_current_weight(self):
        if hasattr(self, 'S_residual') and self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main

    def forward(self, x):
        if hasattr(self, 'S_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main

        return F.linear(x, weight, self.bias)
    
    def compute_orthogonal_loss(self):
        if hasattr(self, 'S_residual') and self.S_residual is not None:
        
            U_hat = torch.cat((self.U_r, self.U_residual), dim=1) 
            V_hat = torch.cat((self.V_r, self.V_residual), dim=0) 
        
            UUT = torch.mm(U_hat.t(), U_hat)
            VVT = torch.mm(V_hat.t(), V_hat)
        
            I_U = torch.eye(UUT.size(0), device=UUT.device)
            I_V = torch.eye(VVT.size(0), device=VVT.device)
        
            loss_orth = torch.norm(UUT - I_U, p='fro')**2 + torch.norm(VVT - I_V, p='fro')**2
        else:
            loss_orth = 0.0
        return loss_orth

    def compute_keepsv_loss(self):
        if hasattr(self, 'weight_original_fnorm') and self.weight_original_fnorm is not None:
        
            current_weight = self.compute_current_weight()
            current_fnorm_sq = torch.norm(current_weight, p='fro')**2
            original_fnorm_sq = self.weight_original_fnorm**2
            loss_ksv = torch.abs(current_fnorm_sq - original_fnorm_sq)
        else:
            loss_ksv = 0.0
        return loss_ksv


class SVDLoRALinear(nn.Module):
    """Method 2: SVD initialization + LoRA (residual set to 0)"""
    def __init__(self, in_features, out_features, r, lora_rank=4, lora_alpha=12, lora_dropout=0.1, bias=True, init_weight=None):
        super(SVDLoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank

        # Principal components only (fixed, residual dropped)
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # LoRA parameters (trainable)
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # LoRA dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Start with frozen principal components
        weight = self.weight_main
        
        # Add LoRA adaptation
        if self.lora_rank > 0:
            lora_weight = self.lora_B @ self.lora_A
            weight = weight + self.scaling * lora_weight

        return F.linear(x, weight, self.bias)


class LoRALinear(nn.Module):
    """Method 3: Pure LoRA on frozen original weights"""
    def __init__(self, in_features, out_features, lora_rank=4, lora_alpha=12, lora_dropout=0.1, bias=True, init_weight=None):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank

        # Original weights (frozen)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # LoRA parameters (trainable)
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # LoRA dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Start with frozen base weight
        weight = self.weight
        
        # Add LoRA adaptation
        if self.lora_rank > 0:
            lora_weight = self.lora_B @ self.lora_A
            weight = weight + self.scaling * lora_weight

        return F.linear(x, weight, self.bias)


# =============================================================================
# Functions to replace nn.Linear in the model
# =============================================================================

def apply_svd_residual_to_self_attn(model, r):
    for name, module in model.named_children():
        if 'self_attn' in name:
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r))
        else:
            apply_svd_residual_to_self_attn(module, r)
    
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False 
    return model

def apply_svd_lora_to_self_attn(model, r, lora_rank=4, lora_alpha=12, lora_dropout=0.1):
    """Replace nn.Linear with SVDLoRALinear in self_attn modules"""
    for name, module in model.named_children():
        if 'self_attn' in name:
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    setattr(parent_module, sub_module_names[-1], 
                           replace_with_svd_lora(sub_module, r, lora_rank, lora_alpha, lora_dropout))
        else:
            apply_svd_lora_to_self_attn(module, r, lora_rank, lora_alpha, lora_dropout)
    
    # Set requires_grad
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['lora_A', 'lora_B', 'bias']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def apply_lora_to_self_attn(model, lora_rank=4, lora_alpha=12, lora_dropout=0.1):
    """Replace nn.Linear with LoRALinear in self_attn modules"""
    for name, module in model.named_children():
        if 'self_attn' in name:
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    setattr(parent_module, sub_module_names[-1], 
                           replace_with_lora(sub_module, lora_rank, lora_alpha, lora_dropout))
        else:
            apply_lora_to_self_attn(module, lora_rank, lora_alpha, lora_dropout)
    
    # Set requires_grad
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['lora_A', 'lora_B', 'bias']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


# =============================================================================
# Replacement functions
# =============================================================================

def replace_with_svd_residual(module, r):
    """Replace nn.Linear with SVDResidualLinear"""
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        new_module.weight_original_fnorm = torch.norm(module.weight.data, p='fro')

        # Perform SVD
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)
        r = min(r, len(S))

        # Principal components (fixed)
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        weight_main = U_r @ torch.diag(S_r) @ Vh_r
        new_module.weight_main.data.copy_(weight_main)

        # Residual components (trainable)
        U_residual = U[:, r:]
        S_residual = S[r:]
        Vh_residual = Vh[r:, :]

        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())
            
            new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
            new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
            new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None
            new_module.S_r = None
            new_module.U_r = None
            new_module.V_r = None

        return new_module
    return module


def replace_with_svd_lora(module, r, lora_rank=4, lora_alpha=12, lora_dropout=0.1):
    """Replace nn.Linear with SVDLoRALinear"""
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        new_module = SVDLoRALinear(
            in_features, out_features, r, 
            lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias=bias, init_weight=module.weight.data.clone()
        )

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        # Perform SVD to get principal components only (residual set to 0)
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)
        r = min(r, len(S))

        # Principal components (fixed)
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        weight_main = U_r @ torch.diag(S_r) @ Vh_r
        new_module.weight_main.data.copy_(weight_main)

        return new_module
    return module


def replace_with_lora(module, lora_rank=4, lora_alpha=12, lora_dropout=0.1):
    """Replace nn.Linear with LoRALinear"""
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        new_module = LoRALinear(
            in_features, out_features,
            lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias=bias, init_weight=module.weight.data.clone()
        )

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        return new_module
    return module
