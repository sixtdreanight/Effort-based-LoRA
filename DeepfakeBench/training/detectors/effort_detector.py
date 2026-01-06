import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from detectors import DETECTOR
from metrics.base_metrics_class import calculate_metrics_for_train

logger = logging.getLogger(__name__)


# ---------------- SVD Main ----------------
class SVDMainOnlyLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True,
                 init_weight=None, init_bias=None):
        super().__init__()
        self.r = min(r, in_features, out_features)

        self.register_buffer(
            "weight_main", torch.zeros(out_features, in_features)
        )
        self.register_buffer(
            "bias", torch.zeros(out_features) if bias else None
        )

        if init_weight is not None:
            self._init_svd(init_weight, init_bias)

    def _init_svd(self, weight, bias=None):
        U, S, Vh = torch.linalg.svd(weight.double(), full_matrices=False)
        U, S, Vh = U[:, :self.r], S[:self.r], Vh[:self.r]
        self.weight_main.copy_((U @ torch.diag(S) @ Vh).to(weight.dtype))
        if self.bias is not None and bias is not None:
            self.bias.copy_(bias)

    def forward(self, x):
        return F.linear(x, self.weight_main, self.bias)


# ---------------- LoRA ----------------
class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=16, dropout=0.1):
        super().__init__()
        self.base = base
        self.scale = alpha / rank

        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base.weight_main.size(1)
        out_f = base.weight_main.size(0)

        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_f, rank) * 0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale


# ---------------- Apply SVD + LoRA ----------------
def apply_svd_main_and_lora(
    model, svd_r=1023, lora_rank=4, lora_alpha=12, lora_dropout=0.1
):
    target = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    replaced = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target):
            parent_name, layer_name = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parent_name]

            svd = SVDMainOnlyLinear(
                module.in_features,
                module.out_features,
                svd_r,
                bias=module.bias is not None,
                init_weight=module.weight.data,
                init_bias=module.bias.data if module.bias is not None else None,
            )
            setattr(parent, layer_name,
                    LoRALinear(svd, lora_rank, lora_alpha, lora_dropout))
            replaced += 1

    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)

    return model, replaced


# ---------------- Detector ----------------
@DETECTOR.register_module(module_name="effort")
class EffortDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone, _ = self.build_backbone(config)
        self.head = nn.Linear(1024, 2)
        self.loss_func = nn.CrossEntropyLoss()

    def build_backbone(self, config):
        clip = CLIPModel.from_pretrained(config["clip_path"])
        lora_cfg = config.get("lora", {})

        if lora_cfg.get("enable", False):
            vision, n = apply_svd_main_and_lora(
                clip.vision_model,
                lora_cfg.get("svd_r", 1023),
                lora_cfg.get("rank", 4),
                lora_cfg.get("alpha", 12),
                lora_cfg.get("dropout", 0.1),
            )
            logger.info(f"SVD + LoRA enabled | replaced {n} layers")
        else:
            vision = clip.vision_model

        return vision, None

    def forward(self, data_dict):
        feat = self.backbone(data_dict["image"])["pooler_output"]
        pred = self.head(feat)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {"cls": pred, "prob": prob, "feat": feat}

    def get_losses(self, data_dict, pred_dict):
        label = data_dict["label"]
        loss = self.loss_func(pred_dict["cls"], label)
        return {"overall": loss}

    def get_train_metrics(self, data_dict, pred_dict):
        return dict(zip(
            ["auc", "eer", "acc", "ap"],
            calculate_metrics_for_train(
                data_dict["label"].detach(),
                pred_dict["cls"].detach()
            )
        ))
