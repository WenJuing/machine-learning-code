"""
original code from apple:
https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit_v2.py
"""

from torch import nn, Tensor
from typing import Dict, Tuple, Optional, Any
from model_config import get_config
from layers import ConvLayer, LinearLayer, GlobalPool, Identity
from modules import InvertedResidual
from modules import MobileViTBlockv2 as Block
from init_utils import initialize_weights
from torch.utils.checkpoint import checkpoint as gradient_checkpoint_fn


# @register_cls_models("mobilevit_v2")
class MobileViTv2(nn.Module):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ architecture
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__()
        num_classes = getattr(opts, "num_classes", 1000)
        pool_type = getattr(opts, "model.layer.global_pool", "mean")

        mobilevit_config = get_config(opts=opts)
        image_channels = mobilevit_config["layer0"]["img_channels"]
        out_channels = mobilevit_config["layer0"]["out_channels"]

        self.dilation = 1
        output_stride = kwargs.get("output_stride", None)
        self.dilate_l4 = False
        self.dilate_l5 = False
        if output_stride == 8:
            self.dilate_l4 = True
            self.dilate_l5 = True
        elif output_stride == 16:
            self.dilate_l5 = True
        
        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=mobilevit_config["layer4"],
            dilate=self.dilate_l4,
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=mobilevit_config["layer5"],
            dilate=self.dilate_l5,
        )
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": out_channels,
            "out": out_channels,
        }

        self.classifier = nn.Sequential(
            GlobalPool(pool_type=pool_type, keep_dim=False),
            LinearLayer(in_features=out_channels, out_features=num_classes, bias=True),
        )
        
        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    def _make_layer(
        self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False
    ) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts, input_channel=input_channel, cfg=cfg, dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts, input_channel=input_channel, cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(
        opts, input_channel: int, cfg: Dict
    ) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(
        self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False
    ) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation,
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        attn_unit_dim = cfg["attn_unit_dim"]
        ffn_multiplier = cfg.get("ffn_multiplier")

        dropout = getattr(opts, "model.classification.mitv2.dropout", 0.0)

        block.append(
            Block(
                opts=opts,
                in_channels=input_channel,
                attn_unit_dim=attn_unit_dim,
                ffn_multiplier=ffn_multiplier,
                n_attn_blocks=cfg.get("attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=dropout,
                ffn_dropout=getattr(
                    opts, "model.classification.mitv2.ffn_dropout", 0.0
                ),
                attn_dropout=getattr(
                    opts, "model.classification.mitv2.attn_dropout", 0.0
                ),
                conv_ksize=3,
                attn_norm_layer=getattr(
                    opts, "model.classification.mitv2.attn_norm_layer", "layer_norm_2d"
                ),
                dilation=self.dilation,
            )
        )

        return nn.Sequential(*block), input_channel
    
    def check_model(self):
        assert (
            self.model_conf_dict
        ), "Model configuration dictionary should not be empty"
        assert self.conv_1 is not None, "Please implement self.conv_1"
        assert self.layer_1 is not None, "Please implement self.layer_1"
        assert self.layer_2 is not None, "Please implement self.layer_2"
        assert self.layer_3 is not None, "Please implement self.layer_3"
        assert self.layer_4 is not None, "Please implement self.layer_4"
        assert self.layer_5 is not None, "Please implement self.layer_5"
        assert self.conv_1x1_exp is not None, "Please implement self.conv_1x1_exp"
        assert self.classifier is not None, "Please implement self.classifier"

    def reset_parameters(self, opts):
        """Initialize model weights"""
        initialize_weights(opts=opts, modules=self.modules())

    def _forward_layer(self, layer: nn.Module, x: Tensor) -> Tensor:
        # Larger models with large input image size may not be able to fit into memory.
        # We can use gradient checkpointing to enable training with large models and large inputs
        return layer(x)
        
    def _extract_features(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self._forward_layer(self.conv_1, x)
        x = self._forward_layer(self.layer_1, x)
        x = self._forward_layer(self.layer_2, x)
        x = self._forward_layer(self.layer_3, x)

        x = self._forward_layer(self.layer_4, x)
        x = self._forward_layer(self.layer_5, x)
        x = self._forward_layer(self.conv_1x1_exp, x)
        return x
    
    def _forward_classifier(self, x: Tensor, *args, **kwargs) -> Tensor:
        # We add another classifier function so that the classifiers
        # that do not adhere to the structure of BaseEncoder can still
        # use neural augmentor
        x = self._extract_features(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Any, *args, **kwargs) -> Any:
        x = self._forward_classifier(x, *args, **kwargs)
        return x
        
def mobile_vit_v2(opts):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt
    m = MobileViTv2(opts)
    return m