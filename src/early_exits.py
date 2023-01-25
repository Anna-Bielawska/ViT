import copy
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from torch.nn import functional as F
from typing import Tuple, OrderedDict
from torchvision.models._utils import _ovewrite_named_param
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from torchvision.models import ViT_B_16_Weights, VisionTransformer
from torchvision.models.vision_transformer import ConvStemConfig, WeightsEnum, MLPBlock, EncoderBlock

# source code: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py


class CustomEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )

        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

        self.early_exit_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(768, 102))
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor):
        
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        _input = input + self.pos_embedding
        _input = self.dropout(_input)

        early_classification = []
        for layer, class_head in zip(self.layers, self.early_exit_heads):
            _input = layer(_input)
            early_classification.append(F.softmax(class_head(_input[:, 0]), dim=-1))

        return self.ln(_input), early_classification


class CustomViT(VisionTransformer):
    """
    Overwrite the original ViT. Replace Encoder with CustomEncoder,
    include early exit classification output in the forward method.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        """
        Initialize the ViT model with CustomEncoder
        and appropriate classification heads layer.
        """

        super().__init__(
            image_size,
            patch_size,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            num_classes,
            representation_size,
            norm_layer,
            conv_stem_configs,
        )

        self.encoder = CustomEncoder(
            self.seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.heads = nn.Sequential(
            OrderedDict([("head", nn.Linear(in_features=768, out_features=102, bias=True))])
            )


    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x, early_classif = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x, early_classif


def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
    ) -> VisionTransformer:
    
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    model = CustomViT(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        state_dict = dict(weights.get_state_dict(progress=progress))
        state_dict_copy = copy.deepcopy(state_dict)
        # get around loading weights for a heads layer
        for key in state_dict:
            if "heads" in key:
                state_dict_copy.pop(key)
        del state_dict
        model.load_state_dict(state_dict=state_dict_copy, strict=False)

    return model


def vit_b_16(*, weights: Optional[ViT_B_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale `_.
    Args:
        weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            `_
            for more details about this class.
    .. autoclass:: torchvision.models.ViT_B_16_Weights
        :members:
    """
    weights = ViT_B_16_Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )