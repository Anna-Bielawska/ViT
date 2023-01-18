import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class BasicViT(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.vit = self.with_freezed_params(vit_b_16(weights =  ViT_B_16_Weights.IMAGENET1K_V1))
        self.vit.heads = nn.Linear(in_features=768, out_features=kwargs.get('out_heads', 102), bias=True)

    def forward(self, x):
        return self.vit(x)

    def with_freezed_params(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model