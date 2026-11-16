"""Explicit attention variants shared by training, evaluation and deployment."""

import torch
from torch import nn
from torch.nn import functional as functional

from simple_network import IncepAttentionCNN_rgb


VARIANTS = {
    "panorama192x1800": {"input_hw": [192, 1800], "parameters": 10820},
    "nadir384x384": {"input_hw": [384, 384], "parameters": 10628},
}


def model_contract(config):
    name = config.get("network_variant", "panorama192x1800")
    if name not in VARIANTS:
        raise ValueError("Unknown attention variant: " + str(name))
    return dict(VARIANTS[name], variant=name)


class NadirAttention384(IncepAttentionCNN_rgb):
    """Original convolution/attention stack with square input and a 196-wide head."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4 * 7 * 7, 16)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = functional.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False)
        x = torch.tanh(self.inception1(x))
        x = torch.tanh(self.inception2(x))
        x = torch.tanh(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        x = x.flatten(1)
        x = torch.tanh(self.fc(x))
        return self.fc2(x)


def create_attention(config):
    contract = model_contract(config)
    model = (
        NadirAttention384() if contract["variant"] == "nadir384x384" else IncepAttentionCNN_rgb()
    )
    if sum(p.numel() for p in model.parameters()) != contract["parameters"]:
        raise ValueError("Attention parameter count changed")
    return model


def load_attention(weights, config, device="cpu"):
    model = create_attention(config).to(device).eval()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model
