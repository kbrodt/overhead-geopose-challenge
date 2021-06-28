from typing import List

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.base.modules import Activation, Flatten
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.unet.decoder import UnetDecoder


class UnetVFLOW(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        attention_type=None,  # "scse"
        use_city=False,
        to_log=False,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        if use_city:
            enc_out_channels = self.encoder.out_channels[:-1] + (
                self.encoder.out_channels[-1] + 5,
            )  # city ohe dim
        else:
            enc_out_channels = self.encoder.out_channels

        self.decoder = UnetDecoder(
            encoder_channels=enc_out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=attention_type,
        )

        self.xydir_head = EncoderRegressionHead(
            in_channels=self.encoder.out_channels[-1],
            out_channels=2,
        )

        self.scale_head = EncoderRegressionHead(
            in_channels=self.encoder.out_channels[-1],
            out_channels=1,
        )

        self.height_head = nn.Conv2d(
            in_channels=decoder_channels[-1],
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

        self.name = "u-{}".format(encoder_name)
        self.initialize()
        self.to_log = to_log

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.xydir_head)
        init.initialize_head(self.height_head)
        init.initialize_head(self.scale_head)

    def forward(self, x):
        use_city = isinstance(x, tuple)
        if use_city:
            x, city, gsd = x

        features = self.encoder(x)
        xydir = self.xydir_head(features[-1])
        scale = self.scale_head(features[-1])

        if use_city:
            size = features[-1].size(2)
            features[-1] = torch.cat(
                [
                    features[-1],
                    nn.functional.interpolate(city, size=(size, size), mode="nearest"),
                    nn.functional.interpolate(gsd, size=(size, size), mode="nearest"),
                ],
                dim=1,
            )

        decoder_output = self.decoder(*features)

        height = self.height_head(decoder_output)
        mag = scale.unsqueeze(2).unsqueeze(3) * height

        scale = scale.squeeze(1)

        return xydir, height, mag, scale

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


class EncoderRegressionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        pool = nn.AdaptiveAvgPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=0.5, inplace=True)
        linear = nn.Linear(in_channels, out_channels, bias=True)
        super().__init__(pool, flatten, dropout, linear)
