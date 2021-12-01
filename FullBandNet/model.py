# -*- coding: utf-8 -*-

import sys
import os
import toml
import paddle
import paddle.nn as nn
from paddle.framework import ParamAttr
from paddle.nn.initializer import XavierNormal, Normal


sys.path.append(os.getcwd())
from audio.feature import offline_laplace_norm, cumulative_laplace_norm


class FullBandNet(nn.Layer):
    def __init__(self, config, mode="train"):
        super().__init__()

        # set mode
        self.mode = mode

        # get args
        self.num_freqs = config["num_freqs"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.look_ahead = config["look_ahead"]

        # net
        self.seq = nn.LSTM(
            self.num_freqs,
            self.hidden_size,
            self.num_layers,
            dropout=self.dropout,
            weight_ih_attr=ParamAttr(initializer=XavierNormal()),
            weight_hh_attr=ParamAttr(initializer=XavierNormal()),
            bias_ih_attr=ParamAttr(initializer=Normal()),
            bias_hh_attr=ParamAttr(initializer=Normal()),
        )
        self.fc = nn.Linear(
            self.hidden_size,
            self.num_freqs * 2,
            weight_attr=ParamAttr(initializer=XavierNormal()),
            bias_attr=ParamAttr(initializer=Normal()),
        )

        # loss
        self.loss = nn.MSELoss()

    def forward(self, noisy_mag):
        # [B, F, T] -> [B, 1, F, T]
        noisy_mag = noisy_mag.unsqueeze(1)
        # pad
        noisy_mag = nn.functional.pad(noisy_mag, [0, self.look_ahead, 0, 0])
        # check num_channels
        [batch_size, num_channels, num_freqs, num_frames] = noisy_mag.shape
        assert num_channels == 1

        # norm
        if self.mode in ["train", "valid"]:
            noisy_mag = offline_laplace_norm(noisy_mag).reshape([batch_size, num_channels * num_freqs, num_frames])
        else:
            noisy_mag = cumulative_laplace_norm(noisy_mag).reshape([batch_size, num_channels * num_freqs, num_frames])
        # net
        noisy_mag = noisy_mag.transpose([0, 2, 1])
        enh_noisy, _ = self.seq(noisy_mag)
        enh_noisy = self.fc(enh_noisy)
        enh_noisy = enh_noisy.transpose([0, 2, 1]).reshape([batch_size, 2, num_freqs, num_frames])
        enh_noisy = enh_noisy[:, :, :, self.look_ahead :].transpose([0, 2, 3, 1])
        return enh_noisy  # [B, F, T, 2]


if __name__ == "__main__":
    # config device
    device = paddle.get_device()
    paddle.set_device(device)
    print(f"device {device}")

    # get config
    toml_path = os.path.join(os.path.dirname(__file__), "config.toml")
    config = toml.load(toml_path)
    # get train args
    use_amp = False if device == "cpu" else config["train"]["use_amp"]
    clip_grad_norm_value = config["train"]["clip_grad_norm_value"]

    # config model
    model = FullBandNet(config["model"])
    print(model)

    # config optimizer
    optimizer = getattr(paddle.optimizer, config["train"]["optimizer"])(
        parameters=model.parameters(),
        learning_rate=config["train"]["lr"],
        grad_clip=nn.ClipGradByNorm(clip_norm=clip_grad_norm_value),
    )

    # scaler
    scaler = paddle.amp.GradScaler()

    # gen test data
    noisy_mag = paddle.randn([3, config["model"]["num_freqs"], 10])
    cIRM = paddle.randn([3, config["model"]["num_freqs"], 10, 2])

    # test model and optimizer
    with paddle.amp.auto_cast(enable=use_amp):
        cRM = model(noisy_mag)
        loss = model.loss(cRM, cIRM)
    scaled = scaler.scale(loss)
    scaled.backward()
    scaler.minimize(optimizer, scaled)
    optimizer.clear_grad()

    pass
