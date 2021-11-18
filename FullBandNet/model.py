# -*- coding: utf-8 -*-

import sys
import os
import toml
import paddle
import paddle.nn as nn

sys.path.append("./")
from audiolib.audio import offline_laplace_norm, cumulative_laplace_norm


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
        self.seq = nn.LSTM(self.num_freqs, self.hidden_size, self.num_layers, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.num_freqs * 2)

        # loss
        self.loss = nn.MSELoss()

    def forward(self, noisy_mag):
        # [B, F, T] -> [B, 1, F, T]
        noisy_mag = noisy_mag.unsqueeze(1)
        # pad
        noisy_mag = nn.functional.pad(noisy_mag, [0, 0, 0, 0, 0, 0, 0, self.look_ahead])
        # check num_channels
        [batch_size, num_channels, num_freqs, num_frames] = noisy_mag.shape
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

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

    # config model
    model = FullBandNet(config["model"]["args"])
    print(model)

    # config optimizer
    optimizer = getattr(paddle.optimizer, config["train"]["optimizer"])(
        parameters=model.parameters(),
        learning_rate=config["train"]["lr"],
    )
    print(optimizer)

    # gen test data
    noisy_mag = paddle.randn([3, config["model"]["args"]["num_freqs"], 10])  # [B, F, T]
    cIRM = paddle.randn([3, config["model"]["args"]["num_freqs"], 10, 2])  # [B, F, T, 2]

    # test model and optimizer
    cRM = model(noisy_mag)
    loss = model.loss(cRM, cIRM)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()

    pass
