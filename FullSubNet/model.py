# -*- coding: utf-8 -*-

import sys
import os
import toml
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

sys.path.append("./")
from audiolib.audio import offline_laplace_norm, cumulative_laplace_norm
from audiolib.feature import drop_band


class FullSubNet(nn.Layer):
    def __init__(self, config, mode="train"):
        super().__init__()

        # set mode
        self.mode = mode

        # get args
        self.num_freqs = config["num_freqs"]
        self.fullband_num_neighbors = config["fullband_num_neighbors"]
        self.subband_num_neighbors = config["subband_num_neighbors"]
        self.fullband_hidden_size = config["fullband_hidden_size"]
        self.subband_hidden_size = config["subband_hidden_size"]
        self.fullband_num_layers = config["fullband_num_layers"]
        self.subband_num_layers = config["subband_num_layers"]
        self.num_groups_in_drop_band = config["num_groups_in_drop_band"]
        self.dropout = config["dropout"]
        self.look_ahead = config["look_ahead"]

        # fullband net
        self.fullband_seq = nn.LSTM(
            self.num_freqs,
            self.fullband_hidden_size,
            self.fullband_num_layers,
            dropout=self.dropout,
        )
        self.fullband_fc = nn.Linear(self.fullband_hidden_size, self.num_freqs)
        self.fullband_activate = nn.ReLU()

        # subband net
        self.subband_seq = nn.LSTM(
            (self.subband_num_neighbors * 2 + 1) + (self.fullband_num_neighbors * 2 + 1),
            self.subband_hidden_size,
            self.subband_num_layers,
            dropout=self.dropout,
        )
        self.subband_fc = nn.Linear(self.subband_hidden_size, 2)

        # loss
        self.loss = nn.MSELoss()

    def forward(self, noisy_mag):
        # [B, F, T] -> [B, 1, F, T]
        noisy_mag = noisy_mag.unsqueeze(1)
        # pad
        # input [B, C, F, T]
        # [left, right, top, bottom]
        # [left, right] -> T
        # [top, bottom] -> F
        noisy_mag = F.pad(noisy_mag, [0, self.look_ahead, 0, 0])
        # check num_channels
        [batch_size, num_channels, num_freqs, num_frames] = noisy_mag.shape
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        # norm
        if self.mode in ["train", "valid"]:
            fullband_in = offline_laplace_norm(noisy_mag).reshape([batch_size, num_channels * num_freqs, num_frames])
        else:
            fullband_in = cumulative_laplace_norm(noisy_mag).reshape([batch_size, num_channels * num_freqs, num_frames])

        # fullband net
        fullband_out, _ = self.fullband_seq(fullband_in.transpose([0, 2, 1]))
        fullband_out = self.fullband_fc(fullband_out)
        fullband_out = self.fullband_activate(fullband_out)
        fullband_out = fullband_out.transpose([0, 2, 1]).reshape([batch_size, 1, num_freqs, num_frames])

        # Unfold fullband model's output, [B, N=F, C, F_f, T]. N is the number of sub-band units
        fullband_out_unfolded = self.unfold(fullband_out, num_neighbors=self.fullband_num_neighbors)
        fullband_out_unfolded = fullband_out_unfolded.reshape(
            [batch_size, num_freqs, self.fullband_num_neighbors * 2 + 1, num_frames]
        )

        # Unfold noisy spectrogram, [B, N=F, C, F_s, T]
        noisy_mag_unfolded = self.unfold(noisy_mag, num_neighbors=self.subband_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(
            [batch_size, num_freqs, self.subband_num_neighbors * 2 + 1, num_frames]
        )

        # Concatenation, [B, F, (F_s + F_f), T]
        subband_in = paddle.concat([noisy_mag_unfolded, fullband_out_unfolded], axis=2)

        # norm
        if self.mode in ["train", "valid"]:
            subband_in = offline_laplace_norm(subband_in)
        else:
            subband_in = cumulative_laplace_norm(subband_in)

        # Speeding up training without significant performance degradation.
        if batch_size > 1:
            subband_in = drop_band(subband_in.transpose([0, 2, 1, 3]), num_groups=self.num_groups_in_drop_band)
            num_freqs = subband_in.shape[2]
            subband_in = subband_in.transpose([0, 2, 1, 3])  # [B, F//num_groups, (F_s + F_f), T]

        # [B*F, F_s+F_f, T]
        subband_in = subband_in.reshape(
            [
                batch_size * num_freqs,
                (self.subband_num_neighbors * 2 + 1) + (self.fullband_num_neighbors * 2 + 1),
                num_frames,
            ]
        )

        # subband net
        subband_out, _ = self.subband_seq(subband_in.transpose([0, 2, 1]))
        subband_out = self.subband_fc(subband_out)  # [B*F, 2, T]
        subband_out = subband_out.reshape([batch_size, num_freqs, 2, num_frames])  # [B, F, 2, T]
        subband_out = subband_out.transpose([0, 2, 1, 3])  # [B, 2, F, T]

        subband_out = subband_out[:, :, :, self.look_ahead :].transpose([0, 2, 3, 1])
        return subband_out  # [B, F, T, 2]

    @staticmethod
    def unfold(input, num_neighbors):
        """
        Along the frequency axis, this function is used for splitting overlapped sub-band units.

        Args:
            input: four-dimension input.
            num_neighbors: number of neighbors in each side.

        Returns:
            Overlapped sub-band units.

        Shapes:
            input: [B, C, F, T]
            return: [B, N, C, F_s, T]. F_s represents the frequency axis of the sub-band unit, e.g. [2, 161, 1, 19, 200]
        """
        assert input.dim() == 4, f"The dim of the input is {input.dim()}. It should be four dim."
        [batch_size, num_channels, num_freqs, num_frames] = input.shape

        if num_neighbors < 1:  # No change on the input
            return input.transpose([0, 2, 1, 3]).reshape([batch_size, num_freqs, num_channels, 1, num_frames])

        output = input.reshape([batch_size * num_channels, 1, num_freqs, num_frames])
        subband_unit_size = num_neighbors * 2 + 1

        # Pad the top and bottom of the original spectrogram
        output = F.pad(output, [0, 0, num_neighbors, num_neighbors], mode="reflect")  # [B*C, 1, F, T]

        output = F.unfold(output, [subband_unit_size, num_frames])  # [B*C, 1*subband_unit_size*num_frames, F]
        assert output.shape[-1] == num_freqs, f"n_freqs != N (sub_band), {num_freqs} != {output.shape[-1]}"

        # Split the dim of the unfolded feature
        output = output.reshape([batch_size, num_channels, subband_unit_size, num_frames, num_freqs])
        output = output.transpose([0, 4, 1, 2, 3])  # [B, N, C, F_s, T]

        return output


if __name__ == "__main__":
    # config device
    device = paddle.get_device()
    paddle.set_device(device)
    print(f"device {device}")

    # get config
    toml_path = os.path.join(os.path.dirname(__file__), "config.toml")
    config = toml.load(toml_path)

    # config model
    model = FullSubNet(config["model"]["args"])
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
    cIRM = drop_band(
        cIRM.transpose([0, 2, 1, 3]), num_groups=config["model"]["args"]["num_groups_in_drop_band"]
    ).transpose([0, 2, 1, 3])

    # test model and optimizer
    cRM = model(noisy_mag)
    loss = model.loss(cRM, cIRM)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()

    pass
