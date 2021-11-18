# -*- coding: utf-8 -*-

import librosa
import numpy as np
import paddle

EPS = np.finfo(np.float32).eps
np.random.seed(0)


def compress_cIRM(mask, K=10, C=0.1):
    mask = -100 * (mask <= -100) + mask * (mask > -100)
    mask = K * (1 - np.exp(-C * mask)) / (1 + np.exp(-C * mask))
    return mask.astype(np.float32)


def decompress_cIRM(mask, K=10, limit=9.9):
    if paddle.is_tensor(mask):
        mask = (
            limit * (mask >= limit).astype(paddle.float32)
            - limit * (mask <= -limit).astype(paddle.float32)
            + mask * (paddle.abs(mask) < limit).astype(paddle.float32)
        )
        mask = -K * paddle.log((K - mask) / (K + mask))
        return mask.astype(paddle.float32)
    else:
        mask = limit * (mask >= limit) - limit * (mask <= -limit) + mask * (np.abs(mask) < limit)
        mask = -K * np.log((K - mask) / (K + mask))
        return mask.astype(np.float32)


def get_cIRM(noisy_spec, clean_spec):
    denominator = (np.real(noisy_spec) ** 2 + np.imag(clean_spec) ** 2) + EPS

    cIRM_real = (np.real(noisy_spec) * np.real(clean_spec) + np.imag(noisy_spec) * np.imag(clean_spec)) / denominator
    cIRM_imag = (np.real(noisy_spec) * np.imag(clean_spec) - np.imag(noisy_spec) * np.real(clean_spec)) / denominator
    cIRM = np.stack((cIRM_real, cIRM_imag), axis=-1)

    return compress_cIRM(cIRM)  # [F, T, 2]
