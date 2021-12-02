# -*- coding: utf-8 -*-

import numpy as np
from pystoi.stoi import stoi
from pesq import pesq
from pesq import PesqError


def SI_SDR(noisy, clean, sr=16000):
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
        noisy: numpy.ndarray, [..., T]
        clean: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    References
        SDR– Half- Baked or Well Done? (http://www.merl.com/publications/docs/TR2019-013.pdf)
    """
    noisy, clean = np.broadcast_arrays(noisy, clean)
    clean_energy = np.sum(clean ** 2, axis=-1, keepdims=True)

    optimal_scaling = np.sum(clean * noisy, axis=-1, keepdims=True) / clean_energy

    projection = optimal_scaling * clean

    noise = noisy - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)


def STOI(noisy, clean, sr=16000):
    """STOI

    Args:
        clean (array): clean data
        noisy (array): noisy data
        sr (int, optional): sample rate. Defaults to 16000.

    Returns:
        score: stoi score
    """
    return stoi(clean, noisy, sr, extended=False)


def WB_PESQ(noisy, clean, sr=16000):
    """WB_PESQ

    Args:
        clean (array): clean data
        noisy (array): noisy data
        sr (int, optional): sample rate. Defaults to 16000.

    Returns:
        score: wb pesq score
    """
    try:
        return pesq(sr, clean, noisy, "wb")
    except PesqError as e:
        print(f"submit issue to https://github.com/ludlows/python-pesq")


def NB_PESQ(noisy, clean, sr=8000):
    """NB_PESQ

    Args:
        clean (array): clean data
        noisy (array): noisy data
        sr (int, optional): sample rate. Defaults to 8000.

    Returns:
        score: wb pesq score
    """
    try:
        return pesq(sr, clean, noisy, "nb")
    except PesqError as e:
        print(f"submit issue to https://github.com/ludlows/python-pesq")


def transform_pesq_range(pesq_score):
    """transform PESQ metric range from [-0.5 ~ 4.5] to [0 ~ 1]
    Args:
        pesq_score (float): pesq score

    Returns:
        pesq_score: transformed pesq score
    """
    return (pesq_score + 0.5) / 5
