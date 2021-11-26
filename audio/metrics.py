# -*- coding: utf-8 -*-

from pystoi.stoi import stoi
from pesq import pesq
from pesq import PesqError


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


def transform_pesq_range(pesq_score):
    """transform PESQ metric range from [-0.5 ~ 4.5] to [0 ~ 1]
    Args:
        pesq_score (float): pesq score

    Returns:
        pesq_score: transformed pesq score
    """
    return (pesq_score + 0.5) / 5
