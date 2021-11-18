# -*- coding: utf-8 -*-

from pystoi.stoi import stoi
from pesq import pesq


def STOI(ref, est, sr=16000):
    return stoi(ref, est, sr, extended=False)


def WB_PESQ(ref, est, sr=16000):
    return pesq(sr, ref, est, "wb")


def transform_pesq_range(pesq_score):
    """
    transform PESQ metric range from [-0.5 ~ 4.5] to [0 ~ 1]
    """
    return (pesq_score + 0.5) / 5
