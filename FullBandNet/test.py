# -*- coding: utf-8 -*-

import sys
import os
import toml
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pesq import PesqError
import paddle
from paddle.io import DataLoader

sys.path.append("./")
from FullBandNet.model import FullBandNet
from FullBandNet.dataset import DNS_Interspeech_2021_Dataset
from audiolib.mask import get_cIRM, decompress_cIRM
from audiolib.metrics import STOI, WB_PESQ


def test(model, test_set, config):
    print(f"{'=' * 20} test {'=' * 20}")

    # config enh path
    enh_path = config["test_dataset"]["path"]["noisy_clean"]
    enh_path = os.path.abspath(os.path.join(enh_path, "..", "enh"))
    os.makedirs(enh_path, exist_ok=True)

    # get args
    sr = config["train_dataset"]["args"]["sr"]
    n_fft = config["train_dataset"]["args"]["n_fft"]
    win_length = config["train_dataset"]["args"]["win_length"]
    hop_length = config["train_dataset"]["args"]["hop_length"]

    # get test_iter
    test_iter = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    model.eval()
    test_loss_total = 0.0
    metrics = {
        "STOI": [],
        "WB_PESQ": [],
    }
    for idx in tqdm(test_iter, desc="test"):
        # get noisy
        noisy, _ = librosa.load(test_set.noisy_files[idx], sr=sr)
        # get clean
        clean, _ = librosa.load(test_set.clean_files[idx], sr=sr)

        # get cIRM
        noisy_spec = librosa.stft(noisy, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        clean_spec = librosa.stft(clean, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        cIRM = get_cIRM(noisy_spec, clean_spec)

        # to ternsor
        noisy_spec = paddle.to_tensor(noisy_spec)
        cIRM = paddle.to_tensor(cIRM)

        # add batch_size
        noisy_spec = noisy_spec.unsqueeze(0)  # [B, F, T]
        cIRM = cIRM.unsqueeze(0)  # [B, F, T, 2]

        with paddle.no_grad():
            noisy_mag = paddle.abs(noisy_spec).astype(paddle.float32)
            cRM = model(noisy_mag)
            loss = model.loss(cRM, cIRM)

            # cum loss
            test_loss_total += loss.item()

            # decompress cRM
            cRM = decompress_cIRM(cRM)
            # enh spec
            noisy_real = paddle.real(noisy_spec)
            noisy_imag = paddle.imag(noisy_spec)
            enh_spec_real = cRM[..., 0] * noisy_real - cRM[..., 1] * noisy_imag
            enh_spec_imag = cRM[..., 1] * noisy_real + cRM[..., 0] * noisy_imag
            enh_spec = paddle.squeeze(enh_spec_real + 1j * enh_spec_imag, axis=0)
            # to numpy
            enh_spec = enh_spec.numpy()
            # enh
            enh = librosa.istft(enh_spec, win_length=win_length, hop_length=hop_length)
            # save metrics
            metrics["STOI"].append(STOI(clean, enh))
            try:
                metrics["WB_PESQ"].append(WB_PESQ(clean, enh))
            except:
                print(
                    f"WB_PESQ error: {PesqError}, I don't know why, submit issue to https://github.com/ludlows/python-pesq"
                )
            # save enh
            enh_file_path = os.path.basename(test_set.noisy_files[idx]).replace("noisy", "noisy_enh")
            sf.write(os.path.join(enh_path, enh_file_path), enh, sr)

    # get test loss
    test_loss = test_loss_total / len(test_iter)
    print(f"{test_loss:.5f} test loss")
    # get metrics
    stoi_score = np.mean(metrics["STOI"])
    wb_pesq_score = np.mean(metrics["WB_PESQ"])
    print(f"{stoi_score:.5f} STOI {wb_pesq_score:.5f} WB_PESQ")


if __name__ == "__main__":
    # config device
    device = paddle.get_device()
    paddle.set_device(device)
    print(f"device {device}")

    # get config
    toml_path = os.path.join(os.path.dirname(__file__), "config.toml")
    config = toml.load(toml_path)

    # get train dataset dataloader
    num_workers = config["train_dataset"]["dataloader"]["num_workers"]
    drop_last = config["train_dataset"]["dataloader"]["drop_last"]

    # get datasets zip path
    datasets_zip_path = os.path.abspath(config["datasets"]["path"]["zip"])

    # get datasets path
    datasets_path = os.path.splitext(datasets_zip_path)[0]

    # get test_iter
    test_set = DNS_Interspeech_2021_Dataset(datasets_path, config, mode="test")

    # config model
    model = FullBandNet(config["model"]["args"], mode="test")
    model_path = os.path.join(os.path.dirname(__file__), "models")
    for path in Path(model_path).rglob("best_model_*.pdparams"):
        print(path.as_posix())
        ckpt = paddle.load(path.as_posix())
        break
    model.set_state_dict(ckpt)
    print(model)

    # test
    test(model, test_set, config)
    pass
