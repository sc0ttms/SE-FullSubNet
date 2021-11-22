# -*- coding: utf-8 -*-

import sys
import os
import toml
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pesq import PesqError
import paddle
from paddle.io import DataLoader


sys.path.append("./")
from FullBandNet.model import FullBandNet
from FullBandNet.dataset import DNS_Interspeech_2021_Dataset
from audiolib.audio import unzip_dataset
from audiolib.mask import decompress_cIRM
from audiolib.metrics import STOI, WB_PESQ, transform_pesq_range


def train(model, train_iter, valid_iter, config):
    # config n_epoch
    n_epoch = config["train"]["n_epoch"]
    # config fft params
    win_length = config["train_dataset"]["args"]["win_length"]
    hop_length = config["train_dataset"]["args"]["hop_length"]
    # init best_score
    best_score = 0.0

    # config model path
    model_path = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_path, exist_ok=True)

    # config optimizer
    optimizer = getattr(paddle.optimizer, config["train"]["optimizer"])(
        parameters=model.parameters(),
        learning_rate=config["train"]["lr"],
    )

    for epoch in range(0, n_epoch):
        print(f"{'=' * 20} {epoch} epoch {'=' * 20}")

        # train
        model.train()
        train_loss_total = 0.0
        for noisy_spec, cIRM in tqdm(train_iter, desc="train"):
            noisy_mag = paddle.abs(noisy_spec).astype(paddle.float32)
            cRM = model(noisy_mag)
            loss = model.loss(cRM, cIRM)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            # cum loss
            train_loss_total += loss.item()
        # get train loss
        train_loss = train_loss_total / len(train_iter)
        print(f"{epoch} epoch {train_loss:.5f} train loss")

        # save cur model
        # del old model
        for path in Path(model_path).rglob("model_*.pdparams"):
            os.remove(path)
        # save new model
        paddle.save(model.state_dict(), os.path.join(model_path, f"model_{epoch}_{train_loss:.5f}.pdparams"))

        # valid
        model.eval()
        valid_loss_total = 0.0
        metrics = {
            "STOI": [],
            "WB_PESQ": [],
        }
        for noisy_spec, clean, cIRM in tqdm(valid_iter, desc="valid"):
            with paddle.no_grad():
                noisy_mag = paddle.abs(noisy_spec).astype(paddle.float32)
                cRM = model(noisy_mag)
                loss = model.loss(cRM, cIRM)
                # cum loss
                valid_loss_total += loss.item()

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
                clean = clean.detach().squeeze(0).cpu().numpy()
                metrics["STOI"].append(STOI(clean, enh))
                try:
                    metrics["WB_PESQ"].append(WB_PESQ(clean, enh))
                except:
                    print(
                        f"WB_PESQ error: {PesqError}, I don't know why, submit issue to https://github.com/ludlows/python-pesq"
                    )

        # get valid loss
        valid_loss = valid_loss_total / len(valid_iter)
        print(f"{epoch} epoch {valid_loss:.5f} valid loss")
        # get metrics
        stoi_score = np.mean(metrics["STOI"])
        wb_pesq_score = np.mean(metrics["WB_PESQ"])
        print(f"{epoch} epoch {stoi_score:.5f} STOI {wb_pesq_score:.5f} WB_PESQ")
        # check best score
        metrics_score = (stoi_score + transform_pesq_range(wb_pesq_score)) / 2
        if metrics_score > best_score:
            best_score = metrics_score
            # save best model
            # del old model
            for path in Path(model_path).rglob("best_model_*.pdparams"):
                os.remove(path) 
            # save new model
            paddle.save(
                model.state_dict(),
                os.path.join(model_path, f"best_model_{epoch}_{stoi_score:.5f}_{wb_pesq_score:.5f}.pdparams"),
            )


if __name__ == "__main__":
    # config device
    device = paddle.get_device()
    paddle.set_device(device)
    print(f"device {device}")

    # get config
    toml_path = os.path.join(os.path.dirname(__file__), "config.toml")
    config = toml.load(toml_path)

    # get train dataset dataloader
    batch_size = config["train_dataset"]["dataloader"]["batch_size"]
    num_workers = config["train_dataset"]["dataloader"]["num_workers"]
    drop_last = config["train_dataset"]["dataloader"]["drop_last"]

    # get datasets zip path
    datasets_zip_path = os.path.abspath(config["datasets"]["path"]["zip"])

    # extract datasets zip
    datasets_path = os.path.splitext(datasets_zip_path)[0]
    if not os.path.exists(datasets_path):
        print("extract ing......")
        datasets_path = unzip_dataset(datasets_zip_path)
    print(f"datasets path {datasets_path}")

    # get train_iter
    train_set = DNS_Interspeech_2021_Dataset(datasets_path, config, mode="train")
    train_iter = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    # get valid_iter
    valid_set = DNS_Interspeech_2021_Dataset(datasets_path, config, mode="valid")
    valid_iter = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    # config model
    model = FullBandNet(config["model"]["args"])
    print(model)

    # load old model
    model_path = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_path, exist_ok=True)
    old_model = os.path.join(model_path, "best_model_0_0.84745_2.08731.pdparams")
    if os.path.exists(old_model):
        ckpt = paddle.load(old_model)
        model.set_state_dict(ckpt)

    train(model, train_iter, valid_iter, config)

    pass
