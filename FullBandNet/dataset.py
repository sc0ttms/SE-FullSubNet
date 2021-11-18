# -*- encoding: utf-8 -*-

import sys
import os
import toml
import zipfile
import librosa
import pandas as pd
from tqdm import tqdm
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader

sys.path.append("./")
from audiolib.mask import get_cIRM
from audiolib.audio import sub_sample, select_file, snr_mix


def unzip_dataset(path):
    extract_dir = os.path.splitext(path)[0]
    fp = zipfile.ZipFile(path, "r")
    fp.extractall(extract_dir)
    return extract_dir


class DNS_Interspeech_2021_Dataset(Dataset):
    def __init__(self, datasets_path, config, mode="train"):
        super().__init__()

        # set mode
        self.mode = mode

        # get args
        args = config["train_dataset"]["args"]
        self.sr = args["sr"]
        self.n_fft = args["n_fft"]
        self.win_length = args["win_length"]
        self.hop_length = args["hop_length"]
        self.silence_length = args["silence_length"]
        self.snr_range = args["snr_range"]
        self.target_level = args["target_level"]
        self.target_level_floating_value = args["target_level_floating_value"]
        self.target_samples_length = int(args["audio_length"] * self.sr)
        self.files_limit = args["files_limit"]

        if self.mode in ["train"]:
            # get path
            clean_path = config["train_dataset"]["path"]["clean"]
            noise_path = config["train_dataset"]["path"]["noise"]
            # read clean, noise
            clean_data = pd.read_csv(clean_path).values
            noise_data = pd.read_csv(noise_path).values
            # limit
            if self.files_limit:
                clean_data = clean_data[: self.files_limit]
                noise_data = noise_data[: self.files_limit]
            # files to list
            self.clean_files = clean_data.reshape(1, len(clean_data))[0]
            self.noise_files = noise_data.reshape(1, len(noise_data))[0]
            # set length
            self.length = len(self.clean_files)
        elif self.mode in ["valid"]:
            # get path
            noisy_clean_path = config["valid_dataset"]["path"]["noisy_clean"]
            # read noisy, clean
            noisy_clean_data = pd.read_csv(noisy_clean_path).values
            # limit
            if self.files_limit:
                noisy_clean_data = noisy_clean_data[: self.files_limit, :]
            # files to list
            self.noisy_files = noisy_clean_data[:, 0].reshape(1, len(noisy_clean_data))[0]
            self.clean_files = noisy_clean_data[:, 1].reshape(1, len(noisy_clean_data))[0]
            # set length
            self.length = len(self.noisy_files)
        elif self.mode in ["test"]:
            # get path
            noisy_clean_path = config["test_dataset"]["path"]["noisy_clean"]
            # read noisy, clean
            noisy_clean_data = pd.read_csv(noisy_clean_path).values
            # limit
            if self.files_limit:
                noisy_clean_data = noisy_clean_data[: self.files_limit, :]
            # files to list
            self.noisy_files = noisy_clean_data[:, 0].reshape(1, len(noisy_clean_data))[0]
            self.clean_files = noisy_clean_data[:, 1].reshape(1, len(noisy_clean_data))[0]
            # set length
            self.length = len(self.noisy_files)
        else:
            assert self.mode not in ["train", "valid", "test"], f"mode only support train, valid."

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode in ["train"]:
            # get clean
            clean_file = self.clean_files[idx]
            clean, _ = librosa.load(clean_file, sr=self.sr)
            clean, _ = sub_sample(clean, self.target_samples_length)

            # get noise
            noise = select_file(self.noise_files, self.silence_length, self.sr, len(clean))
            assert len(clean) == len(noise), f"Inequality: {len(clean)} {len(noise)}"

            # get snr
            snr = np.random.randint(self.snr_range[0], self.snr_range[-1] + 1)

            # snr mix
            noisy, clean = snr_mix(
                clean=clean,
                noise=noise,
                snr=snr,
                target_level=self.target_level,
                target_level_floating_value=self.target_level_floating_value,
            )

            # get cIRM
            noisy_spec = librosa.stft(noisy, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
            clean_spec = librosa.stft(clean, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
            cIRM = get_cIRM(noisy_spec, clean_spec)

            return noisy_spec, cIRM
        elif self.mode in ["valid"]:
            # get noisy
            noisy_file = self.noisy_files[idx]
            noisy, _ = librosa.load(noisy_file, sr=self.sr)
            # get clean
            clean_file = self.clean_files[idx]
            clean, _ = librosa.load(clean_file, sr=self.sr)

            # get cIRM
            noisy_spec = librosa.stft(noisy, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
            clean_spec = librosa.stft(clean, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
            cIRM = get_cIRM(noisy_spec, clean_spec)

            return noisy_spec, clean, cIRM
        elif self.mode in ["test"]:
            return idx
        assert self.mode not in ["train", "valid", "test"], f"mode only support train, valid."


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
    for noisy_spec, cIRM in tqdm(train_iter, desc="train_iter"):
        noisy_mag = paddle.abs(noisy_spec).astype(paddle.float32)
        print(noisy_mag.shape, cIRM.shape)

    # get valid_iter
    valid_set = DNS_Interspeech_2021_Dataset(datasets_path, config, mode="valid")
    valid_iter = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    for noisy_spec, clean, cIRM in tqdm(valid_iter, desc="valid_iter"):
        noisy_mag = paddle.abs(noisy_spec).astype(paddle.float32)
        print(noisy_spec.shape, cIRM.shape)

    pass
