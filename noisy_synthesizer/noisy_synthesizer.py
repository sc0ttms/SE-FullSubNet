# -*- coding: utf-8 -*-

import sys
import os
import toml
import librosa

sys.path.append("./")
from audiolib.audio import save_files_to_csv, split_clean, gen_noisy


def nosiy_synthesizer(config):
    # get config params
    datasets_path = config["datasets"]["path"]
    synthesizer_path = config["synthesizer"]["path"]
    synthesizer_args = config["synthesizer"]["args"]

    # get datasets path
    clean_path = os.path.abspath(datasets_path["clean"])
    noise_path = os.path.abspath(datasets_path["noise"])

    # get synthesizer path
    st_noise_path = os.path.abspath(synthesizer_path["noise"])
    st_train_clean_path = os.path.abspath(synthesizer_path["train_clean"])
    st_valid_path = os.path.abspath(synthesizer_path["valid"])
    st_test_path = os.path.abspath(synthesizer_path["test"])

    # get synthesizer args
    sr = synthesizer_args["sr"]
    audio_length = synthesizer_args["audio_length"]
    silence_length = synthesizer_args["silence_length"]
    total_hours = synthesizer_args["total_hours"]
    snr_range = synthesizer_args["snr_range"]

    # find all files
    clean_files = librosa.util.find_files(clean_path, ext="wav")
    noise_files = librosa.util.find_files(noise_path, ext="wav")

    # split datasets
    train_clean_files, valid_clean_files, test_clean_files = split_clean(clean_files, 0.1)

    # save noise files
    save_files_to_csv(st_noise_path, noise_files, "noise.csv")
    # save noise files
    save_files_to_csv(st_train_clean_path, train_clean_files, "train_clean.csv")

    # valid noisy synthesizer
    gen_noisy(
        valid_clean_files,
        noise_files,
        audio_length,
        silence_length,
        sr,
        snr_range,
        total_hours,
        st_valid_path,
        desc="valid noisy synthesizer",
    )

    # test noisy synthesizer
    gen_noisy(
        test_clean_files,
        noise_files,
        audio_length,
        silence_length,
        sr,
        snr_range,
        total_hours,
        st_test_path,
        desc="test noisy synthesizer",
    )


if __name__ == "__main__":
    # get config
    toml_path = os.path.join(os.path.dirname(__file__), "noisy_synthesizer.toml")
    config = toml.load(toml_path)

    # run noisy synthesizer
    nosiy_synthesizer(config)
    print("noisy synthesizer done!")
