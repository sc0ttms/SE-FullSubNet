# -*- coding: utf-8 -*-

import os
import librosa
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
import paddle


EPS = np.finfo(np.float32).eps
np.random.seed(0)


def save_files_to_csv(path, files, name):
    df = pd.DataFrame({"files": files})
    df.to_csv(os.path.join(path, name), index=False)


def split_clean(clean, scale=0.1):
    clean_len = len(clean)
    valid_len = int(clean_len * scale)
    test_len = valid_len

    clean_idx = np.arange(clean_len)
    valid_and_test_mask = np.random.choice(clean_idx, valid_len + test_len, replace=False)
    valid_mask = valid_and_test_mask[:valid_len]
    test_mask = valid_and_test_mask[valid_len + 1 :]

    train = [clean[file] for file in range(clean_len) if file not in valid_and_test_mask]
    valid = [clean[file] for file in range(clean_len) if file in valid_mask]
    test = [clean[file] for file in range(clean_len) if file in test_mask]

    return train, valid, test


def gen_noisy(clean_files, noise_files, audio_length, silence_length, sr, snr_range, total_hours, save_path, desc=None):
    # check save path exist
    noisy_path = os.path.join(save_path, "noisy")
    os.makedirs(noisy_path, exist_ok=True)
    clean_path = os.path.join(save_path, "clean")
    os.makedirs(clean_path, exist_ok=True)

    # cal params
    target_samples_length = int(audio_length * sr)
    num_files = int(total_hours * 60 * 60 / audio_length)

    # gen noisy
    noisy_clean_files = []
    for idx in tqdm(range(num_files), desc=desc):
        # get clean
        clean_file = random_select_from(clean_files)
        clean, _ = librosa.load(clean_file, sr=sr)
        clean, _ = sub_sample(clean, target_samples_length)

        # get noise data
        noise = select_file(noise_files, silence_length, sr, len(clean))
        assert len(clean) == len(noise), f"Inequality: {len(clean)} {len(noise)}"

        # get snr
        snr = np.random.randint(snr_range[0], snr_range[-1] + 1)

        # snr mix
        noisy, clean = snr_mix(clean, noise, snr)

        # save noisy and clean
        noisy_added_path = os.path.join(noisy_path, f"{idx}_{snr}_noisy.wav")
        noisy_clean_files.append(noisy_added_path)
        sf.write(noisy_added_path, noisy, sr)
        clean_added_path = os.path.join(clean_path, f"{idx}_{snr}_clean.wav")
        noisy_clean_files.append(clean_added_path)
        sf.write(clean_added_path, clean, sr)

    # save list
    df = pd.DataFrame({"noisy": noisy_clean_files[::2], "clean": noisy_clean_files[1::2]})
    df.to_csv(os.path.join(save_path, os.path.basename(save_path), ".csv"), index=False)


def random_select_from(dataset_list):
    return np.random.choice(dataset_list)


def sub_sample(audio, sub_sample_length, start_position=-1):
    assert np.ndim(audio) == 1, f"Only support 1D data. The dim is {np.ndim(audio)}"
    length = len(audio)

    if length > sub_sample_length:
        if start_position < 0:
            start_position = np.random.randint(length - sub_sample_length)
        end_position = start_position + sub_sample_length
        audio = audio[start_position:end_position]
    elif length < sub_sample_length:
        audio = np.append(audio, np.zeros(sub_sample_length - length, dtype=np.float32))
    else:
        pass

    assert len(audio) == sub_sample_length

    return audio, start_position


def select_file(source_files, silence_length, sr, target_samples_length):
    output = np.zeros(0, dtype=np.float32)
    silence = np.zeros(int(silence_length * sr), dtype=np.float32)
    remaining_samples_length = target_samples_length

    while remaining_samples_length > 0:
        source_file = random_select_from(source_files)
        output_new_added, _ = librosa.load(source_file, sr=sr)
        output = np.append(output, output_new_added)
        remaining_samples_length -= len(output_new_added)

        # add silence
        if remaining_samples_length > 0:
            silence_samples_length = min(remaining_samples_length, len(silence))
            output = np.append(output, silence[:silence_samples_length])
            remaining_samples_length -= silence_samples_length

    if len(output) > target_samples_length:
        start_position = np.random.randint(len(output) - target_samples_length)
        end_position = start_position + target_samples_length
        output = output[start_position:end_position]

    return output


def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)


def normalize(audio, target_level=-25):
    """Normalize the signal to the target level"""
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def snr_mix(clean, noise, snr, target_level=-25, target_level_floating_value=10, clipping_threshold=0.99):
    # Normalizing to -25 dB FS
    clean = clean / (max(abs(clean)) + EPS)
    clean = normalize(clean, target_level)
    clean_rms = (clean ** 2).mean() ** 0.5

    noise = noise / (max(abs(noise)) + EPS)
    noise = normalize(noise, target_level)
    noise_rms = (noise ** 2).mean() ** 0.5

    # Set the noise level for a given SNR
    noise_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + EPS)
    noise = noise * noise_scalar

    # Mix noise and clean speech
    noisy = clean + noise

    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(
        target_level - target_level_floating_value, target_level + target_level_floating_value
    )
    noisy_rms = (noisy ** 2).mean() ** 0.5
    noisy_scalar = 10 ** (noisy_rms_level / 20) / (noisy_rms + EPS)
    noisy = noisy * noisy_scalar
    clean = clean * noisy_scalar

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisy):
        noisy_maxamplevel = max(abs(noisy)) / (clipping_threshold - EPS)
        noisy = noisy / noisy_maxamplevel
        clean = clean / noisy_maxamplevel

    return noisy, clean


def offline_laplace_norm(input):
    """

    Args:
        input: [B, C, F, T]

    Returns:
        [B, C, F, T]
    """
    # utterance-level mu
    mu = paddle.mean(input, axis=list(range(1, input.dim())), keepdim=True)

    normed = input / (mu + EPS)

    return normed


def cumulative_laplace_norm(input):
    """

    Args:
        input: [B, C, F, T]

    Returns:
        [B, C, F, T]
    """
    [batch_size, num_channels, num_freqs, num_frames] = input.shape
    input = input.reshape([batch_size * num_channels, num_freqs, num_frames])

    step_sum = paddle.sum(input, axis=1)  # [B * C, F, T] => [B, T]
    cumulative_sum = paddle.cumsum(step_sum, axis=-1)  # [B, T]

    entry_count = paddle.arange(
        num_freqs,
        num_freqs * num_frames + 1,
        num_freqs,
        dtype=input.dtype,
    )
    entry_count = entry_count.reshape([1, num_frames])  # [1, T]
    entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

    cumulative_mean = cumulative_sum / entry_count  # B, T
    cumulative_mean = cumulative_mean.reshape([batch_size * num_channels, 1, num_frames])

    normed = input / (cumulative_mean + EPS)

    return normed.reshape([batch_size, num_channels, num_freqs, num_frames])
