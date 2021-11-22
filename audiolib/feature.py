# -*- coding: utf-8 -*-

import paddle


def drop_band(input, num_groups=2):
    """
    Reduce computational complexity of the sub-band part in the FullSubNet model.

    Shapes:
        input: [B, C, F, T]
        return: [B, C, F // num_groups, T]
    """
    [batch_size, _, num_freqs, _] = input.shape
    assert (
        batch_size > num_groups
    ), f"Batch size = {batch_size}, num_groups = {num_groups}. The batch size should larger than the num_groups."

    if num_groups <= 1:
        # No demand for grouping
        return input

    # Each sample must has the same number of the frequencies for parallel training.
    # Therefore, we need to drop those remaining frequencies in the high frequency part.
    if num_freqs % num_groups != 0:
        input = input[..., : (num_freqs - (num_freqs % num_groups)), :]
        num_freqs = input.shape[2]

    output = []
    for group_idx in range(num_groups):
        samples_indices = paddle.arange(group_idx, batch_size, num_groups)
        freqs_indices = paddle.arange(group_idx, num_freqs, num_groups)

        selected_samples = paddle.index_select(input, samples_indices, axis=0)
        selected = paddle.index_select(selected_samples, freqs_indices, axis=2)

        output.append(selected)  # [B // num_groups, C, F // num_groups, T]

    return paddle.concat(output, axis=0)
