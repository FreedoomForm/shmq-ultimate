from __future__ import annotations

import random


def split_value(value: int) -> tuple[int, int]:
    low = value & 0xF
    high = (value >> 4) & 0xF
    signed_high = high - 16 if high >= 8 else high
    return low, signed_high


def check_identity() -> None:
    for value in range(-128, 128):
        low, high = split_value(value)
        assert value == low + 16 * high, (value, low, high)


def check_correction() -> None:
    rng = random.Random(204)
    rows, groups, channels = 7, 3, 11
    for _ in range(100):
        activations = [rng.randrange(-127, 128) for _ in range(rows * groups * 128)]
        q = [rng.randrange(16) for _ in range(channels * groups * 128)]
        zero = [rng.randrange(16) for _ in range(channels * groups)]
        sa = [rng.random() for _ in range(rows * groups)]
        sw = [rng.random() for _ in range(channels * groups)]
        for row in range(rows):
            for channel in range(channels):
                native = 0.0
                corrected = 0.0
                for group in range(groups):
                    a_base = (row * groups + group) * 128
                    q_base = (channel * groups + group) * 128
                    dot_q_a = sum(q[q_base + k] * activations[a_base + k] for k in range(128))
                    dot_q_low = 0
                    dot_q_high = 0
                    sum_a = 0
                    for k in range(128):
                        low, high = split_value(activations[a_base + k])
                        dot_q_low += q[q_base + k] * low
                        dot_q_high += q[q_base + k] * high
                        sum_a += activations[a_base + k]
                    assert dot_q_a == dot_q_low + 16 * dot_q_high
                    native += sa[group * rows + row] * sw[group * channels + channel] * dot_q_a
                    corrected += sa[group * rows + row] * sw[group * channels + channel] * (
                        dot_q_low + 16 * dot_q_high - zero[group * channels + channel] * sum_a
                    )
                reference = 0.0
                for group in range(groups):
                    a_base = (row * groups + group) * 128
                    q_base = (channel * groups + group) * 128
                    reference += sa[group * rows + row] * sw[group * channels + channel] * sum(
                        (q[q_base + k] - zero[group * channels + channel]) * activations[a_base + k]
                        for k in range(128)
                    )
                assert abs(corrected - reference) < 1e-9


if __name__ == "__main__":
    check_identity()
    check_correction()
    print("native decomposition reference: PASS")
