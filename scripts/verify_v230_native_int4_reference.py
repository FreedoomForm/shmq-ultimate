from __future__ import annotations

import random


def pack_uint4(values: list[int]) -> list[int]:
    assert len(values) % 2 == 0
    return [values[i] | (values[i + 1] << 4) for i in range(0, len(values), 2)]


def unpack_uint4(packed: list[int], count: int) -> list[int]:
    result: list[int] = []
    for byte in packed:
        result.extend((byte & 0xF, (byte >> 4) & 0xF))
    return result[:count]


def decompose_signed_int8(value: int) -> tuple[int, int]:
    low = value & 0xF
    high = value >> 4
    assert value == low + 16 * high
    assert 0 <= low <= 15
    assert -8 <= high <= 7
    return low, high


def run() -> None:
    rng = random.Random(230)
    for _ in range(20_000):
        activation = rng.randint(-128, 127)
        weight = rng.randint(0, 15)
        zero = rng.randint(0, 15)
        low, high = decompose_signed_int8(activation)
        reference = activation * (weight - zero)
        decomposed = low * (weight - zero) + 16 * high * (weight - zero)
        assert reference == decomposed

    for _ in range(200):
        rows, cols, depth = 8, 8, 32
        activation = [[rng.randint(-128, 127) for _ in range(depth)] for _ in range(rows)]
        weight = [[rng.randint(0, 15) for _ in range(depth)] for _ in range(cols)]
        zero = [rng.randint(0, 15) for _ in range(cols)]
        reference = [
            [sum(activation[m][k] * (weight[n][k] - zero[n]) for k in range(depth))
             for n in range(cols)]
            for m in range(rows)
        ]
        low_product = [[0 for _ in range(cols)] for _ in range(rows)]
        high_product = [[0 for _ in range(cols)] for _ in range(rows)]
        zero_correction = [[0 for _ in range(cols)] for _ in range(rows)]
        for m in range(rows):
            for n in range(cols):
                for k in range(depth):
                    low, high = decompose_signed_int8(activation[m][k])
                    low_product[m][n] += low * weight[n][k]
                    high_product[m][n] += high * weight[n][k]
                    zero_correction[m][n] += activation[m][k] * zero[n]
        reconstructed = [
            [low_product[m][n] + 16 * high_product[m][n] - zero_correction[m][n]
             for n in range(cols)]
            for m in range(rows)
        ]
        assert reconstructed == reference

    values = [rng.randint(0, 15) for _ in range(128)]
    assert unpack_uint4(pack_uint4(values), len(values)) == values
    print("v230 native INT4 reference: PASS")


if __name__ == "__main__":
    run()
