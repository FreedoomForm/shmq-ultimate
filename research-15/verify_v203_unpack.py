from __future__ import annotations

import random


def byte_perm(x: int, y: int, selector: int) -> int:
    source = [(x >> (8 * i)) & 0xFF for i in range(4)] + [(y >> (8 * i)) & 0xFF for i in range(4)]
    result = 0
    for out in range(4):
        index = (selector >> (4 * out)) & 0x7
        result |= source[index] << (8 * out)
    return result


def vsub4(a: int, b: int) -> int:
    result = 0
    for i in range(4):
        value = ((a >> (8 * i)) & 0xFF) - ((b >> (8 * i)) & 0xFF)
        result |= (value & 0xFF) << (8 * i)
    return result


for _ in range(10000):
    word = random.randrange(1 << 32)
    zero = random.randrange(16)
    low = word & 0x0F0F0F0F
    high = (word >> 4) & 0x0F0F0F0F
    first = vsub4(byte_perm(low, high, 0x5140), zero * 0x01010101)
    last = vsub4(byte_perm(low, high, 0x7362), zero * 0x01010101)
    actual = []
    for value in (first, last):
        actual.extend((value >> (8 * i)) & 0xFF for i in range(4))
    expected = []
    for i in range(8):
        code = (word >> (4 * i)) & 0x0F
        expected.append((code - zero) & 0xFF)
    assert actual == expected, (word, zero, actual, expected)
print("v203 unpack mapping passed 10000 deterministic randomized cases")
