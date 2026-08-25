from collections import Counter

# TensorOpMultiplicand<ElementSize=4, Crosswise=64>
elements_per_access = 128 // 4
tile_contiguous = 128 // (128 // 8)
factor = tile_contiguous * elements_per_access // 64
partition_contiguous = 4
partition_strided = 4

def layout(c, s, stride=64):
    vec_c = c // elements_per_access
    vec_s = s // factor
    tile_c = vec_c // (tile_contiguous // factor)
    tile_c_res = vec_c % (tile_contiguous // factor) + (s % factor) * (tile_contiguous // factor)
    tile_s_res = vec_s % (tile_contiguous // factor)
    part_c = tile_c_res // partition_contiguous
    part_s = tile_s_res // partition_strided
    part_c_res = tile_c_res % partition_contiguous
    part_s_res = tile_s_res % partition_strided
    perm_c = part_c_res ^ (part_s_res % 4)
    perm_part = part_c ^ (part_s % 2)
    elem_c = (tile_c * tile_contiguous + perm_part * partition_contiguous + perm_c) * elements_per_access + (c % elements_per_access)
    elem_s = vec_s
    return elem_c + elem_s * stride * factor

vals = [layout(c, s) for c in range(64) for s in range(128)]
print('constants', elements_per_access, tile_contiguous, factor, min(vals), max(vals), len(set(vals)), len(vals))
print('physical modulo 2', Counter(v % 2 for v in vals))
print('byte occupancy', len(set(v // 2 for v in vals)), 'bytes', 64*128//2)
print('first 32 coords/offsets', [(c, s, layout(c,s)) for c,s in [(0,i) for i in range(32)]])
print('first bytes', sorted(set(v//2 for v in vals))[:20])
