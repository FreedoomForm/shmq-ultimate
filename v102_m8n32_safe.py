from pathlib import Path

root = Path(r"C:\Users\User\Downloads\Multi-main\Multi-main")
src = root / "shmq-ultimate" / "external" / "MixLLM" / "mixllm" / "kernels" / "three_level_sm75.cu"
v51 = root / "research-15" / "versions" / "v86_branch_hoist" / "three_level_sm75.before.cu"
src.write_bytes(v51.read_bytes())
lines = src.read_text(encoding="utf-8").splitlines()

if any("wmma::fragment<wmma::accumulator, 8, 32, kTile, int>" in line for line in lines):
    raise RuntimeError("v102_already_present")

for i, line in enumerate(lines):
    if line == "  const int row_base = blockIdx.y * kTile;":
        lines[i] = "  int row_base = blockIdx.y * kTile;"
    if "int8_t b_int8[kPrefillWarps][kTile * kTile];" in line:
        lines[i] = line.replace("[kTile * kTile]", "[32 * kTile]")

start = next((i for i, line in enumerate(lines) if "float scaled_accumulators[kTile * kTile / kWarpSize] = {};" in line), -1)
if start < 0:
    raise RuntimeError("int_path_start")
lines.insert(start, "  row_base += (warp / 2) * 8;")
lines.insert(start + 1, "  channel_base = (tile_id - (precision == 4 ? 0 : tiles4)) * kPrefillChannels + (warp % 2) * 32;")
end = next((i for i in range(start + 2, len(lines)) if lines[i] == "#endif"), -1)
if end < 0:
    raise RuntimeError("int_path_end")

for i in range(start, end):
    line = lines[i]
    if "wmma::fragment<wmma::accumulator" in line and "kTile, kTile, kTile, int>" in line:
        lines[i] = line.replace("kTile, kTile, kTile, int>", "8, 32, kTile, int>")
    elif "wmma::fragment<wmma::matrix_a" in line and "kTile, kTile, kTile, signed char," in line:
        lines[i] = line.replace("kTile, kTile, kTile, signed char,", "8, 32, kTile, signed char,")
    elif "wmma::fragment<wmma::matrix_b" in line and "kTile, kTile, kTile, signed char," in line:
        lines[i] = line.replace("kTile, kTile, kTile, signed char,", "8, 32, kTile, signed char,")
    elif line == "  const bool full_rows = row_base + kTile <= rows;":
        lines[i] = "  const bool full_rows = row_base + 8 <= rows;"
    elif line == "  const bool full_channels = channel_base + kTile <= partition_size;":
        lines[i] = "  const bool full_channels = channel_base + 32 <= partition_size;"
    elif "for (int linear = threadIdx.x; linear < kTile * kTile;" in line:
        lines[i] = line.replace("kTile * kTile", "8 * kTile")
    elif "wmma::store_matrix_sync(accumulator_int[warp], accumulator, kTile," in line:
        lines[i] = line.replace("accumulator, kTile,", "accumulator, 32,")
    elif "b, reinterpret_cast<signed char*>(b_int8[warp]), kTile);" in line:
        lines[i] = line.replace(", kTile);", ", 32);")

partial = next((i for i in range(start, end) if "for (int linear = lane; linear < kTile * kTile;" in lines[i]), -1)
if partial < 0:
    raise RuntimeError("partial_start")
partial_end = next((i for i in range(partial, end) if lines[i] == "        __syncwarp();"), -1)
if partial_end < 0:
    raise RuntimeError("partial_end")
new_partial = [
    "        for (int linear = lane; linear < 32 * kTile;",
    "             linear += kWarpSize) {",
    "          const int channel_offset = linear / kTile;",
    "          const int k_offset = linear % kTile;",
    "          const int channel = channel_base + channel_offset;",
    "          int8_t weight = 0;",
    "          if (channel < partition_size) {",
    "            const int k = k_base + k_offset;",
    "            weight = precision == 4",
    "                ? expanded_int4[channel * width + k]",
    "                : weight_int8[channel * width + k];",
    "          }",
    "          b_int8[warp][channel_offset * kTile + k_offset] = weight;",
    "        }",
    "        __syncwarp();",
]
lines[partial : partial_end + 1] = new_partial

store = next((i for i, line in enumerate(lines) if "wmma::store_matrix_sync(accumulator_int[warp], accumulator, 32," in line), -1)
if store < 0:
    raise RuntimeError("store_marker")
out = next((i for i in range(store, len(lines)) if "for (int linear = lane, item = 0; linear < kTile * kTile;" in lines[i]), -1)
if out < 0:
    raise RuntimeError("output_loop")
lines[out] = lines[out].replace("kTile * kTile", "8 * 32")
if out + 2 >= len(lines):
    raise RuntimeError("output_loop_bounds")
if lines[out + 2] == "      const int tile_row = linear / kTile;":
    lines[out + 2] = "      const int tile_row = linear / 32;"
if lines[out + 3] == "      const int local_channel = channel_base + linear % kTile;":
    lines[out + 3] = "      const int local_channel = channel_base + linear % 32;"

src.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("v102_m8n32_safe_applied")
