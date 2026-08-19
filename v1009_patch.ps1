$src = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'
$a = [Collections.ArrayList](Get-Content -LiteralPath $src)
$raw = Get-Content -LiteralPath $src -Raw
if ($raw.Contains('wmma_m16n16k16_s8s8s32(')) { throw 'helper_already_present' }
$m = $a.IndexOf('__global__ void three_level_tensorcore_kernel(')
if ($m -lt 0) { throw 'kernel_marker' }
$h = @(
'',
'template <typename FragA, typename FragB, typename FragC>',
'__device__ __forceinline__ void wmma_m16n16k16_s8s8s32(',
'    FragC& accumulator, const FragA& a, const FragB& b) {',
'  const uint32_t* A = reinterpret_cast<const uint32_t*>(a.x);',
'  const uint32_t* B = reinterpret_cast<const uint32_t*>(b.x);',
'  int c0 = accumulator.x[0];',
'  int c1 = accumulator.x[1];',
'  int c2 = accumulator.x[2];',
'  int c3 = accumulator.x[3];',
'  int c4 = accumulator.x[4];',
'  int c5 = accumulator.x[5];',
'  int c6 = accumulator.x[6];',
'  int c7 = accumulator.x[7];',
'  int d0, d1, d2, d3, d4, d5, d6, d7;',
'  asm volatile(',
'      "wmma.mma.sync.aligned.row.col.m16n16k16.s32.s8.s8.s32 "',
'      "{%0,%1,%2,%3,%4,%5,%6,%7}, "',
'      "{%8,%9}, {%10,%11}, "',
'      "{%12,%13,%14,%15,%16,%17,%18,%19};"',
'      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3), "=r"(d4), "=r"(d5), "=r"(d6), "=r"(d7)',
'      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]),',
'        "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(c5), "r"(c6), "r"(c7));',
'  accumulator.x[0] = d0; accumulator.x[1] = d1;',
'  accumulator.x[2] = d2; accumulator.x[3] = d3;',
'  accumulator.x[4] = d4; accumulator.x[5] = d5;',
'  accumulator.x[6] = d6; accumulator.x[7] = d7;',
'}'
)
for ($z = $h.Count - 1; $z -ge 0; $z--) { $a.Insert($m, $h[$z]) }
$indices = @()
for ($q = 0; $q -lt $a.Count; $q++) { if ($a[$q] -match '^\s+wmma::mma_sync\(accumulator, a, b, accumulator\);$') { $indices += $q } }
if ($indices.Count -ne 2) { throw "mma_occurrences_$($indices.Count)" }
$a[$indices[1]] = '  wmma_m16n16k16_s8s8s32(accumulator, a, b);'
[IO.File]::WriteAllLines($src, [string[]]$a, (New-Object Text.UTF8Encoding($false)))
Write-Output "v1009_helper_inserted occurrences=$($indices.Count)"
