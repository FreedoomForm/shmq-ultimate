$p = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'
$raw = Get-Content -LiteralPath $p -Raw
$oldDecl = '__shared__ __align__(16) int8_t a_int8[kTile * kTile];'
$newDecl = '__shared__ __align__(16) int8_t a_int8[4][8 * kTile];'
$oldWrite = 'a_int8[linear] = row < rows'
$newWrite = 'a_int8[warp / 2][linear] = row < rows'
$oldLoad = 'a, reinterpret_cast<signed char*>(a_int8), kTile);'
$newLoad = 'a, reinterpret_cast<signed char*>(a_int8[warp / 2]), kTile);'
if (([regex]::Matches($raw, [regex]::Escape($oldDecl))).Count -ne 1) { throw 'v1024_decl_marker' }
if (([regex]::Matches($raw, [regex]::Escape($oldWrite))).Count -ne 1) { throw 'v1024_write_marker' }
if (([regex]::Matches($raw, [regex]::Escape($oldLoad))).Count -ne 1) { throw 'v1024_load_marker' }
$raw = $raw.Replace($oldDecl, $newDecl).Replace($oldWrite, $newWrite).Replace($oldLoad, $newLoad)
[IO.File]::WriteAllText($p, $raw, (New-Object Text.UTF8Encoding($false)))
Write-Output 'v1024_per_row_a_applied'
