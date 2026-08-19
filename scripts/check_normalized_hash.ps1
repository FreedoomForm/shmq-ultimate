$p = 'C:\Users\User\Downloads\Multi-main\Multi-main\shmq-ultimate\external\MixLLM\mixllm\kernels\three_level_sm75.cu'
$raw = [IO.File]::ReadAllText($p)
$norm = $raw.Replace(([string][char]13 + [char]10), ([string][char]10)).Replace(([string][char]13), ([string][char]10))
$sha = [Security.Cryptography.SHA256]::Create()
$bytes = [Text.Encoding]::UTF8.GetBytes($norm)
$h = ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
Write-Output ('normalized_text_sha=' + $h)
Write-Output 'expected_notebook_sha=ab8fdaaa426b978a85468b9be429ced8cec603ef3911192155635a745b7576e2'
