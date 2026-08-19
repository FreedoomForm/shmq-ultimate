$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$py = Join-Path $root 'shmq-ultimate\external\MixLLM\mixllm\nn\modules\three_level_linear.py'
$backup = Join-Path $root 'research-15\audit_backups\three_level_linear.before-v105-native-forward.py'
New-Item -ItemType Directory -Force -Path (Split-Path $backup -Parent) | Out-Null
Copy-Item -LiteralPath $py -Destination $backup -Force
$text = Get-Content -LiteralPath $py -Raw
$marker = '    def forward(self, x: torch.Tensor) -> torch.Tensor:'
$start = $text.IndexOf($marker)
if ($start -lt 0) { throw 'Forward method marker was not found' }
$new = @'
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        if x.is_cuda and tuple(torch.cuda.get_device_capability(x.device)) == (7, 5):
            from mixllm.sm75_backend import load_sm75_backend, three_level_linear

            load_sm75_backend(torch)
            flat = x.reshape(-1, self.in_features)
            result = three_level_linear(self, flat, torch)
            if self.bias is not None:
                result = result + self.bias.float()
            return result.reshape(*x.shape[:-1], self.out_features).to(x.dtype)
        result = torch.nn.functional.linear(x.float(), self.dequantize_weight(),
                                            None if self.bias is None else self.bias.float())
        return result.to(x.dtype)
'@
$text = $text.Substring(0, $start) + $new
[IO.File]::WriteAllText($py, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output ('PY='+$py)
Write-Output ('BACKUP='+$backup)
Write-Output ('NATIVE_FORWARD='+$text.Contains('three_level_linear(self, flat, torch)'))
Write-Output ('FALLBACK_RETAINED='+$text.Contains('torch.nn.functional.linear(x.float(), self.dequantize_weight()'))
Write-Output ('RESHAPE_RETAINED='+$text.Contains('result.reshape(*x.shape[:-1], self.out_features)'))
