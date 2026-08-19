$root = 'C:\Users\User\Downloads\Multi-main\Multi-main'
$patch = Join-Path $root 'shmq-ultimate\external\MixLLM\vllm_v0.9.0_patch\0002-add-mixllm-three-level-support.patch'
$backup = Join-Path $root 'research-15\audit_backups\0002-add-mixllm-three-level-support.before-v1.patch'
New-Item -ItemType Directory -Force -Path (Split-Path $backup -Parent) | Out-Null
Copy-Item -LiteralPath $patch -Destination $backup -Force
$text = Get-Content -LiteralPath $patch -Raw
$oldGate = @'
+        if contract.backend not in {"auto", "reference"}:
+            raise ValueError(
+                "vLLM 0.9.0 three-level serving is gated to backend=reference; "
+                "the native fused operator has not passed the packed ABI gate"
+            )
'@
$newGate = @'
+        if contract.backend not in {"auto", "sm75"}:
+            raise ValueError(
+                "vLLM 0.9.0 three-level serving requires backend=auto or sm75; "
+                "the installed native implementation is SM75-specific"
+            )
'@
if (-not $text.Contains($oldGate)) { throw 'Expected backend gate block was not found exactly' }
$text = $text.Replace($oldGate, $newGate)
$old = '+        return 70'
if (-not $text.Contains($old)) { throw 'Expected capability return 70 was not found' }
$text = $text.Replace($old, '+        return 75')
$old = '+        if not hasattr(layer,  _sm75_int4_expanded):'
if (-not $text.Contains($old)) { throw 'Expected INT4 cache guard was not found' }
$text = $text.Replace($old, '+        if not hasattr(layer, "_sm75_int4_expanded"):')
$old = '+        if not hasattr(layer, _sm75_fp16_placeholders):'
if (-not $text.Contains($old)) { throw 'Expected FP16 cache guard was not found' }
$text = $text.Replace($old, '+        if not hasattr(layer, "_sm75_fp16_placeholders"):')
[IO.File]::WriteAllText($patch, $text, (New-Object Text.UTF8Encoding($false)))
Write-Output ('PATCH='+$patch)
Write-Output ('BACKUP='+$backup)
Write-Output ('CAPABILITY_75='+$text.Contains('+        return 75'))
Write-Output ('NATIVE_GATE='+$text.Contains('+        if contract.backend not in {"auto", "sm75"}:'))
Write-Output ('QUOTED_INT4='+$text.Contains('+        if not hasattr(layer, "_sm75_int4_expanded"):'))
Write-Output ('QUOTED_FP16='+$text.Contains('+        if not hasattr(layer, "_sm75_fp16_placeholders"):'))
