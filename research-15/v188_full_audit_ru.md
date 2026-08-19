# Полный аудит SHMQ-Ultimate v188

## Резюме

v188 является **лучшей исследовательской основой**, чем исходный v51, потому что в нём исправлены несколько реальных инфраструктурных дефектов: изоляция CUTLASS vendor-архива, provenance и source-manifest, отдельная telemetry для quantization/GEMM/end-to-end, peak-memory accounting, проверка partition invariants, capability gating и контрактный smoke-test pinned vLLM patch. Однако v188 **не является production baseline**: Kaggle T4 сообщил `terminal_decision=no_go`, потому что mixed-prefill gate не прошёл, а full-model Qwen quality и throughput имеют статус `not_run`.

Главный вывод аудита: проблемы из приложенной таблицы разделяются на три категории. Первая — уже исправленные защитные контракты и telemetry. Вторая — частично исправленные проблемы, где измерение добавлено, но сама причина производительной или memory-регрессии остаётся. Третья — отсутствующие gate-возможности: реальный Qwen/Qwen2.5-0.5B quality path и полноценная проверка patched vLLM в среде, где установлен vLLM 0.9.0.

## Фактический v188 результат на T4

| Сценарий | E2E speedup vs FP16 | GEMM speedup vs FP16 | Gate |
|---|---:|---:|---|
| Mixed QKV, rows=1 | 1.1963× | 1.3569× | passed |
| Mixed QKV, rows=16 | 0.2884× | 0.2835× | failed |
| Mixed QKV, rows=128 | 0.2762× | 0.2794× | failed |
| Pure INT4, rows=128 | 0.5921× | 0.6067× | diagnostic only |
| Pure INT8, rows=128 | 0.7132× | 0.7376× | diagnostic only |
| Pure FP16, rows=128 | 0.1404× | 0.1440× | diagnostic only |

v188 улучшил rows=128 по сравнению с v187: `0.0995× → 0.2762×`. Это подтверждает, что staged CUTLASS path является полезным направлением для больших M. Но rows=16 ухудшился по сравнению с v187: `0.3104× → 0.2884×`, а оба результата остаются намного медленнее dense FP16. Поэтому нельзя считать, что пункт 1 устранён полностью.

## Матрица проблем из приложения

| Приоритет | Проблема | Статус в v188 | Доказательство | Что должно быть исправлено |
|---:|---|---|---|---|
| 1 | Direct-WMMA повторно загружает A/B и проигрывает cuBLAS по occupancy/staging | **Частично исправлено** | `three_level_sm75.cu:104-280` активен для rows<32 и использует global `wmma::load_matrix_sync`; v188 активирует staged CUTLASS только для integer rows>=32. T4 показывает улучшение rows=128, но rows=16 всё ещё 0.2884× | Создать измеряемый fast path для rows=16 без изменения арифметики, затем отдельно оптимизировать CUTLASS rows>=32. Нельзя объявлять проблему решённой до прохождения mixed-prefill gate.
| 2 | Quantization launch незаметен в GEMM, но влияет на E2E | **Исправлено как наблюдаемость** | `sm75_backend.py:393-400` отдельно меряет quantization, GEMM, E2E и dense; v188 quantization примерно 0.026 ms | Сохранить telemetry и добавить regression assertions на schema/отношения времени. Устранение самой quantization стоимости не требуется, поскольку она не главный bottleneck.
| 3 | Expanded INT4 и временные tensors могут портить memory/occupancy | **Accounting исправлен, runtime-cost не устранён** | `sm75_backend.py:191-222` кеширует expanded INT4; `:402-453` сообщает expanded bytes и peak allocation. v188 rows=128 mixed peak E2E 62,603,776 bytes против dense 60,461,056 bytes; CUTLASS создаёт transposed metadata в `three_level_sm75.cu:560-569` при каждом launch | Кешировать transposed `[groups, channels]` metadata per module/state/device, сохраняя allocator stream ownership. Отдельно измерить persistent cache bytes и per-call allocation delta. Это безопаснее, чем пытаться убрать expanded INT4 до доказательства, что он доминирует.
| 4 | Проверяется только operator/reference correctness, full-model Qwen quality — `not_run` | **Не исправлено** | `build_mixllm_3level_kaggle.py:156-161` явно выставляет `full_model_qwen_quality='not_run'`; notebook запускает только toy model-gate allocator tests и microbenchmarks. `test_model_gate.py` содержит только `TinyCausalLM` | Добавить opt-in, но обязательный для production-report, Qwen/Qwen2.5-0.5B quality gate: deterministic calibration/evaluation inputs, reference loss/perplexity, quantized loss delta, max/mean last-token logit error, finite/deterministic checks, peak VRAM и explicit failure status.
| 5 | vLLM path и SM75 module path могут иметь разные backend-selection contracts | **Config contracts исправлены; real patched-vLLM execution incomplete** | `runtime_capability.py:28-47` корректно запрещает unvalidated SM75; `test_vllm_three_level.py` проверяет pinned commit/config/remapping. Но Kaggle v188 сообщает `vllm_apply_path='unavailable_environment'`, потому что `vllm` отсутствует, поэтому actual import/apply/smoke execution не подтверждён | Разделить `patch_contract`, `apply_execution` и `runtime_execution`; не считать unavailable environment passed. Добавить explicit production blocker и отдельный environment setup/skip reason. Проверить patched vLLM там, где vLLM 0.9.0 действительно установлен.

## Дополнительные найденные проблемы

### 1. Production decision не блокируется отсутствием full-model quality

`build_mixllm_3level_kaggle.py:215-219` рассчитывает `production_ready` только из native correctness и decode/prefill E2E. Full-model quality и throughput не входят в условие production readiness, хотя report сохраняет их как `not_run`. Это создаёт риск ложного interpretation: `terminal_decision='go'` мог бы появиться даже без проверки реальной Qwen-модели. Исправление должно требовать `full_model_qwen_quality='passed'` для production claim либо явно называть текущий режим `operator_only` и запрещать `go`.

### 2. vLLM отсутствующий в окружении разрешается как нефатальный результат

`VLLM_APPLY_CELL` правильно не подменяет отсутствие vLLM успехом, но итоговая execution-логика не требует `vllm_apply_path='passed'`. Для operator-only T4 microbenchmark это допустимо, но для заявления «custom vLLM patch поддержан» недостаточно. Нужно хранить два отдельных решения: `operator_production` и `vllm_production`, чтобы unavailable environment не маскировался.

### 3. CUTLASS metadata cache подготовлен в объектной модели, но не используется active path

`ThreeLevelLinear` содержит `_sm75_prefill_metadata`, однако `sm75_backend.py` его не заполняет, а `run_cutlass_int_partition` каждый раз выполняет `scale.transpose(...).contiguous()` и аналогичный zero transpose. Это конкретная missing implementation, не просто theoretical optimization. Она влияет на allocation pressure и повторный host-side setup, хотя v188 показывает, что это не единственная причина rows=128 gap.

### 4. Performance contract проверяет только отдельные gate shapes, но не защищает dispatch thresholds

Source contract тесты проверяют наличие dispatch strings, но не выполняют отдельную regression-проверку, что rows=16 остаётся direct-WMMA, rows>=32 — staged CUTLASS, а rows=1 — decode. Нужно добавить более точный dispatch-contract seam или runtime instrumentation, чтобы последующая оптимизация не стала неактивным изменением, как исторический v143.

### 5. Full-model gate должен различать packed reference и native SM75 backend

`run_model_gate()` возвращает `backend='packed_reference'`, потому что он запускает модель через обычный module forward и capability-based native path зависит от CUDA. Notebook сейчас не вызывает `run_model_gate()` на Qwen вообще. Production report должен явно сохранять backend, device capability и whether native SM75 was exercised; иначе quality evidence может быть получен на reference path и ошибочно приписан CUDA kernel.

## Что уже не следует «чинить» повторно

Следующие части v188 следует сохранить, а не переписывать без нового доказательства:

| Часть | Причина сохранения |
|---|---|
| Vendor extraction через staging directory | v184 выявил overwrite исторического `sm75_cutlass_testbed.h`; v185/v188 изоляция исправила это и contracts проходят.
| Packed decode path без expanded INT4 | Decode gate проходит; расширение INT4 необходимо только для prefill.
| Native activation quantizer contract | Quantization telemetry показывает малую стоимость; reference/native correctness тесты покрывают ABI.
| Partition validation и graph-capture guard | Защищают duplicates/missing channels и запрещают lazy cache initialization во время CUDA graph capture.
| Runtime capability rejection | Предотвращает молчаливый запуск неподтверждённого SM75/Ampere пути.
| v188 staged CUTLASS rows>=32 | Это измеримо улучшило rows=128 относительно v187 и является лучшей текущей гипотезой для large-M, хотя ещё не production-pass.

## План безопасного исправления

Сначала добавить regression contracts и обновить report schema до двух независимых решений: `operator_production` и `model_vllm_production`. Затем реализовать кеширование transposed CUTLASS metadata с тем же state/device signature и `recordStream` для active stream. После этого добавить реальный Qwen 0.5B quality gate как отдельный Kaggle cell с bounded execution и explicit `not_run`/`unavailable_environment` semantics. Наконец, исследовать rows=16 direct-WMMA fast path отдельно от large-M CUTLASS, потому что v188 доказал, что эти regimes имеют разные оптимальные dataflows.

Ни одно изменение не должно считаться принятым, если оно меняет модель, benchmark settings, quality threshold, arithmetic, partition counts или обходит missing environment статус. Kaggle T4 остаётся обязательным источником performance evidence.

## Источники

1. v188 T4 gate artifact: `shmq-ultimate/mixllm_3level_kaggle/latest-output-v188-computer/mixllm_3level_gate.json`.
2. Current SM75 kernel: `external/MixLLM/mixllm/kernels/three_level_sm75.cu`, lines 104–280 and 552–753.
3. Current backend/cache/telemetry: `external/MixLLM/mixllm/sm75_backend.py`, lines 191–222 and 309–465.
4. Current model gate: `external/MixLLM/mixllm/model_gate.py`, lines 114–185.
5. Kaggle builder/report gate: `scripts/build_mixllm_3level_kaggle.py`, lines 48–126 and 128–222.
6. Model-gate tests: `external/MixLLM/mixllm/test/test_model_gate.py`, lines 40–115.
7. Original MixLLM launch family: `external/MixLLM/mixllm/kernels/mix_mma_multistage.cuh`, lines 178–349 and 353–520.
8. Original configuration catalog: `external/MixLLM/mixllm/kernels/mix_mma_config.h`, lines 22–179.
9. NVIDIA, “NVIDIA Turing Architecture In-Depth”: <https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/>.
10. NVIDIA, “CUDA Programming Guide”: <https://docs.nvidia.com/cuda/cuda-programming-guide/index.html>.


## Дополнительная проверка Kaggle model input

Kaggle API supports a `model_sources` field in kernel metadata for attaching a Kaggle Model to a notebook. The current v188 `kernel-metadata.json` has no `dataset_sources` or `model_sources`, so the Qwen2.5-0.5B quality gate cannot be expected to run from a local model input. The repair should attach the exact model through `model_sources` if the account/API accepts the documented model path, and otherwise report an explicit unavailable-environment status rather than downloading a substitute or silently skipping the quality claim.

Reference: [Kaggle Models documentation](https://www.kaggle.com/docs/models); [Kaggle kernel metadata model_sources announcement](https://www.kaggle.com/product-feedback/391480).
