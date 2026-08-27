# Standalone ADB QMX benchmark, 2026-08-27

This directory contains the standalone `llama-batched-bench` results collected on a Samsung Galaxy S26 Ultra without launching the Android app. It tests the methodology described in [Qualcomm's QMX article](https://www.qualcomm.com/developer/blog/2026/04/llama-models-acceleration-on-cpu-qmx). The useful result is clear: Gemma 4 E2B Q8_0 runs successfully through the QMX/SME path and shows a large one-thread advantage over a genuine generic-NEON build. A full-QMX Gemma 3 4B Q8_0 run is not valid on this 12 GB phone because the repacked model exceeds physical-memory headroom and causes extreme Android swap activity.

## Test device

- Device: Samsung Galaxy S26 Ultra, `SM-S948B`
- SoC property: `SM8850`
- Android: 16
- Physical RAM reported by `/proc/meminfo`: 11,389,624 KiB
- CPU topology observed through sysfs: CPU 0-5 max 3,628,800 kHz; CPU 6-7 max 4,742,400 kHz
- Android fixed-performance mode: enabled during measured series
- Root access: unavailable

The Qualcomm article says its frequencies were fixed at 3,953 MHz for the small cluster and 4,394 MHz for the big cluster. Those exact controls are not available from an unrooted production device. Android fixed-performance mode is an approximation, not the same control.

## Software and models

The exact Qualcomm article snapshot and a newer compatible snapshot were tested separately.

| Component | Revision |
|---|---|
| Qualcomm article llama.cpp | `25f40ca65f1aa596f8b1702bbac4bc48a45b87d7` |
| Measured standalone llama.cpp | `2e92ecd0247d25f09797f8fdb044a166522fc05d` |
| Android NDK | r28b (`28.1.13356709`) |
| Android API | 29 |

| Model | Bytes | SHA-256 |
|---|---:|---|
| Gemma 3 4B IT Q8_0 | 4,130,402,176 | `81bf0583ab5bad155a5a3b15d155a880a1a1e4f7de2de5c06f10f64ac49f8336` |
| Gemma 4 E2B IT Q8_0 | 4,967,497,152 | `996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a` |

The QMX runner was compiled for `armv9.2-a+sve2+sme+dotprod+i8mm` with KleidiAI enabled. The control was compiled for `armv8-a` with `GGML_CPU_KLEIDIAI=OFF` and `GGML_CPU_ALL_VARIANTS=OFF`. This matters because setting `GGML_KLEIDIAI_SME=0` alone does not create a generic-NEON control.

## Which Qualcomm configuration was reproduced

The article contains two different configurations:

1. Its experimental setup and charts specify 128 prompt tokens, 128 decode tokens, and 1 or 4 threads.
2. Its copy-paste example uses 512 prompt tokens, 200 decode tokens, and 8 threads on a Gemma 3 270M Q4_0 model.

The primary comparison here follows the chart configuration because it is the source of the published Q8 speedup values:

```text
-c 2048 -b 2048 -ub 512 -npp 128 -ntg 128 -npl 1 -t {1,4} -fa on
```

Four alternating QMX/NEON trials were run for Gemma 4 at each thread count. Trial 1 was treated as warm-up and trials 2-4 were averaged. Prefill throughput is used as a fixed-prompt TTFT compute proxy. It is not UI-to-first-visible-token latency.

## Exact article-commit outcome

The exact article commit does not provide a runnable two-model comparison on this phone:

- Gemma 3 Q8_0 loads and allocates a 3,931.42 MiB `CPU_KLEIDIAI` buffer, but the first inference operation exits with code 132 and `Illegal instruction`.
- Gemma 4 E2B Q8_0 is rejected as `unknown model architecture: 'gemma4'` because that snapshot predates Gemma 4 support.
- The Gemma 3 run with `GGML_KLEIDIAI_SME=0` still allocates a 3,931.42 MiB `CPU_KLEIDIAI` buffer. It is therefore not the article's claimed plain-NEON baseline.

See `exact-blog-commit-gemma3-qmx-probe.log`, `exact-blog-commit-gemma4-qmx-probe.log`, and `exact-blog-commit-gemma3-sme0-probe.log` for the raw output.

## Gemma 4 E2B Q8_0 result

Runtime evidence from the QMX runner includes:

```text
kleidiai: primary q8 kernel feature SME
kleidiai: SME enabled (GGML_KLEIDIAI_SME=1 debug override)
CPU_KLEIDIAI model buffer size = 4385.35 MiB
```

The generic-NEON runner has no KleidiAI selection and no `CPU_KLEIDIAI` allocation. The raw proof is in `gemma4-qmx-runtime-evidence.log` and `gemma4-neon-runtime-evidence.log`.

### Warmed means, trials 2-4

| Threads | Backend | Prefill tok/s | Decode tok/s |
|---:|---|---:|---:|
| 1 | QMX/SME | 23.946 | 14.808 |
| 1 | Generic NEON | 8.033 | 6.477 |
| 4 | QMX/SME | 38.268 | 6.622 |
| 4 | Generic NEON | 20.289 | 11.760 |

| Threads | Prefill speedup | Decode speedup |
|---:|---:|---:|
| 1 | **2.981x** | **2.286x** |
| 4 | **1.886x** | **0.563x** |

The one-thread result is repeatable. QMX warmed prefill ranged from 22.393 to 24.792 tok/s and decode from 14.590 to 15.019 tok/s. Generic NEON stayed within 8.031-8.034 tok/s prefill and 6.474-6.481 tok/s decode.

The four-thread mean is not a trustworthy hardware comparison. Results became bimodal while the phone warmed from about 38 C to 39.5 C. QMX prefill ranged from 27.290 to 44.211 tok/s after warm-up, and generic-NEON decode alternated between about 6.4 and 14.6 tok/s. The article does not publish its CPU affinity, while this phone has six lower-capacity cores and two higher-capacity cores. The per-trial values and temperatures are retained in `gemma4-trials.csv` instead of being hidden behind the mean.

### Comparison with Qualcomm's published Q8 summary

Qualcomm reports maximum Q8 gains of 2.9x TTFT and 1.5x decode at one thread, and 2.0x TTFT and 1.05x decode at four threads. Our stable Gemma 4 E2B one-thread ratios are 2.981x prefill and 2.286x decode.

This is not evidence that this build beats Qualcomm's. The model is Gemma 4 E2B 4.6B, while the article's `Gemma 4B` link resolves to Gemma 3 4B. The llama.cpp revision differs because the article revision cannot execute safely here. The frequencies are not locked to Qualcomm's values, and our prefill number is a `llama-batched-bench` throughput-derived TTFT proxy. Our control is deliberately generic `armv8-a` NEON, while Qualcomm has not published enough baseline build detail to prove that its NEON binary had the same instruction set. A weaker control increases the apparent speedup. The proper conclusion is that the one-thread QMX effect is real and of the same general magnitude as the article, not that the numbers are directly interchangeable.

## Gemma 3 4B Q8_0 result

The newer compatible QMX runner selects SME, but full repacking allocates:

```text
CPU_Mapped model buffer size = 3932.82 MiB
CPU_KLEIDIAI model buffer size = 7429.44 MiB
CPU compute buffer size = 522.13 MiB
KV buffers = 40.00 MiB + 174.00 MiB
```

Those allocations already total about 11.8 GiB before Android, the benchmark executable, page tables, and other process memory. The phone reports about 10.86 GiB of usable physical RAM.

A literal 128/128, one-thread QMX run was attempted. It produced no result and was stopped after 2 minutes 56 seconds:

- At 1 minute 30 seconds, available RAM was 886,112 KiB and about 8.85 GiB of Android swap was occupied.
- Battery temperature increased from 38.9 C to 42.9 C.
- The process did not respond immediately to `TERM` while in compute, so it was stopped with `SIGKILL`.

This run is invalid as a QMX performance measurement because it measures paging and compressed swap. The observations are in `gemma3-qmx-aborted-observations.csv`; the captured process output is in `gemma3-qmx-1t-run1.log`.

The generic-NEON runner completed normally:

| Threads | Warmed prefill mean | Warmed decode mean | Notes |
|---:|---:|---:|---|
| 1 | 5.850 tok/s | 4.620 tok/s | Stable across trials 2-4 |
| 4 | 8.925 tok/s | 6.070 tok/s | Bimodal scheduler/thermal behavior |

There is no valid full-QMX Gemma 3 speedup to report on this 12 GB device. Dividing the NEON result by a paging-dominated or aborted run would be misleading. This also explains why the app used partial tensor repacking: it keeps the model interactive by avoiding the standalone full-repack memory cliff.

## Reproduction

Build and deploy both standalone runners:

```powershell
.\scripts\build-standalone.ps1
.\scripts\deploy-standalone.ps1
```

Push a model separately because GGUF files are intentionally excluded from Git:

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb shell mkdir -p /data/local/tmp/qmx-standalone/models
& $adb push .\models\gemma-4-E2B-it-Q8_0.gguf /data/local/tmp/qmx-standalone/models/
```

Run the chart configuration. The first of four trials is marked as warm-up in the CSV:

```powershell
.\scripts\run-standalone-benchmark.ps1 `
  -RemoteModel /data/local/tmp/qmx-standalone/models/gemma-4-E2B-it-Q8_0.gguf `
  -ModelLabel gemma4
```

Do not launch the full-QMX Gemma 3 Q8_0 series on a 12 GB device without first checking the logged `CPU_KLEIDIAI` allocation and memory headroom. The exact run in this directory demonstrated severe swap pressure and was intentionally stopped.

## Bottom line

- The benchmark is now a standalone native ADB test, not an Android app benchmark.
- QMX execution is proven for Gemma 4 E2B by SME kernel selection and a non-zero KleidiAI buffer.
- Gemma 4 has a repeatable 2.981x prefill and 2.286x decode advantage over genuine generic NEON at one thread.
- Four-thread results need a disclosed CPU-affinity and frequency-control method before they should be compared with Qualcomm's chart.
- Full-QMX Gemma 3 Q8_0 does not fit cleanly in this phone's physical-memory budget with the newer compatible KleidiAI packing, so no valid speedup is claimed.
