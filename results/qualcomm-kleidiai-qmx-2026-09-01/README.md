# Qualcomm KleidiAI QMX SME1 validation, 2026-09-01

This run validates the Android app after replacing llama.cpp's fetched Arm
KleidiAI release with Qualcomm's `kleidi-ai-qmx` branch. The tested phone was a
Samsung SM-S948B with SM8850 on Android 16. `/proc/cpuinfo` reports `sme` and
SME matrix data types but does not report `sme2`, so this is the SME1 path.

## Source and binary proof

- upstream llama.cpp base commit: `2e92ecd0247d25f09797f8fdb044a166522fc05d`
- app-owned integration patch: `patches/llama.cpp-qualcomm-qmx.patch`
- Qualcomm KleidiAI QMX commit: `8316de1358b5c3589120af0f3d99477e7c16b2e7`
- The packaged Armv9.2 backend exports the exact Qualcomm
  `qmx_mopa` and `qmx_dot` kernel and run symbols.
- Runtime instrumentation immediately before KleidiAI's kernel invocation
  confirms `qmx_mopa` executes during GEMM/prefill and `qmx_dot` executes during
  GEMV/decode for both tested models.
- The control uses the same APK and selected blocks but starts a fresh process
  with SME disabled. It selects KleidiAI I8MM and emits no Qualcomm QMX
  execution marker.

The direct lines are retained in [`runtime-evidence.txt`](runtime-evidence.txt).

## Automatic layer-selection validation

The final APK was also installed and run without a `qmx_layers` intent extra.
For Gemma 4 E2B Q8_0, the app measured a two-block 138.90 MiB QMX probe and,
from 6,993 MiB of available memory at launch, selected 8 of 35 blocks. It
purged the unloaded probe allocation, reloaded a 569.17 MiB
`CPU_KLEIDIAI` buffer, reported `active=true`, and executed both the Qualcomm
QMX GEMM/prefill and GEMV/decode kernels. This is a diagnostic validation of
the automatic path, not an additional warmed benchmark comparison.

## Method

Each mode ran in a fresh app process with Android fixed-performance mode enabled:

- Q8_0 model
- six leading transformer blocks assigned for KleidiAI repacking
- one inference thread
- 128 prompt-processing tokens
- 128 decode tokens
- six trials per process
- first trial discarded as warm-up

Android fixed-performance mode and stay-awake were disabled again after capture.
The complete per-trial data are in
[`benchmark-runs.csv`](benchmark-runs.csv).

## Warmed results

| Model and phase | Qualcomm QMX SME1 | KleidiAI I8MM | Speedup |
|---|---:|---:|---:|
| Gemma 3 4B Q8, prefill | 6.4966 tok/s | 6.4226 tok/s | 1.0115x |
| Gemma 3 4B Q8, decode | 6.2839 tok/s | 5.8796 tok/s | 1.0688x |
| Gemma 4 E2B Q8, prefill | 8.3086 tok/s | 8.2176 tok/s | 1.0111x |
| Gemma 4 E2B Q8, decode | 8.3293 tok/s | 8.0948 tok/s | 1.0290x |

These ratios prove that the Qualcomm kernels execute, but they are not a direct
reproduction of Qualcomm's maximum article results. This app repacks supported
tensors from only six blocks, and its control is optimized KleidiAI I8MM rather
than a separately compiled plain-NEON binary. Gemma 3 has 34 blocks and uses a
1,084.08 MiB QMX repack for the selected six. Gemma 4 has 35 blocks and uses a
430.27 MiB QMX repack for the selected six. Operations and blocks outside the
selected QMX-compatible linear projections remain on other CPU paths.

The benchmark measures llama.cpp prompt-processing and text-generation
throughput. Prefill throughput is a fixed-workload compute proxy, not complete
UI-to-first-visible-token latency.
