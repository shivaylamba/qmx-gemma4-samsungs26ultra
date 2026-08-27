# From a Build Flag to Proof: Building a QMX-Accelerated Gemma Chat App on the Galaxy S26 Ultra

*A developer’s account of integrating llama.cpp, KleidiAI, and Qualcomm Matrix Extensions into a real Kotlin Android app, and learning that “enabled” is not the same as “executing.”*

On-device LLM demos often stop at a terminal benchmark. I wanted something closer to a product: a Kotlin chat app that could import a GGUF model, stream responses, remember follow-up questions, stay usable when the keyboard opened, and prove that inference was actually running on the CPU’s Qualcomm Matrix Extension (QMX) path.

The target was a Samsung Galaxy S26 Ultra (`SM-S948B`, Qualcomm `SM8850`). The first model was Gemma 4 E2B Instruct Q8_0; later, I added the Gemma 3 4B Q8_0 model linked as “Gemma 4B” in [Qualcomm’s QMX article](https://www.qualcomm.com/developer/blog/2026/04/llama-models-acceleration-on-cpu-qmx) so I could make a closer benchmark comparison.

The finished sample, build instructions, and benchmark notes are available in the [qmx-gemma4-samsungs26ultra repository](https://github.com/shivaylamba/qmx-gemma4-samsungs26ultra).

![Multi-turn Gemma chat running on the Galaxy S26 Ultra](qmx-multiturn.png)

## QMX is CPU acceleration, not an NPU backend

QMX is a matrix-multiplication capability in the CPU. It is compatible with Arm Scalable Matrix Extension (SME) instructions and is designed to accelerate operations such as the GEMM and GEMV kernels that dominate transformer inference. In our stack, [llama.cpp](https://github.com/ggml-org/llama.cpp) calls optimized kernels supplied through [KleidiAI](https://github.com/ARM-software/kleidiai), which selects an SME implementation when the binary, runtime setting, and CPU capabilities all agree.

The application path looks like this:

```text
Kotlin chat UI
    ↓
JNI inference wrapper
    ↓
llama.cpp / ggml CPU backend
    ↓
KleidiAI Q8 SME kernels
    ↓
QMX-capable CPU
```

There is no QNN, Vulkan, OpenCL, GPU, or NPU inference backend in the APK. That distinction matters: a phone can contain several accelerators, but this project is specifically about accelerating CPU execution.

## Building an app, not just a benchmark binary

Qualcomm’s article gives the essential native recipe: use Android NDK r28b, enable `GGML_CPU_KLEIDIAI`, compile for Armv9.2 with SME-related features, set `GGML_KLEIDIAI_SME=1`, and run llama.cpp’s benchmark. That is a useful hardware and performance starting point. Turning it into a usable Android application adds another layer of engineering.

The sample uses a Kotlin activity for model selection and chat, a small JNI boundary, and a native inference engine built around llama.cpp. The model is imported through Android’s document picker and copied into private app storage rather than bundled in the APK. The native layer keeps a 2,048-token context, preserves the conversation and KV cache between turns, formats messages using the model’s chat template, and streams generated text back to Kotlin.

There were also mundane but important product details. The first UI placed the composer underneath Samsung Keyboard, so the layout now consumes IME insets and keeps the input above the keyboard. A **Clear** action resets both the visible messages and native conversation state. A small llama.cpp patch passes `enable_thinking=false` to Gemma’s template so users see the answer rather than hidden reasoning markup.

![The chat composer remains visible above Samsung Keyboard](qmx-keyboard.png)

## How the app asks for QMX

The native runtime must choose its kernel before model initialization, so the app sets the documented environment switch before loading the inference library:

```kotlin
val qmxMode = intent.getStringExtra("qmx_mode") ?: "1"
Os.setenv("GGML_KLEIDIAI_SME", qmxMode, true)
```

The CMake build enables the KleidiAI CPU integration for `arm64-v8a`:

```cmake
set(GGML_SYSTEM_ARCH "ARM")
set(GGML_CPU_KLEIDIAI ON)
```

That still does not mean every model tensor automatically uses a QMX-optimized buffer. KleidiAI repacks supported quantized weights into a CPU buffer suited to its kernels. In this app, `QMX_ACCELERATED_LAYERS` controls how many leading transformer blocks are assigned to the regular CPU buffer type and therefore become eligible for KleidiAI repacking. Other tensors remain memory-mapped and execute through the regular llama.cpp Arm CPU kernels.

This is why “six QMX layers” does not mean the other layers run on the GPU or NPU. All layers run on the CPU; the difference is which CPU kernel and weight layout they use.

## Proving that QMX is executing

My first lesson was that a green badge based on a requested flag would prove almost nothing. The app reports **QMX ACTIVE** only after it observes all three of these native runtime signals:

```text
kleidiai: primary q8 kernel feature SME
kleidiai: SME enabled
load_tensors: CPU_KLEIDIAI model buffer size = 430.27 MiB
```

Together, these show that KleidiAI selected an SME Q8 kernel, enabled SME, and actually repacked model weights into its buffer. The APK inspection provides a second line of evidence: it contains the CPU variants, llama.cpp, and `libkleidiai.so`, with none of the GPU/NPU backends mentioned earlier.

For a clean verification run:

```powershell
adb shell am force-stop com.example.qmxgemma
adb logcat -c
adb shell am start -n com.example.qmxgemma/.MainActivity
adb logcat -s ai-chat:I InferenceEngineImpl:I QmxGemma:I
```

This runtime evidence is more reliable than inferring execution from build options. A feature can compile successfully yet fall back at runtime, or fail only when the first incompatible instruction executes.

## The memory cost nobody sees in a speedup chart

Q8 weights are already large, and KleidiAI’s optimized representation is an additional allocation rather than a free replacement for the mapped model. The Android process also contains the Kotlin UI, JNI/native runtime, and a persistent chat context. Increasing the number of repacked layers therefore increases peak memory quickly.

For Gemma 4 E2B, the conservative app default repacks 6 of 35 blocks, producing a 430.27 MiB `CPU_KLEIDIAI` buffer. For Gemma 3 4B, the same six-block setting consumes 1,084.08 MiB. The rest of each model remains mapped and still executes on the CPU, just outside the QMX-repacked path.

I probed Gemma 3 by increasing the number of blocks placed into the QMX-repacked `CPU_KLEIDIAI` buffer. The limit I hit was not whether the CPU could execute QMX instructions; it was whether Android could keep the larger repacked buffer inside the app process. The highest layer placement I could fit was 30 of 34 blocks. Beyond that, the issue became process memory pressure from the QMX/KleidiAI buffer plus the mapped model and chat runtime.

The practical answer was to keep six layers as the safe interactive default. The controlled QMX versus generic NEON benchmark below used 20 of 34 blocks because that was the matched run captured with both APKs and stable measurements, not because 20 was the absolute maximum layer count that could fit.

## The benchmark trap: “SME off” was not NEON

The app includes an ADB-triggered harness with 128 prompt tokens, 128 generated tokens, one sequence, and explicit one- or four-thread contexts. Each mode runs in a fresh process because KleidiAI selects kernels during initialization.

Our first A/B test changed only this value:

```text
GGML_KLEIDIAI_SME=1  → SME/QMX
GGML_KLEIDIAI_SME=0  → non-SME
```

The result looked disappointing. With only six Gemma 3 blocks repacked, one-thread prefill improved by 1.012× and decode by 1.074×. Four-thread results were effectively flat. Gemma 4 E2B was similarly close at 1.006× prefill and 1.027× decode.

The logs explained why: the supposed baseline selected KleidiAI’s optimized I8MM Q8 kernel. It was a valid non-SME comparison, but it was not the generic NEON-only baseline described in Qualcomm’s methodology. We were comparing one optimized CPU path against another while accelerating only a fraction of the model.

To create a real control, I built a second APK from the same source revision and benchmark harness with `-march=armv8-a`, `GGML_CPU_KLEIDIAI=OFF`, and `GGML_CPU_ALL_VARIANTS=OFF`. It packaged one generic CPU library, emitted no KleidiAI selection, and allocated no `CPU_KLEIDIAI` buffer. The QMX build used the same model and harness with 20 of 34 Gemma 3 blocks repacked for the measured comparison.

## Results: Gemma 3 4B Q8_0, QMX versus generic NEON

Both builds ran six trials per mode on the same phone with Android fixed-performance mode enabled. The first run was discarded as warm-up. Context size was 256 for the isolated test, batch and micro-batch were 128, Flash Attention resolved on, and thread count was held constant across each pair.

| Phase | Threads | QMX/SME, 20/34 blocks | Generic NEON | Speedup |
|---|---:|---:|---:|---:|
| 128-token prefill | 1 | 11.795 tok/s | 5.849 tok/s | **2.017×** |
| Decode | 1 | 7.400 tok/s | 4.712 tok/s | **1.571×** |
| 128-token prefill | 4 | 38.436 tok/s | 16.314 tok/s | **2.356×** |
| Decode | 4 | 12.805 tok/s | 10.997 tok/s | **1.164×** |

These results finally showed the expected shape: the biggest gains appear during compute-heavy prefill, while decode improves less because it is more constrained by memory traffic. That matches Qualcomm’s explanation of why matrix acceleration helps GEMM-dominated prefill more than the GEMV- and bandwidth-heavy decode phase.

The numbers need two qualifications. First, our prefill ratio is calculated from fixed-prompt processing throughput, so it is a useful TTFT compute proxy rather than an end-to-end measurement from UI submission to first visible token. Second, one warmed one-thread QMX decode trial fell to 4.848 tok/s while the other warmed trials clustered between 8.030 and 8.046 tok/s. Keeping that outlier produces the conservative 1.571× result above; excluding it would produce 1.706×.

## How this compares with Qualcomm’s published results

For Q8 models, Qualcomm reports maximum speedups of 2.9× TTFT and 1.5× decode with one thread, and 2.0× TTFT and 1.05× decode with four threads. Their tests used 128-token prompts and 128-token decode, fixed CPU frequencies, and a NEON-only baseline on the Snapdragon 8 Elite Gen 5 Mobile platform.

Our closest matched Gemma 3 experiment reaches 2.017×/1.571× with one thread and 2.356×/1.164× with four. That is encouraging, but it is not an apples-to-apples reproduction. The measured comparison used 20 of 34 blocks in the QMX-repacked buffer, even though layer-fit testing could reach 30 of 34 before memory pressure became the limiting factor. The app also used Android’s fixed-performance mode rather than Qualcomm’s stated per-cluster frequency locks, and our prefill metric is not full TTFT. The four-thread series warmed the phone from roughly 34 °C to 39 °C and showed late thermal decline.

The correct conclusion is not that our build “beats” or “loses to” Qualcomm’s. It is that QMX is demonstrably executing, provides a large advantage over a genuine generic-NEON control, and follows the same prefill-versus-decode pattern. Reproducing exact laboratory numbers requires the same source dependency graph, CPU affinity, frequency controls, thermal state, model file, tensor placement, and metric definition.

## What failed and what it taught us

I also tried the exact llama.cpp commit pinned in Qualcomm’s article, built with NDK r28b, Android API 29, KleidiAI, and the documented Armv9.2/SME flags. Its non-SME run repacked all 34 Gemma 3 blocks, but still selected KleidiAI I8MM rather than a plain NEON baseline. Its QMX run repacked the full model and then terminated with `SIGILL` inside `libggml-cpu.so`.

The device reports base SME but not SME2. In that older source combination, the Q8 path selected SME2-named kernels from a base-SME feature check. The newer llama.cpp/KleidiAI revision pinned by this app takes a base-SME-compatible path and works. This is exactly why native dependency revisions must be pinned together and validated on the target device.

Trying the article’s very large `-b 2048 -ub 512` settings with a full model also triggered heavy paging on this app/device setup. Once the process is swapping gigabytes, the measurement describes storage pressure more than matrix compute. Bigger buffers are not automatically better on Android.

## Reproduce the app

The repository pins the native dependency and includes an idempotent setup script:

```powershell
git clone --recurse-submodules https://github.com/shivaylamba/qmx-gemma4-samsungs26ultra.git
cd qmx-gemma4-samsungs26ultra
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1

$env:JAVA_HOME = "C:\path\to\jdk-21"
.\gradlew.bat :app:assembleDebug :app:lintDebug
adb install -r .\app\build\outputs\apk\debug\app-debug.apk
```

Open the app, choose a supported Q8_0 GGUF, wait for the native load to finish, and confirm that the badge reports **QMX ACTIVE**. The complete README contains the tested model hashes, exact ADB benchmark commands, layer-control option, and power-setting cleanup commands.

## Takeaways

The most valuable part of this project was not getting a large speedup number. It was learning how to make that number defensible.

- QMX acceleration happens on the CPU through SME-compatible KleidiAI kernels; it is not QNN or NPU execution.
- A build flag requests a path. Kernel-selection logs plus a non-zero repacked buffer prove that the path actually ran.
- `GGML_KLEIDIAI_SME=0` can still select an optimized I8MM KleidiAI kernel, so it should not automatically be labeled “NEON baseline.”
- Partial layer repacking is a useful compromise for an interactive Android app, but memory policy can prevent full-model QMX placement even when the hardware supports it.
- Prefill benefits more than decode because the former is compute-bound and the latter quickly becomes memory-bound.
- Mobile benchmarks need fresh processes, warm-up handling, thermal reporting, identical model files, and explicit metric definitions.

The next step is to split interactive inference and laboratory benchmarking into separate native processes, measure actual first-visible-token latency from the UI, and test whether a smaller context or Q4 path can increase QMX coverage without introducing instruction-set compatibility issues.

## References

- [Qualcomm: Accelerating LLAMA inference on mobile CPUs using Qualcomm Matrix Extensions](https://www.qualcomm.com/developer/blog/2026/04/llama-models-acceleration-on-cpu-qmx)
- [Project source and reproducibility notes](https://github.com/shivaylamba/qmx-gemma4-samsungs26ultra)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Arm KleidiAI](https://github.com/ARM-software/kleidiai)
- [Gemma 3 4B GGUF used by Qualcomm’s model link](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/tree/main)
- [Gemma 4 E2B GGUF used by the app](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF)
