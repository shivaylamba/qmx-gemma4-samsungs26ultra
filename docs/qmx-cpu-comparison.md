# Comparison with QMX-CPU

This comparison was made against Kartikey's
[`QMX-CPU`](https://github.com/carrycooldude/QMX-CPU) repository at commit
`882adcf32592d75a9d94ba86476eec2195715e8f`.

## What QMX-CPU does well

- It is a compact example centered on Gemma 3 270M, so developers can try a
  small model without the memory pressure of a 4B model.
- Its standalone scripts cover Q4 and Q8 models and one-thread and four-thread
  measurements.
- Its documentation distinguishes SME QMX from the KleidiAI I8MM path that can
  remain active when SME is disabled.

## Important implementation differences

| Area | QMX-CPU | This application |
| --- | --- | --- |
| Primary model | Gemma 3 270M | Gemma 4 E2B and other imported GGUF models |
| Model placement | Uses llama.cpp defaults; no dynamic layer planner | Measures a two-layer repack, then chooses a model-specific count from current Android memory headroom |
| QMX source integration | The build script clones Qualcomm KleidiAI, but does not pass that checkout to llama.cpp's CMake configuration | Pins Qualcomm KleidiAI and passes it through `FETCHCONTENT_SOURCE_DIR_KLEIDIAI_DOWNLOAD` |
| QMX kernel binding | No app-owned llama.cpp patch selecting the explicit Qualcomm SME1 symbols | App-owned patch binds and logs `qmx_mopa` and `qmx_dot` |
| Runtime proof | UI reports QMX after native initialization | Requires SME selection, a `CPU_KLEIDIAI` model buffer, and logs actual QMX GEMM and GEMV invocation |
| Native libraries | Includes prebuilt shared libraries | Builds the application libraries from pinned source revisions |
| Chat behavior | Re-formats individual prompts | Preserves multi-turn chat state in the model context |
| UI status | Uses a polished dark dashboard, but includes estimated speed text and treats successful initialization as QMX active | Adapts the dashboard, bubbles, suggestions, and floating composer while showing only runtime-observed QMX status, selected blocks, and buffer size |

The QMX-CPU approach is reasonable for a 270M model because all supported
weights can normally be repacked without approaching the phone's memory limit.
It does not solve the dynamic-layer question for a 4B model. This application
therefore keeps an explicit `qmx_layers` override for reproducible experiments,
but normal launches do not hard-code six or any other final count.

## Automatic selection in this application

On a normal QMX launch the application:

1. Reads the model's transformer-block count from GGUF metadata.
2. Loads two blocks through the CPU buffer type and observes the real
   `CPU_KLEIDIAI` repacked-buffer size reported by llama.cpp.
3. Reads Android's currently available memory and low-memory threshold.
4. Reserves space for the mapped GGUF, app and context allocations, KV cache,
   and a 25 percent packing margin.
5. Unloads the probe and reloads the model with the calculated number of blocks.

This is intentionally conservative. Android memory availability changes with
background applications, context size, model quantization, and the model's
packing density. Automatic selection reduces the chance of paging or process
termination, but it is not a promise that every eligible layer can fit and it
does not imply every operation in a selected block runs on QMX. QMX accelerates
eligible quantized linear GEMM and GEMV work; attention, softmax, KV-cache work,
and unsupported tensors continue on other CPU kernels.

All llama.cpp integration changes are stored under `patches/` in this
repository and applied by `scripts/setup.ps1`. No llama.cpp pull request or
separate application-specific llama.cpp fork is required.

## UI patterns incorporated

The application adopts the strongest presentation ideas from QMX-CPU without
changing inference ownership:

- a compact dark hardware and model dashboard;
- a visible QMX status badge backed by native observations;
- separate streamed user and assistant bubbles;
- horizontally scrollable prompt suggestions;
- a fixed composer with send and stop states;
- model download, model import, and conversation clearing retained in the
  dashboard header.

Hard-coded throughput labels were intentionally not carried over. Thread-count
experiments remain explicit benchmark inputs rather than a cosmetic toggle that
silently reloads the chat model.
