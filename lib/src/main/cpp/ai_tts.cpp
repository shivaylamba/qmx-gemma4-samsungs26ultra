#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <malloc.h>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "logging.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

namespace {

constexpr int32_t VOICE_CONTEXT_SIZE = 2048;
constexpr int32_t VOICE_BATCH_SIZE = 128;
constexpr int32_t DEFAULT_MAX_FRAMES = 240;

std::mutex voice_mutex;
llama_model * voice_model = nullptr;
llama_context * voice_context = nullptr;
mtmd_context * voice_mtmd = nullptr;
common_sampler * voice_sampler = nullptr;

std::atomic<bool> voice_sme_enabled(false);
std::atomic<bool> voice_q8_kernel_selected(false);
std::atomic<bool> voice_qmx_gemm_executed(false);
std::atomic<bool> voice_qmx_gemv_executed(false);
std::atomic<int> voice_kleidiai_buffer_centimib(0);
std::atomic<int> voice_layers(0);
std::atomic<int> voice_total_layers(0);

std::string voice_error;

double elapsed_ms(int64_t start_us, int64_t end_us) {
    return static_cast<double>(end_us - start_us) / 1000.0;
}

bool emit_pcm_chunk(
        JNIEnv * env, jobject callback, jmethodID method, const float * pcm,
        size_t n_samples, int32_t sample_rate, bool is_final) {
    if (callback == nullptr || method == nullptr) {
        return true;
    }
    std::vector<jshort> pcm16(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        const float value = std::max(-1.0f, std::min(1.0f, pcm[i]));
        pcm16[i] = static_cast<jshort>(value * 32767.0f);
    }
    jshortArray samples = env->NewShortArray(static_cast<jsize>(n_samples));
    if (samples == nullptr) {
        return false;
    }
    if (n_samples > 0) {
        env->SetShortArrayRegion(samples, 0, static_cast<jsize>(n_samples), pcm16.data());
    }
    env->CallVoidMethod(callback, method, samples, sample_rate, is_final ? JNI_TRUE : JNI_FALSE);
    env->DeleteLocalRef(samples);
    return !env->ExceptionCheck();
}

void voice_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    aichat_android_log_callback(level, text, user_data);
    if (text == nullptr) {
        return;
    }
    if (std::strstr(text, "kleidiai: SME enabled") != nullptr) {
        voice_sme_enabled.store(true);
    }
    if (std::strstr(text, "kleidiai: primary q8 kernel feature SME") != nullptr) {
        voice_q8_kernel_selected.store(true);
    }
    if (std::strstr(text, "CPU_KLEIDIAI model buffer size") != nullptr) {
        const char * equals = std::strchr(text, '=');
        if (equals != nullptr) {
            const double mib = std::strtod(equals + 1, nullptr);
            voice_kleidiai_buffer_centimib.store(static_cast<int>(std::lround(mib * 100.0)));
        }
    }
    if (std::strstr(text, "executing Qualcomm QMX Q8 SME1 GEMM/prefill") != nullptr) {
        voice_qmx_gemm_executed.store(true);
    }
    if (std::strstr(text, "executing Qualcomm QMX Q8 SME1 GEMV/decode") != nullptr) {
        voice_qmx_gemv_executed.store(true);
    }
}

void free_voice_model() {
    if (voice_sampler != nullptr) {
        common_sampler_free(voice_sampler);
        voice_sampler = nullptr;
    }
    if (voice_mtmd != nullptr) {
        mtmd_free(voice_mtmd);
        voice_mtmd = nullptr;
    }
    if (voice_context != nullptr) {
        llama_free(voice_context);
        voice_context = nullptr;
    }
    if (voice_model != nullptr) {
        llama_model_free(voice_model);
        voice_model = nullptr;
    }
    voice_layers.store(0);
    voice_kleidiai_buffer_centimib.store(0);
}

std::string qmx_status_json() {
    const char * sme_override = std::getenv("GGML_KLEIDIAI_SME");
    const bool process_sme_selected = voice_sme_enabled.load() &&
                                      voice_q8_kernel_selected.load();
    const bool explicit_sme_selected = sme_override != nullptr &&
                                       std::strcmp(sme_override, "1") == 0;
    // KleidiAI emits its selection messages once per process. Gemma can consume
    // those messages before telemetry is switched to the voice engine, so keep
    // the current voice model's packed buffer plus the explicit SME setting as
    // the durable selection signal. inferenceConfirmed additionally requires
    // the QMX-specific GEMM and GEMV execution markers captured below.
    const bool selected = voice_kleidiai_buffer_centimib.load() > 0 &&
                          (process_sme_selected || explicit_sme_selected);
    std::ostringstream out;
    out << "{"
        << "\"selected\":" << (selected ? "true" : "false") << ','
        << "\"gemmExecuted\":" << (voice_qmx_gemm_executed.load() ? "true" : "false") << ','
        << "\"gemvExecuted\":" << (voice_qmx_gemv_executed.load() ? "true" : "false") << ','
        << "\"bufferMiB\":" << std::fixed << std::setprecision(2)
        << voice_kleidiai_buffer_centimib.load() / 100.0 << ','
        << "\"layers\":" << voice_layers.load()
        << ",\"totalLayers\":" << voice_total_layers.load()
        << "}";
    return out.str();
}

} // namespace

extern "C" JNIEXPORT void JNICALL
Java_com_arm_aichat_VoiceInferenceEngine_nativeInit(
        JNIEnv * env, jobject, jstring native_lib_dir) {
    std::lock_guard<std::mutex> lock(voice_mutex);
    llama_log_set(voice_log_callback, nullptr);
    mtmd_log_set(voice_log_callback, nullptr);

    const char * path = env->GetStringUTFChars(native_lib_dir, nullptr);
    ggml_backend_load_all_from_path(path);
    env->ReleaseStringUTFChars(native_lib_dir, path);
    llama_backend_init();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_arm_aichat_VoiceInferenceEngine_nativeLoad(
        JNIEnv * env, jobject, jstring backbone_path, jstring mmproj_path, jint threads,
        jint requested_qmx_layers) {
    std::lock_guard<std::mutex> lock(voice_mutex);
    free_voice_model();
    voice_error.clear();
    // KleidiAI intentionally logs the first QMX GEMM and GEMV once per process.
    // Keep the matching proof flags process-scoped so a model reload cannot turn
    // already-observed execution into a false negative.

    const char * backbone = env->GetStringUTFChars(backbone_path, nullptr);
    const char * mmproj = env->GetStringUTFChars(mmproj_path, nullptr);

    alignas(64) unsigned char mapped_buffer_probe[64] = {};
    ggml_backend_buffer_t mapped_buffer = ggml_backend_cpu_buffer_from_ptr(
            mapped_buffer_probe, sizeof(mapped_buffer_probe));
    ggml_backend_buffer_type_t mapped_buffer_type = ggml_backend_buffer_get_type(mapped_buffer);

    std::string qmx_blocks_pattern = "^blk\\.[0-9]+\\.";
    if (requested_qmx_layers > 0) {
        std::ostringstream qmx_blocks;
        qmx_blocks << "^blk\\.(";
        for (int layer = 0; layer < requested_qmx_layers; ++layer) {
            if (layer > 0) {
                qmx_blocks << "|";
            }
            qmx_blocks << layer;
        }
        qmx_blocks << ")\\.";
        qmx_blocks_pattern = qmx_blocks.str();
    }
    const llama_model_tensor_buft_override tensor_overrides[] = {
            {qmx_blocks_pattern.c_str(), ggml_backend_cpu_buffer_type()},
            {".*", mapped_buffer_type},
            {nullptr, nullptr},
    };
    llama_model_params model_params = llama_model_default_params();
    model_params.tensor_buft_overrides = tensor_overrides;

    voice_model = llama_model_load_from_file(backbone, model_params);
    ggml_backend_buffer_free(mapped_buffer);
    if (voice_model == nullptr) {
        voice_error = "Could not load the Qwen3-TTS backbone";
        env->ReleaseStringUTFChars(backbone_path, backbone);
        env->ReleaseStringUTFChars(mmproj_path, mmproj);
        return 1;
    }
    const int total_layers = static_cast<int>(llama_model_n_layer(voice_model));
    voice_total_layers.store(total_layers);
    voice_layers.store(requested_qmx_layers > 0
            ? std::min(static_cast<int>(requested_qmx_layers), total_layers)
            : total_layers);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = VOICE_CONTEXT_SIZE;
    context_params.n_batch = VOICE_BATCH_SIZE;
    context_params.n_ubatch = VOICE_BATCH_SIZE;
    context_params.n_threads = threads;
    context_params.n_threads_batch = threads;
    context_params.embeddings = true;
    voice_context = llama_init_from_model(voice_model, context_params);
    if (voice_context == nullptr) {
        voice_error = "Could not create the Qwen3-TTS context";
        free_voice_model();
        env->ReleaseStringUTFChars(backbone_path, backbone);
        env->ReleaseStringUTFChars(mmproj_path, mmproj);
        return 2;
    }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = false;
    mtmd_params.n_threads = threads;
    mtmd_params.print_timings = true;
    voice_mtmd = mtmd_init_from_file(mmproj, voice_model, mtmd_params);
    if (voice_mtmd == nullptr || mtmd_gen_audio_get_info(voice_mtmd).type == MTMD_GEN_AUDIO_TYPE_NONE) {
        voice_error = "Could not load the Qwen3-TTS audio projector";
        free_voice_model();
        env->ReleaseStringUTFChars(backbone_path, backbone);
        env->ReleaseStringUTFChars(mmproj_path, mmproj);
        return 3;
    }

    common_params_sampling sampling_params;
    sampling_params.temp = 0.7f;
    sampling_params.top_k = 50;
    sampling_params.top_p = 1.0f;
    sampling_params.seed = 42;
    voice_sampler = common_sampler_init(voice_model, sampling_params);
    if (voice_sampler == nullptr) {
        voice_error = "Could not initialize the TTS sampler";
        free_voice_model();
        env->ReleaseStringUTFChars(backbone_path, backbone);
        env->ReleaseStringUTFChars(mmproj_path, mmproj);
        return 4;
    }

    env->ReleaseStringUTFChars(backbone_path, backbone);
    env->ReleaseStringUTFChars(mmproj_path, mmproj);
    return 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_VoiceInferenceEngine_nativeSynthesize(
        JNIEnv * env, jobject, jstring prompt_value, jstring language_value,
        jstring output_path_value, jint threads, jint max_frames, jobject pcm_callback) {
    std::lock_guard<std::mutex> lock(voice_mutex);
    if (voice_model == nullptr || voice_context == nullptr || voice_mtmd == nullptr || voice_sampler == nullptr) {
        voice_error = "Load both model files before synthesizing";
        return nullptr;
    }

    const char * prompt = env->GetStringUTFChars(prompt_value, nullptr);
    const char * language = env->GetStringUTFChars(language_value, nullptr);
    const char * output_path = env->GetStringUTFChars(output_path_value, nullptr);
    jmethodID pcm_callback_method = nullptr;
    if (pcm_callback != nullptr) {
        jclass callback_class = env->GetObjectClass(pcm_callback);
        pcm_callback_method = env->GetMethodID(callback_class, "onPcmChunk", "([SIZ)V");
        env->DeleteLocalRef(callback_class);
        if (pcm_callback_method == nullptr) {
            voice_error = "PCM callback method is unavailable";
            env->ReleaseStringUTFChars(prompt_value, prompt);
            env->ReleaseStringUTFChars(language_value, language);
            env->ReleaseStringUTFChars(output_path_value, output_path);
            return nullptr;
        }
    }

    llama_set_n_threads(voice_context, threads, threads);
    llama_memory_clear(llama_get_memory(voice_context), false);
    common_sampler_reset(voice_sampler);
    voice_error.clear();

    mtmd_helper::gen_audio generator(voice_context, voice_mtmd);
    mtmd_helper_gen_audio_inp input{};
    input.seq_id = 0;
    input.prompt = prompt;
    input.prompt_len = std::strlen(prompt);
    input.speaker_ref = nullptr;
    input.lang = language;
    input.top_k = 50;
    input.top_p = 1.0f;
    input.seed = 42;
    input.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;

    const int64_t start_us = ggml_time_us();
    if (generator.set_input(&input) != 0) {
        voice_error = "TTS rejected the input text";
        env->ReleaseStringUTFChars(prompt_value, prompt);
        env->ReleaseStringUTFChars(language_value, language);
        env->ReleaseStringUTFChars(output_path_value, output_path);
        return nullptr;
    }

    while (true) {
        const int32_t remaining = generator.step_prompt(VOICE_BATCH_SIZE);
        if (remaining < 0) {
            voice_error = "TTS prompt processing failed";
            env->ReleaseStringUTFChars(prompt_value, prompt);
            env->ReleaseStringUTFChars(language_value, language);
            env->ReleaseStringUTFChars(output_path_value, output_path);
            return nullptr;
        }
        if (remaining == 0) {
            break;
        }
    }
    const int64_t prompt_done_us = ggml_time_us();

    llama_token sampled = common_sampler_sample(voice_sampler, voice_context, -1);
    common_sampler_accept(voice_sampler, sampled, true);
    const float * hidden_state = llama_get_embeddings_ith(voice_context, -1);
    int frames = 0;
    int64_t first_frame_us = 0;
    int64_t first_pcm_us = 0;
    bool stop = false;
    const int frame_limit = max_frames > 0 ? max_frames : DEFAULT_MAX_FRAMES;

    while (!stop && frames < frame_limit) {
        const float * next_hidden_state = nullptr;
        if (generator.step_gen(sampled, hidden_state, &next_hidden_state, &stop) != 0) {
            voice_error = "Audio generation failed at frame " + std::to_string(frames);
            env->ReleaseStringUTFChars(prompt_value, prompt);
            env->ReleaseStringUTFChars(language_value, language);
            env->ReleaseStringUTFChars(output_path_value, output_path);
            return nullptr;
        }
        if (next_hidden_state == nullptr) {
            break;
        }
        if (frames == 0) {
            first_frame_us = ggml_time_us();
        }
        frames++;
        int32_t chunk_sample_rate = 0;
        const float * chunk_pcm = nullptr;
        size_t chunk_samples = 0;
        if (generator.get_pcm_chunk(&chunk_sample_rate, &chunk_pcm, &chunk_samples) != 0) {
            voice_error = "Could not read streaming PCM";
            env->ReleaseStringUTFChars(prompt_value, prompt);
            env->ReleaseStringUTFChars(language_value, language);
            env->ReleaseStringUTFChars(output_path_value, output_path);
            return nullptr;
        }
        if (chunk_samples > 0) {
            if (first_pcm_us == 0) {
                first_pcm_us = ggml_time_us();
            }
            if (!emit_pcm_chunk(env, pcm_callback, pcm_callback_method, chunk_pcm,
                                chunk_samples, chunk_sample_rate, false)) {
                voice_error = "PCM playback callback failed";
                env->ReleaseStringUTFChars(prompt_value, prompt);
                env->ReleaseStringUTFChars(language_value, language);
                env->ReleaseStringUTFChars(output_path_value, output_path);
                return nullptr;
            }
        }
        hidden_state = next_hidden_state;
        sampled = common_sampler_sample(voice_sampler, voice_context, -1);
        common_sampler_accept(voice_sampler, sampled, true);
    }
    const int64_t generation_done_us = ggml_time_us();

    int32_t sample_rate = 0;
    const char * wav_data = nullptr;
    size_t wav_size = 0;
    int64_t sample_count = 0;
    if (generator.get_output(&sample_rate, &wav_data, &wav_size, &sample_count) != 0 || wav_data == nullptr) {
        voice_error = "Could not assemble the output WAV";
        env->ReleaseStringUTFChars(prompt_value, prompt);
        env->ReleaseStringUTFChars(language_value, language);
        env->ReleaseStringUTFChars(output_path_value, output_path);
        return nullptr;
    }
    const float * final_pcm = nullptr;
    size_t final_samples = 0;
    int32_t final_sample_rate = 0;
    const int pcm_status = generator.get_pcm_chunk(&final_sample_rate, &final_pcm, &final_samples);
    if (first_pcm_us == 0 && final_samples > 0) {
        first_pcm_us = ggml_time_us();
    }
    if (pcm_status != 0 ||
        !emit_pcm_chunk(env, pcm_callback, pcm_callback_method, final_pcm, final_samples,
                        final_sample_rate > 0 ? final_sample_rate : sample_rate, true)) {
        voice_error = "Could not finish streaming PCM";
        env->ReleaseStringUTFChars(prompt_value, prompt);
        env->ReleaseStringUTFChars(language_value, language);
        env->ReleaseStringUTFChars(output_path_value, output_path);
        return nullptr;
    }

    std::ofstream wav(output_path, std::ios::binary | std::ios::trunc);
    wav.write(wav_data, static_cast<std::streamsize>(wav_size));
    wav.close();
    const int64_t end_us = ggml_time_us();

    env->ReleaseStringUTFChars(prompt_value, prompt);
    env->ReleaseStringUTFChars(language_value, language);
    env->ReleaseStringUTFChars(output_path_value, output_path);

    if (!wav) {
        voice_error = "Could not write the output WAV";
        return nullptr;
    }

    const double prompt_ms = elapsed_ms(start_us, prompt_done_us);
    const double first_frame_ms = first_frame_us > 0 ? elapsed_ms(start_us, first_frame_us) : 0.0;
    const double generation_ms = elapsed_ms(prompt_done_us, generation_done_us);
    const double wav_ms = elapsed_ms(generation_done_us, end_us);
    const double total_ms = elapsed_ms(start_us, end_us);
    const double audio_ms = sample_rate > 0 ? sample_count * 1000.0 / sample_rate : 0.0;
    const double rtf = audio_ms > 0 ? total_ms / audio_ms : 0.0;

    std::ostringstream result;
    result << std::fixed << std::setprecision(3)
           << "{"
           << "\"promptMs\":" << prompt_ms << ','
           << "\"firstFrameMs\":" << first_frame_ms << ','
           << "\"firstPcmMs\":" << (first_pcm_us > 0 ? elapsed_ms(start_us, first_pcm_us) : 0.0) << ','
           << "\"generationMs\":" << generation_ms << ','
           << "\"wavMs\":" << wav_ms << ','
           << "\"totalMs\":" << total_ms << ','
           << "\"audioMs\":" << audio_ms << ','
           << "\"rtf\":" << rtf << ','
           << "\"frames\":" << frames << ','
           << "\"sampleRate\":" << sample_rate << ','
           << "\"threads\":" << threads << ','
           << "\"qmx\":" << qmx_status_json()
           << "}";
    return env->NewStringUTF(result.str().c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_VoiceInferenceEngine_nativeQmxStatus(JNIEnv * env, jobject) {
    const std::string result = qmx_status_json();
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_arm_aichat_VoiceInferenceEngine_nativeActivateTelemetry(JNIEnv *, jobject) {
    llama_log_set(voice_log_callback, nullptr);
    mtmd_log_set(voice_log_callback, nullptr);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_VoiceInferenceEngine_nativeLastError(JNIEnv * env, jobject) {
    return env->NewStringUTF(voice_error.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_arm_aichat_VoiceInferenceEngine_nativeUnload(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(voice_mutex);
    free_voice_model();
    LOGi("Native heap purge after voice unload: %s",
         mallopt(M_PURGE_ALL, 0) == 1 ? "complete" : "not supported");
}

extern "C" JNIEXPORT void JNICALL
Java_com_arm_aichat_VoiceInferenceEngine_nativeShutdown(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(voice_mutex);
    free_voice_model();
    llama_backend_free();
}
