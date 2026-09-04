package com.example.qmxassistant

import android.content.Context
import android.os.StatFs
import android.util.Log
import androidx.core.content.edit
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

internal class AssistantModelRepository(context: Context) {
    data class ModelSpec(
        val name: String,
        val bytes: Long,
        val sha256: String,
        val url: String,
    )

    sealed interface State {
        data object Idle : State
        data class Active(val downloadedBytes: Long, val totalBytes: Long) : State
        data object Complete : State
        data class Failed(val reason: String) : State
    }

    private val appContext = context.applicationContext
    private val preferences = appContext.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
    private val modelDir: File
        get() = File(requireNotNull(appContext.getExternalFilesDir(null)), "assistant-models")

    val gemmaFile: File get() = File(modelDir, GEMMA.name)
    val voiceBackboneFile: File get() = File(modelDir, VOICE_BACKBONE.name)
    val voiceProjectorFile: File get() = File(modelDir, VOICE_PROJECTOR.name)
    val kittenModelFile: File get() = File(modelDir, KITTEN_MODEL.name)
    val kittenVoicesFile: File get() = File(modelDir, KITTEN_VOICES.name)
    val kittenRulesFile: File get() = File(modelDir, KITTEN_RULES.name)
    val kittenListFile: File get() = File(modelDir, KITTEN_LIST.name)
    val totalBytes: Long get() = FILES.sumOf(ModelSpec::bytes)

    fun query(): State {
        val current = state.get()
        return if (current == State.Idle && isVerified()) State.Complete else current
    }

    fun hasCompleteFiles(): Boolean = FILES.all { spec ->
        File(modelDir, spec.name).let { it.isFile && it.length() == spec.bytes }
    }

    fun isVerified(): Boolean = preferences.getBoolean(VERIFIED, false) && hasCompleteFiles()

    fun verifyAndRemember(): Boolean {
        FILES.forEach { spec ->
            val file = File(modelDir, spec.name)
            if (!file.isFile || file.length() != spec.bytes || sha256(file) != spec.sha256) {
                return false
            }
        }
        preferences.edit { putBoolean(VERIFIED, true) }
        return true
    }

    fun enqueue() {
        val token = synchronized(DOWNLOAD_LOCK) {
            if (state.get() is State.Active) return
            modelDir.mkdirs()
            check(StatFs(modelDir.absolutePath).availableBytes >= totalBytes + FREE_SPACE_HEADROOM) {
                "At least ${formatBytes(totalBytes + FREE_SPACE_HEADROOM)} free storage is required."
            }
            preferences.edit { remove(VERIFIED) }
            state.set(State.Active(existingBytes(), totalBytes))
            generation.incrementAndGet()
        }
        executor.execute {
            runCatching {
                FILES.forEachIndexed { index, spec -> downloadOne(spec, index, token) }
                check(token == generation.get()) { "Download cancelled" }
                preferences.edit { putBoolean(VERIFIED, true) }
                state.set(State.Complete)
            }.onFailure { error ->
                if (token == generation.get()) {
                    Log.e(TAG, "Combined model download failed", error)
                    state.set(State.Failed(error.message ?: error.javaClass.simpleName))
                }
            }
        }
    }

    fun cancel() {
        generation.incrementAndGet()
        state.set(State.Idle)
        preferences.edit { remove(VERIFIED) }
        FILES.forEach { spec -> File(modelDir, "${spec.name}.part").delete() }
    }

    private fun downloadOne(spec: ModelSpec, index: Int, token: Long) {
        val destination = File(modelDir, spec.name)
        if (destination.isFile && destination.length() == spec.bytes && sha256(destination) == spec.sha256) {
            publishProgress(index, spec.bytes, token)
            return
        }

        destination.delete()
        val partial = File(modelDir, "${spec.name}.part")
        if (partial.length() > spec.bytes) partial.delete()
        var lastError: Throwable? = null
        repeat(MAX_ATTEMPTS) { attempt ->
            check(token == generation.get()) { "Download cancelled" }
            runCatching {
                downloadAttempt(spec, partial, index, token)
                check(partial.length() == spec.bytes) {
                    "${spec.name} has ${partial.length()} bytes; expected ${spec.bytes}"
                }
                check(sha256(partial) == spec.sha256) { "SHA-256 mismatch for ${spec.name}" }
                check(partial.renameTo(destination)) { "Could not publish ${spec.name}" }
                publishProgress(index, spec.bytes, token)
                return
            }.onFailure { error ->
                lastError = error
                Log.w(TAG, "Attempt ${attempt + 1}/$MAX_ATTEMPTS failed for ${spec.name}", error)
                if (attempt + 1 < MAX_ATTEMPTS) Thread.sleep(1_000L * (attempt + 1))
            }
        }
        throw lastError ?: IllegalStateException("Download failed for ${spec.name}")
    }

    private fun downloadAttempt(spec: ModelSpec, partial: File, index: Int, token: Long) {
        var offset = partial.length()
        val connection = (URL(spec.url).openConnection() as HttpURLConnection).apply {
            instanceFollowRedirects = true
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            setRequestProperty("Accept-Encoding", "identity")
            setRequestProperty("User-Agent", "QMX-Voice-Assistant/Android")
            if (offset > 0) setRequestProperty("Range", "bytes=$offset-")
        }
        try {
            val response = connection.responseCode
            check(response == HttpURLConnection.HTTP_OK || response == HttpURLConnection.HTTP_PARTIAL) {
                "HTTP $response for ${spec.name}"
            }
            if (offset > 0 && response == HttpURLConnection.HTTP_OK) {
                partial.delete()
                offset = 0
            }
            BufferedInputStream(connection.inputStream, IO_BUFFER_BYTES).use { input ->
                FileOutputStream(partial, offset > 0).buffered(IO_BUFFER_BYTES).use { output ->
                    val buffer = ByteArray(IO_BUFFER_BYTES)
                    var current = offset
                    while (true) {
                        check(token == generation.get()) { "Download cancelled" }
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                        current += count
                        publishProgress(index, current, token)
                    }
                }
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun publishProgress(index: Int, currentFileBytes: Long, token: Long) {
        if (token != generation.get()) return
        val prior = FILES.take(index).sumOf { spec ->
            File(modelDir, spec.name).takeIf { it.length() == spec.bytes }?.length() ?: 0L
        }
        state.set(State.Active((prior + currentFileBytes).coerceAtMost(totalBytes), totalBytes))
    }

    private fun existingBytes(): Long = FILES.sumOf { spec ->
        val complete = File(modelDir, spec.name)
        if (complete.length() == spec.bytes) spec.bytes
        else File(modelDir, "${spec.name}.part").length().coerceAtMost(spec.bytes)
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered(IO_BUFFER_BYTES).use { input ->
            val buffer = ByteArray(IO_BUFFER_BYTES)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        val GEMMA = ModelSpec(
            name = "gemma-4-E2B-it-Q8_0.gguf",
            bytes = 4_967_497_152L,
            sha256 = "996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a",
            url = "https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q8_0.gguf?download=true",
        )
        val VOICE_BACKBONE = ModelSpec(
            name = "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
            bytes = 1_847_874_400L,
            sha256 = "ac7931aeb2e7aad1a6ed6602d353a5679c9d096b18ce8204ac730a8408d572e1",
            url = "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/main/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf?download=true",
        )
        val VOICE_PROJECTOR = ModelSpec(
            name = "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
            bytes = 446_422_912L,
            sha256 = "6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2",
            url = "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/main/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf?download=true",
        )
        val KITTEN_MODEL = ModelSpec(
            name = "kitten_tts_nano_v0_8.onnx",
            bytes = 56_767_095L,
            sha256 = "320564d2615f235de972ca27a7f39551c94185cfa24ca85b07a29084135f1e5e",
            url = "https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32/resolve/7a1db645b1f3ab9420761d87428e042b9cec3f26/kitten_tts_nano_v0_8.onnx?download=true",
        )
        val KITTEN_VOICES = ModelSpec(
            name = "kitten-tts-nano-0.8-voices.npz",
            bytes = 3_278_902L,
            sha256 = "8aa7cee235abb0739cb51e6559685f65a4dacd95568833d05699b1633f519b3f",
            url = "https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32/resolve/7a1db645b1f3ab9420761d87428e042b9cec3f26/voices.npz?download=true",
        )
        val KITTEN_RULES = ModelSpec(
            name = "kitten-tts-en_rules",
            bytes = 161_691L,
            sha256 = "8e75e9341ea735cc514b29a7d3a95c6c241c1cc176ad43e5699b8f7f66ab3194",
            url = "https://raw.githubusercontent.com/espeak-ng/espeak-ng/59eb19938f12e30881c81d86ce4a7de25414c9f4/dictsource/en_rules",
        )
        val KITTEN_LIST = ModelSpec(
            name = "kitten-tts-en_list",
            bytes = 102_788L,
            sha256 = "24eb79018ed6253c10682096de672ce9265c1fe15c3e19e7f754d57a0fcd9790",
            url = "https://raw.githubusercontent.com/espeak-ng/espeak-ng/59eb19938f12e30881c81d86ce4a7de25414c9f4/dictsource/en_list",
        )
        private val FILES = listOf(
            GEMMA,
            VOICE_BACKBONE,
            VOICE_PROJECTOR,
            KITTEN_MODEL,
            KITTEN_VOICES,
            KITTEN_RULES,
            KITTEN_LIST,
        )
        private const val PREFERENCES = "assistant_model_download"
        private const val VERIFIED = "verified"
        private const val TAG = "QmxAssistantModels"
        private const val MAX_ATTEMPTS = 3
        private const val CONNECT_TIMEOUT_MS = 30_000
        private const val READ_TIMEOUT_MS = 60_000
        private const val IO_BUFFER_BYTES = 4 * 1024 * 1024
        private const val FREE_SPACE_HEADROOM = 1_024L * 1024 * 1024
        private val DOWNLOAD_LOCK = Any()
        private val executor = Executors.newSingleThreadExecutor()
        private val generation = AtomicLong(0)
        private val state = AtomicReference<State>(State.Idle)

        fun formatBytes(bytes: Long): String = when {
            bytes >= 1_000_000_000L -> "%.2f GB".format(bytes / 1_000_000_000.0)
            bytes >= 1_000_000L -> "%.1f MB".format(bytes / 1_000_000.0)
            else -> "$bytes B"
        }
    }
}
