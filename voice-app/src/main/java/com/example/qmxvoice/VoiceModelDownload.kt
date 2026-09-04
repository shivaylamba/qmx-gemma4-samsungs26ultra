package com.example.qmxvoice

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

class VoiceModelDownload(private val context: Context) {
    data class ModelFile(val name: String, val bytes: Long, val sha256: String)

    sealed interface State {
        data object Idle : State
        data class Active(val downloadedBytes: Long, val totalBytes: Long) : State
        data object Complete : State
        data class Failed(val reason: String) : State
    }

    private val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
    private val modelDir: File
        get() = File(requireNotNull(context.getExternalFilesDir(null)), "voice-models")

    val backboneFile: File get() = File(modelDir, BACKBONE.name)
    val mmprojFile: File get() = File(modelDir, MMPROJ.name)
    val totalBytes: Long get() = FILES.sumOf { it.bytes }

    fun enqueue() {
        val token = synchronized(DOWNLOAD_LOCK) {
            if (state.get() is State.Active) return
            modelDir.mkdirs()
            check(StatFs(modelDir.absolutePath).availableBytes >= totalBytes + FREE_SPACE_HEADROOM) {
                "At least ${formatBytes(totalBytes + FREE_SPACE_HEADROOM)} free storage is required."
            }
            preferences.edit { remove(VERIFIED) }
            state.set(State.Active(existingPartialBytes(), totalBytes))
            generation.incrementAndGet()
        }
        executor.execute {
            runCatching {
                FILES.forEachIndexed { index, model -> downloadOne(model, index, token) }
                check(token == generation.get()) { "Download cancelled" }
                preferences.edit { putBoolean(VERIFIED, true) }
                state.set(State.Complete)
            }.onFailure { error ->
                if (token == generation.get()) {
                    Log.e(TAG, "Model download failed", error)
                    state.set(State.Failed(error.message ?: error.javaClass.simpleName))
                }
            }
        }
    }

    fun query(): State {
        val current = state.get()
        return if (current == State.Idle && isVerified()) State.Complete else current
    }

    fun verifyAndRemember(): Boolean {
        FILES.forEach { model ->
            val file = File(modelDir, model.name)
            if (!file.isFile || file.length() != model.bytes || sha256(file) != model.sha256) return false
        }
        preferences.edit { putBoolean(VERIFIED, true) }
        return true
    }

    fun isVerified(): Boolean {
        val preference = preferences.getBoolean(VERIFIED, false)
        val filesValid = hasCompleteFiles()
        Log.i(TAG, "verified=$preference filesValid=$filesValid dir=${modelDir.absolutePath} " +
            FILES.joinToString { model -> "${model.name}:${File(modelDir, model.name).length()}" })
        return preference && filesValid
    }

    fun hasCompleteFiles(): Boolean = FILES.all { model ->
        File(modelDir, model.name).let { it.isFile && it.length() == model.bytes }
    }

    fun cancel() {
        generation.incrementAndGet()
        state.set(State.Idle)
        preferences.edit { remove(VERIFIED) }
        FILES.forEach { model ->
            File(modelDir, "${model.name}.part").delete()
            File(modelDir, model.name).takeIf { it.exists() && it.length() != model.bytes }?.delete()
        }
    }

    private fun downloadOne(model: ModelFile, index: Int, token: Long) {
        val destination = File(modelDir, model.name)
        if (destination.isFile && destination.length() == model.bytes && sha256(destination) == model.sha256) {
            publishProgress(index, model.bytes, token)
            return
        }

        destination.delete()
        val partial = File(modelDir, "${model.name}.part")
        if (partial.length() > model.bytes) partial.delete()
        if (partial.length() == model.bytes && sha256(partial) == model.sha256) {
            check(partial.renameTo(destination)) { "Could not publish ${model.name}" }
            publishProgress(index, model.bytes, token)
            return
        }

        var lastError: Throwable? = null
        repeat(MAX_ATTEMPTS) { attempt ->
            check(token == generation.get()) { "Download cancelled" }
            runCatching {
                downloadAttempt(model, partial, index, token)
                check(partial.length() == model.bytes) {
                    "${model.name} has ${partial.length()} bytes; expected ${model.bytes}"
                }
                check(sha256(partial) == model.sha256) { "SHA-256 mismatch for ${model.name}" }
                check(partial.renameTo(destination)) { "Could not publish ${model.name}" }
                publishProgress(index, model.bytes, token)
                return
            }.onFailure { error ->
                lastError = error
                Log.w(TAG, "Attempt ${attempt + 1}/$MAX_ATTEMPTS failed for ${model.name}", error)
                if (attempt + 1 < MAX_ATTEMPTS) Thread.sleep(1_000L * (attempt + 1))
            }
        }
        throw lastError ?: IllegalStateException("Download failed for ${model.name}")
    }

    private fun downloadAttempt(model: ModelFile, partial: File, index: Int, token: Long) {
        var offset = partial.length()
        val connection = (URL(downloadUrl(model.name)).openConnection() as HttpURLConnection).apply {
            instanceFollowRedirects = true
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            setRequestProperty("Accept-Encoding", "identity")
            setRequestProperty("User-Agent", "QMX-Voice-Lab/Android")
            if (offset > 0) setRequestProperty("Range", "bytes=$offset-")
        }
        try {
            val response = connection.responseCode
            check(response == HttpURLConnection.HTTP_OK || response == HttpURLConnection.HTTP_PARTIAL) {
                "HTTP $response for ${model.name}"
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
        val prior = FILES.take(index).sumOf { model ->
            val complete = File(modelDir, model.name)
            if (complete.length() == model.bytes) model.bytes else 0L
        }
        state.set(State.Active((prior + currentFileBytes).coerceAtMost(totalBytes), totalBytes))
    }

    private fun existingPartialBytes(): Long = FILES.sumOf { model ->
        val complete = File(modelDir, model.name)
        if (complete.length() == model.bytes) model.bytes
        else File(modelDir, "${model.name}.part").length().coerceAtMost(model.bytes)
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
        val BACKBONE = ModelFile(
            "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
            1_847_874_400L,
            "ac7931aeb2e7aad1a6ed6602d353a5679c9d096b18ce8204ac730a8408d572e1",
        )
        val MMPROJ = ModelFile(
            "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
            446_422_912L,
            "6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2",
        )
        private val FILES = listOf(BACKBONE, MMPROJ)
        private const val REPO = "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF"
        private const val PREFERENCES = "voice_model_download"
        private const val VERIFIED = "verified"
        private const val TAG = "QmxVoiceModels"
        private const val MAX_ATTEMPTS = 3
        private const val CONNECT_TIMEOUT_MS = 30_000
        private const val READ_TIMEOUT_MS = 60_000
        private const val IO_BUFFER_BYTES = 4 * 1024 * 1024
        private const val FREE_SPACE_HEADROOM = 768L * 1024 * 1024
        private val DOWNLOAD_LOCK = Any()
        private val executor = Executors.newSingleThreadExecutor()
        private val generation = AtomicLong(0)
        private val state = AtomicReference<State>(State.Idle)

        private fun downloadUrl(name: String) = "https://huggingface.co/$REPO/resolve/main/$name?download=true"
        fun formatBytes(bytes: Long): String = when {
            bytes >= 1_000_000_000L -> "%.2f GB".format(bytes / 1_000_000_000.0)
            bytes >= 1_000_000L -> "%.1f MB".format(bytes / 1_000_000.0)
            else -> "$bytes B"
        }
    }
}
