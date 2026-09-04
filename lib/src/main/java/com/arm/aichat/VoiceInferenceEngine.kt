package com.arm.aichat

import android.content.Context
import android.system.Os
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

data class VoiceQmxStatus(
    val selected: Boolean,
    val gemmExecuted: Boolean,
    val gemvExecuted: Boolean,
    val bufferMiB: Double,
    val layers: Int,
    val totalLayers: Int,
) {
    val inferenceConfirmed: Boolean
        get() = selected && gemmExecuted && gemvExecuted
}

data class VoiceLatency(
    val promptMs: Double,
    val firstFrameMs: Double,
    val firstPcmMs: Double,
    val generationMs: Double,
    val wavMs: Double,
    val totalMs: Double,
    val audioMs: Double,
    val realTimeFactor: Double,
    val frames: Int,
    val sampleRate: Int,
    val threads: Int,
    val qmx: VoiceQmxStatus,
)

fun interface PcmChunkListener {
    fun onPcmChunk(samples: ShortArray, sampleRate: Int, isFinal: Boolean)
}

class VoiceInferenceEngine private constructor(context: Context) {
    private val nativeLibDir = context.applicationInfo.nativeLibraryDir
    @Volatile
    private var initialized = false
    @Volatile
    private var loaded = false

    val isLoaded: Boolean
        get() = loaded

    private external fun nativeInit(nativeLibDir: String)
    private external fun nativeLoad(
        backbonePath: String,
        mmprojPath: String,
        threads: Int,
        qmxLayers: Int,
    ): Int
    private external fun nativeSynthesize(
        prompt: String,
        language: String,
        outputPath: String,
        threads: Int,
        maxFrames: Int,
        pcmChunkListener: PcmChunkListener?,
    ): String?
    private external fun nativeQmxStatus(): String
    private external fun nativeActivateTelemetry()
    private external fun nativeLastError(): String
    private external fun nativeUnload()
    private external fun nativeShutdown()

    suspend fun initialize() = withContext(Dispatchers.IO) {
        if (initialized) return@withContext
        Os.setenv("GGML_KLEIDIAI_SME", "1", true)
        System.loadLibrary("ai-chat")
        nativeInit(nativeLibDir)
        initialized = true
    }

    suspend fun load(
        backbone: File,
        mmproj: File,
        threads: Int = 1,
        qmxLayers: Int = ALL_QMX_LAYERS,
    ) = withContext(Dispatchers.IO) {
        check(initialized) { "Initialize the voice runtime first" }
        require(backbone.isFile) { "Backbone model not found" }
        require(mmproj.isFile) { "Audio projector model not found" }
        require(qmxLayers >= ALL_QMX_LAYERS) { "QMX layer count cannot be negative" }
        val result = nativeLoad(backbone.absolutePath, mmproj.absolutePath, threads, qmxLayers)
        check(result == 0) { nativeLastError().ifBlank { "Native model load failed ($result)" } }
        loaded = true
    }

    /** Routes process-wide llama.cpp and multimodal logs back to the voice proof tracker. */
    fun activateTelemetry() {
        check(initialized) { "Initialize the voice runtime first" }
        nativeActivateTelemetry()
    }

    suspend fun synthesize(
        text: String,
        output: File,
        threads: Int,
        language: String = "en",
        maxFrames: Int = 240,
        pcmChunkListener: PcmChunkListener? = null,
    ): VoiceLatency = withContext(Dispatchers.IO) {
        check(loaded) { "Load the model before synthesizing" }
        require(text.isNotBlank()) { "Enter text to speak" }
        output.parentFile?.mkdirs()
        val raw = nativeSynthesize(
            text,
            language,
            output.absolutePath,
            threads,
            maxFrames,
            pcmChunkListener,
        )
            ?: error(nativeLastError().ifBlank { "Voice synthesis failed" })
        parseLatency(JSONObject(raw))
    }

    fun qmxStatus(): VoiceQmxStatus = parseQmx(JSONObject(nativeQmxStatus()))

    fun unload() {
        if (loaded) nativeUnload()
        loaded = false
    }

    fun destroy() {
        if (initialized) nativeShutdown()
        loaded = false
        initialized = false
    }

    private fun parseLatency(json: JSONObject) = VoiceLatency(
        promptMs = json.getDouble("promptMs"),
        firstFrameMs = json.getDouble("firstFrameMs"),
        firstPcmMs = json.optDouble("firstPcmMs", json.getDouble("totalMs")),
        generationMs = json.getDouble("generationMs"),
        wavMs = json.getDouble("wavMs"),
        totalMs = json.getDouble("totalMs"),
        audioMs = json.getDouble("audioMs"),
        realTimeFactor = json.getDouble("rtf"),
        frames = json.getInt("frames"),
        sampleRate = json.getInt("sampleRate"),
        threads = json.getInt("threads"),
        qmx = parseQmx(json.getJSONObject("qmx")),
    )

    private fun parseQmx(json: JSONObject) = VoiceQmxStatus(
        selected = json.getBoolean("selected"),
        gemmExecuted = json.getBoolean("gemmExecuted"),
        gemvExecuted = json.getBoolean("gemvExecuted"),
        bufferMiB = json.getDouble("bufferMiB"),
        layers = json.getInt("layers"),
        totalLayers = json.optInt("totalLayers", json.getInt("layers")),
    )

    companion object {
        const val ALL_QMX_LAYERS = 0

        @Volatile
        private var instance: VoiceInferenceEngine? = null

        fun getInstance(context: Context): VoiceInferenceEngine =
            instance ?: synchronized(this) {
                instance ?: VoiceInferenceEngine(context.applicationContext).also { instance = it }
            }
    }
}
