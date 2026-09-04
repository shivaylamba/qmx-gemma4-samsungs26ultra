package com.example.qmxassistant

internal enum class VoiceBackend(
    val preferenceValue: String,
    val displayName: String,
) {
    QWEN_QMX("qwen-qmx", "Qwen3-TTS 1.7B · QMX/SME CPU"),
    KITTEN_CPU("kitten-cpu", "KittenTTS Nano 0.8 FP32 · ONNX CPU"),
    ;

    companion object {
        fun fromPreference(value: String?): VoiceBackend =
            entries.firstOrNull { it.preferenceValue == value } ?: QWEN_QMX
    }
}

internal data class SpeechLatency(
    val backend: VoiceBackend,
    val firstPcmMs: Double,
    val totalMs: Double,
    val audioMs: Double,
    val realTimeFactor: Double,
    val threads: Int,
    val qmxLayers: Int = 0,
    val runtime: String,
)
