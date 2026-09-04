package com.example.qmxassistant

import kotlin.math.floor

internal data class VoiceQmxPlan(
    val layers: Int,
    val totalLayers: Int,
    val budgetMiB: Double,
    val measuredMiBPerLayer: Double,
)

internal fun planVoiceQmxLayers(
    availableBytes: Long,
    lowMemoryThresholdBytes: Long,
    modelBytes: Long,
    probeLayers: Int,
    probeBufferMiB: Double,
    totalLayers: Int,
): VoiceQmxPlan {
    require(probeLayers > 0 && probeBufferMiB > 0.0 && totalLayers >= probeLayers)
    val oneMiB = 1024.0 * 1024.0
    val measuredMiBPerLayer = probeBufferMiB / probeLayers
    val safetyBytes = (availableBytes * 0.15).toLong()
    val runtimeReserveBytes = 1024L * 1024L * 1024L
    val budgetBytes = (
        availableBytes - lowMemoryThresholdBytes - modelBytes - safetyBytes - runtimeReserveBytes
    ).coerceAtLeast(0L)
    val budgetMiB = budgetBytes / oneMiB
    val planned = floor(budgetMiB / measuredMiBPerLayer).toInt()
        .coerceIn(probeLayers, totalLayers)
    return VoiceQmxPlan(planned, totalLayers, budgetMiB, measuredMiBPerLayer)
}

internal fun fasterTtsThreads(oneThreadRtf: Double, fourThreadRtf: Double): Int =
    if (fourThreadRtf < oneThreadRtf) 4 else 1
