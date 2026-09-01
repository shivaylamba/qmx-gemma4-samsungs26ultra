package com.example.qmxgemma

import kotlin.math.floor
import kotlin.math.max

internal data class QmxLayerPlan(
    val selectedLayers: Int,
    val repackBudgetMiB: Double,
    val observedMiBPerLayer: Double,
)

/**
 * Converts live Android memory headroom plus a small measured QMX probe into a
 * model-specific layer count. The probe makes this independent of Gemma 3 vs.
 * Gemma 4 packing density and avoids assuming that GGUF bytes equal packed
 * KleidiAI bytes.
 */
internal object QmxLayerPlanner {
    private const val MIB = 1024.0 * 1024.0
    private const val BASE_APP_AND_CONTEXT_RESERVE_MIB = 1536.0
    private const val PACKING_SAFETY_FACTOR = 1.25

    fun calculate(
        availableMemoryBytes: Long,
        lowMemoryThresholdBytes: Long,
        modelBytes: Long,
        totalLayers: Int,
        probeLayers: Int,
        probeBufferMiB: Double,
    ): QmxLayerPlan {
        require(availableMemoryBytes >= 0)
        require(lowMemoryThresholdBytes >= 0)
        require(modelBytes >= 0)
        require(totalLayers > 0)
        require(probeLayers in 1..totalLayers)
        require(probeBufferMiB > 0.0)

        val reserveMiB = max(
            BASE_APP_AND_CONTEXT_RESERVE_MIB,
            (lowMemoryThresholdBytes / MIB) * 2.0,
        )
        val availableMiB = availableMemoryBytes / MIB
        val modelMiB = modelBytes / MIB
        val minimumBudgetMiB = probeBufferMiB
        val repackBudgetMiB = max(
            minimumBudgetMiB,
            availableMiB - modelMiB - reserveMiB,
        )
        val observedMiBPerLayer = probeBufferMiB / probeLayers
        val safeMiBPerLayer = observedMiBPerLayer * PACKING_SAFETY_FACTOR
        val selectedLayers = floor(repackBudgetMiB / safeMiBPerLayer)
            .toInt()
            .coerceIn(probeLayers, totalLayers)

        return QmxLayerPlan(
            selectedLayers = selectedLayers,
            repackBudgetMiB = repackBudgetMiB,
            observedMiBPerLayer = observedMiBPerLayer,
        )
    }
}
