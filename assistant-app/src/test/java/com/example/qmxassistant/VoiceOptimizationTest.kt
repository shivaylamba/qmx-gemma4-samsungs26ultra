package com.example.qmxassistant

import org.junit.Assert.assertEquals
import org.junit.Test

class VoiceOptimizationTest {
    @Test
    fun `planner uses measured packing density and keeps memory reserves`() {
        val plan = planVoiceQmxLayers(
            availableBytes = 7L * 1024 * 1024 * 1024,
            lowMemoryThresholdBytes = 512L * 1024 * 1024,
            modelBytes = 2300L * 1024 * 1024,
            probeLayers = 4,
            probeBufferMiB = 400.0,
            totalLayers = 28,
        )

        assertEquals(22, plan.layers)
        assertEquals(100.0, plan.measuredMiBPerLayer, 0.01)
    }

    @Test
    fun `planner never selects fewer layers than the successful probe`() {
        val plan = planVoiceQmxLayers(
            availableBytes = 2L * 1024 * 1024 * 1024,
            lowMemoryThresholdBytes = 1024L * 1024 * 1024,
            modelBytes = 2300L * 1024 * 1024,
            probeLayers = 4,
            probeBufferMiB = 400.0,
            totalLayers = 28,
        )

        assertEquals(4, plan.layers)
    }

    @Test
    fun `thread selector chooses the lower real time factor`() {
        assertEquals(4, fasterTtsThreads(oneThreadRtf = 3.0, fourThreadRtf = 2.0))
        assertEquals(1, fasterTtsThreads(oneThreadRtf = 1.5, fourThreadRtf = 2.0))
    }
}
