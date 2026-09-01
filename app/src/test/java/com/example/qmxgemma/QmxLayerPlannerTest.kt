package com.example.qmxgemma

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class QmxLayerPlannerTest {
    @Test
    fun selectsAllLayersForSmallModelWithAmpleMemory() {
        val plan = QmxLayerPlanner.calculate(
            availableMemoryBytes = gib(7.0),
            lowMemoryThresholdBytes = mib(512.0),
            modelBytes = mib(300.0),
            totalLayers = 18,
            probeLayers = 2,
            probeBufferMiB = 40.0,
        )

        assertEquals(18, plan.selectedLayers)
    }

    @Test
    fun adaptsToObservedPackingDensityAndCurrentMemory() {
        val densePacking = QmxLayerPlanner.calculate(
            availableMemoryBytes = gib(7.0),
            lowMemoryThresholdBytes = mib(512.0),
            modelBytes = gib(3.84),
            totalLayers = 34,
            probeLayers = 2,
            probeBufferMiB = 360.0,
        )
        val lighterPacking = QmxLayerPlanner.calculate(
            availableMemoryBytes = gib(7.0),
            lowMemoryThresholdBytes = mib(512.0),
            modelBytes = gib(4.61),
            totalLayers = 35,
            probeLayers = 2,
            probeBufferMiB = 144.0,
        )

        assertEquals(7, densePacking.selectedLayers)
        assertEquals(10, lighterPacking.selectedLayers)
        assertTrue(lighterPacking.selectedLayers > densePacking.selectedLayers)
    }

    @Test
    fun neverDropsBelowProbeWhenMemoryIsTight() {
        val plan = QmxLayerPlanner.calculate(
            availableMemoryBytes = gib(4.0),
            lowMemoryThresholdBytes = gib(1.0),
            modelBytes = gib(4.0),
            totalLayers = 34,
            probeLayers = 2,
            probeBufferMiB = 360.0,
        )

        assertEquals(2, plan.selectedLayers)
        assertEquals(360.0, plan.repackBudgetMiB, 0.001)
    }

    private fun mib(value: Double): Long = (value * 1024 * 1024).toLong()
    private fun gib(value: Double): Long = (value * 1024 * 1024 * 1024).toLong()
}
