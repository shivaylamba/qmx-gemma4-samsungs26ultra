package com.example.qmxgemma

import com.arm.aichat.InferenceEngine
import org.junit.Assert.assertEquals
import org.junit.Test

class EngineStartupActionTest {
    @Test
    fun initializedEngineLoadsModel() {
        assertEquals(
            EngineStartupAction.LOAD_MODEL,
            engineStartupAction(InferenceEngine.State.Initialized),
        )
    }

    @Test
    fun modelReadyEngineRestoresExistingModel() {
        assertEquals(
            EngineStartupAction.RESTORE_MODEL,
            engineStartupAction(InferenceEngine.State.ModelReady),
        )
    }

    @Test
    fun transitionalEngineKeepsWaiting() {
        assertEquals(
            EngineStartupAction.WAIT,
            engineStartupAction(InferenceEngine.State.LoadingModel),
        )
    }
}
