package com.example.qmxassistant

import org.junit.Assert.assertEquals
import org.junit.Test

class AssistantStartupActionTest {
    @Test
    fun `active download is monitored after recreation`() {
        assertEquals(
            AssistantStartupAction.MONITOR_DOWNLOAD,
            assistantStartupAction(
                AssistantModelRepository.State.Active(10, 100),
                verified = false,
                completeFiles = false,
            ),
        )
    }

    @Test
    fun `copied complete files are verified before loading`() {
        assertEquals(
            AssistantStartupAction.VERIFY_FILES,
            assistantStartupAction(
                AssistantModelRepository.State.Idle,
                verified = false,
                completeFiles = true,
            ),
        )
    }

    @Test
    fun `verified files load immediately`() {
        assertEquals(
            AssistantStartupAction.LOAD_MODELS,
            assistantStartupAction(
                AssistantModelRepository.State.Complete,
                verified = true,
                completeFiles = true,
            ),
        )
    }
}
