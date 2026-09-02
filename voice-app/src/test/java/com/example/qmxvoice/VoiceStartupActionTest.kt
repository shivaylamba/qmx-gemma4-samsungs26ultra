package com.example.qmxvoice

import org.junit.Assert.assertEquals
import org.junit.Test

class VoiceStartupActionTest {
    @Test
    fun activeDownloadIsMonitoredAfterRecreation() {
        assertEquals(
            VoiceStartupAction.MONITOR_DOWNLOAD,
            voiceStartupAction(
                VoiceModelDownload.State.Active(128, 1024),
                verified = false,
                completeFiles = false,
            ),
        )
    }

    @Test
    fun completeUnverifiedFilesAreVerified() {
        assertEquals(
            VoiceStartupAction.VERIFY_FILES,
            voiceStartupAction(
                VoiceModelDownload.State.Complete,
                verified = false,
                completeFiles = true,
            ),
        )
    }

    @Test
    fun verifiedFilesLoadImmediately() {
        assertEquals(
            VoiceStartupAction.LOAD_MODEL,
            voiceStartupAction(
                VoiceModelDownload.State.Idle,
                verified = true,
                completeFiles = true,
            ),
        )
    }
}
