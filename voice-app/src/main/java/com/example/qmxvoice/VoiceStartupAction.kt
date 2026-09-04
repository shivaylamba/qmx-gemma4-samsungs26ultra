package com.example.qmxvoice

internal enum class VoiceStartupAction {
    MONITOR_DOWNLOAD,
    LOAD_MODEL,
    VERIFY_FILES,
    SHOW_DOWNLOAD,
    SHOW_FAILURE,
}

internal fun voiceStartupAction(
    state: VoiceModelDownload.State,
    verified: Boolean,
    completeFiles: Boolean,
): VoiceStartupAction = when (state) {
    is VoiceModelDownload.State.Active -> VoiceStartupAction.MONITOR_DOWNLOAD
    is VoiceModelDownload.State.Failed -> VoiceStartupAction.SHOW_FAILURE
    VoiceModelDownload.State.Complete,
    VoiceModelDownload.State.Idle,
    -> when {
        verified -> VoiceStartupAction.LOAD_MODEL
        completeFiles -> VoiceStartupAction.VERIFY_FILES
        else -> VoiceStartupAction.SHOW_DOWNLOAD
    }
}
