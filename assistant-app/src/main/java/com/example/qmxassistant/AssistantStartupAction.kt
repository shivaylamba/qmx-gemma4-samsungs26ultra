package com.example.qmxassistant

internal enum class AssistantStartupAction {
    MONITOR_DOWNLOAD,
    VERIFY_FILES,
    LOAD_MODELS,
    SHOW_DOWNLOAD,
    SHOW_FAILURE,
}

internal fun assistantStartupAction(
    state: AssistantModelRepository.State,
    verified: Boolean,
    completeFiles: Boolean,
): AssistantStartupAction = when {
    state is AssistantModelRepository.State.Active -> AssistantStartupAction.MONITOR_DOWNLOAD
    state is AssistantModelRepository.State.Failed -> AssistantStartupAction.SHOW_FAILURE
    verified -> AssistantStartupAction.LOAD_MODELS
    state == AssistantModelRepository.State.Complete || completeFiles -> AssistantStartupAction.VERIFY_FILES
    else -> AssistantStartupAction.SHOW_DOWNLOAD
}
