package com.example.qmxgemma

import com.arm.aichat.InferenceEngine

internal enum class EngineStartupAction {
    WAIT,
    LOAD_MODEL,
    RESTORE_MODEL,
    SHOW_ERROR,
}

internal fun engineStartupAction(state: InferenceEngine.State): EngineStartupAction = when (state) {
    InferenceEngine.State.Initialized -> EngineStartupAction.LOAD_MODEL
    InferenceEngine.State.ModelReady -> EngineStartupAction.RESTORE_MODEL
    is InferenceEngine.State.Error -> EngineStartupAction.SHOW_ERROR
    else -> EngineStartupAction.WAIT
}
