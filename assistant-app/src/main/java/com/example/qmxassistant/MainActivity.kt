package com.example.qmxassistant

import android.app.ActivityManager
import android.content.Context
import android.media.MediaPlayer
import android.os.Bundle
import android.os.SystemClock
import android.system.Os
import android.view.View
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.widget.AdapterView
import android.widget.Button
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.edit
import androidx.lifecycle.lifecycleScope
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.VoiceInferenceEngine
import com.arm.aichat.VoiceLatency
import com.arm.aichat.VoiceQmxStatus
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var statusBadge: TextView
    private lateinit var statusText: TextView
    private lateinit var modelText: TextView
    private lateinit var metricsText: TextView
    private lateinit var transcriptText: TextView
    private lateinit var transcriptScroll: ScrollView
    private lateinit var progress: ProgressBar
    private lateinit var downloadButton: Button
    private lateinit var sendButton: Button
    private lateinit var newChatButton: Button
    private lateinit var playButton: Button
    private lateinit var benchmarkButton: Button
    private lateinit var promptInput: TextInputEditText
    private lateinit var speakSwitch: MaterialSwitch
    private lateinit var voiceBackendSpinner: Spinner

    private lateinit var repository: AssistantModelRepository
    private lateinit var voice: VoiceInferenceEngine
    private lateinit var kittenVoice: KittenTtsEngine
    private lateinit var llm: InferenceEngine
    private val session = AssistantSession.process
    private var monitorJob: Job? = null
    private var workJob: Job? = null
    private var player: MediaPlayer? = null
    private var pcmPlayer: StreamingPcmPlayer? = null
    private var selectedVoiceBackend = VoiceBackend.QWEN_QMX
    private val preferences by lazy {
        getSharedPreferences(VOICE_PREFERENCES, Context.MODE_PRIVATE)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        statusBadge = findViewById(R.id.statusBadge)
        statusText = findViewById(R.id.statusText)
        modelText = findViewById(R.id.modelText)
        metricsText = findViewById(R.id.metricsText)
        transcriptText = findViewById(R.id.transcriptText)
        transcriptScroll = findViewById(R.id.transcriptScroll)
        progress = findViewById(R.id.progress)
        downloadButton = findViewById(R.id.downloadButton)
        sendButton = findViewById(R.id.sendButton)
        newChatButton = findViewById(R.id.newChatButton)
        playButton = findViewById(R.id.playButton)
        benchmarkButton = findViewById(R.id.benchmarkButton)
        promptInput = findViewById(R.id.promptInput)
        speakSwitch = findViewById(R.id.speakSwitch)
        voiceBackendSpinner = findViewById(R.id.voiceBackendSpinner)

        repository = AssistantModelRepository(applicationContext)
        voice = VoiceInferenceEngine.getInstance(applicationContext)
        kittenVoice = KittenTtsEngine()
        selectedVoiceBackend = VoiceBackend.fromPreference(
            preferences.getString(PREFERRED_VOICE_BACKEND, null),
        )
        voiceBackendSpinner.setSelection(selectedVoiceBackend.ordinal, false)
        voiceBackendSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selected = VoiceBackend.entries.getOrElse(position) { VoiceBackend.QWEN_QMX }
                if (selected == selectedVoiceBackend) return
                selectedVoiceBackend = selected
                preferences.edit { putString(PREFERRED_VOICE_BACKEND, selected.preferenceValue) }
                if (assistantReady && workJob?.isActive != true) switchVoiceBackend()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
        Os.setenv(QMX_SWITCH, "1", true)
        Os.setenv(LLM_QMX_LAYERS_VARIABLE, LLM_QMX_LAYERS.toString(), true)

        renderTranscript()
        setControlsEnabled(false)
        downloadButton.setOnClickListener { toggleDownload() }
        sendButton.setOnClickListener { runPrompt() }
        newChatButton.setOnClickListener { clearConversation() }
        playButton.setOnClickListener { playLastAudio() }
        benchmarkButton.setOnClickListener { runVoiceBenchmark() }
        promptInput.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                runPrompt()
                true
            } else {
                false
            }
        }

        lifecycleScope.launch { initializeOrResume() }
    }

    private suspend fun initializeOrResume() {
        runCatching { voice.initialize() }
            .onFailure { showError(getString(R.string.load_failed, it.message ?: it.javaClass.simpleName)) }
            .onSuccess {
                val state = repository.query()
                when (assistantStartupAction(state, repository.isVerified(), repository.hasCompleteFiles())) {
                    AssistantStartupAction.MONITOR_DOWNLOAD -> monitorDownload()
                    AssistantStartupAction.VERIFY_FILES -> verifyAndLoad()
                    AssistantStartupAction.LOAD_MODELS -> loadAssistant()
                    AssistantStartupAction.SHOW_DOWNLOAD -> showDownloadReady()
                    AssistantStartupAction.SHOW_FAILURE -> {
                        showDownloadReady()
                        showError(
                            getString(
                                R.string.download_failed,
                                (state as AssistantModelRepository.State.Failed).reason,
                            ),
                        )
                    }
                }
            }
    }

    private fun toggleDownload() {
        if (repository.query() is AssistantModelRepository.State.Active) {
            repository.cancel()
            monitorJob?.cancel()
            showDownloadReady()
            return
        }
        runCatching { repository.enqueue() }
            .onFailure { showError(getString(R.string.download_failed, it.message ?: it.javaClass.simpleName)) }
            .onSuccess { monitorDownload() }
    }

    private fun monitorDownload() {
        monitorJob?.cancel()
        monitorJob = lifecycleScope.launch {
            while (isActive) {
                when (val state = repository.query()) {
                    AssistantModelRepository.State.Idle -> {
                        showDownloadReady()
                        return@launch
                    }
                    is AssistantModelRepository.State.Active -> showDownloadProgress(state)
                    AssistantModelRepository.State.Complete -> {
                        verifyAndLoad()
                        return@launch
                    }
                    is AssistantModelRepository.State.Failed -> {
                        showDownloadReady()
                        showError(getString(R.string.download_failed, state.reason))
                        return@launch
                    }
                }
                delay(DOWNLOAD_POLL_MS)
            }
        }
    }

    private fun showDownloadProgress(state: AssistantModelRepository.State.Active) {
        val fraction = (state.downloadedBytes.toDouble() / state.totalBytes).coerceIn(0.0, 1.0)
        progress.visibility = View.VISIBLE
        progress.isIndeterminate = false
        progress.max = 1000
        progress.progress = (fraction * progress.max).toInt()
        statusText.text = getString(
            R.string.download_progress,
            (fraction * 100).toInt(),
            AssistantModelRepository.formatBytes(state.downloadedBytes),
            AssistantModelRepository.formatBytes(state.totalBytes),
        )
        downloadButton.text = getString(R.string.cancel_download)
        downloadButton.isEnabled = true
        voiceBackendSpinner.isEnabled = false
    }

    private suspend fun verifyAndLoad() {
        setBusy(getString(R.string.verifying))
        val verified = withContext(Dispatchers.IO) { repository.verifyAndRemember() }
        if (verified) loadAssistant() else {
            showDownloadReady()
            showError(getString(R.string.download_failed, "checksum mismatch"))
        }
    }

    private suspend fun loadAssistant() = modelLoadMutex.withLock {
        if (assistantReady) {
            llm = AiChat.getInferenceEngine(applicationContext)
            if (llm.state.value is InferenceEngine.State.ModelReady) {
                if (selectedVoiceBackend == VoiceBackend.KITTEN_CPU) ensureKittenLoaded()
                else withContext(Dispatchers.IO) { kittenVoice.close() }
                showReady()
                return@withLock
            }
            assistantReady = false
        }

        setBusy(getString(R.string.loading_voice))
        runCatching {
            if (selectedVoiceBackend == VoiceBackend.QWEN_QMX) proveVoiceQmxOnce()
            loadLlmWithHistory()
            if (selectedVoiceBackend == VoiceBackend.KITTEN_CPU) ensureKittenLoaded()
            assistantReady = true
        }.onFailure { error ->
            assistantReady = false
            runCatching { if (voice.isLoaded) voice.unload() }
            showError(getString(R.string.load_failed, error.message ?: error.javaClass.simpleName))
        }.onSuccess { showReady() }
    }

    private suspend fun ensureKittenLoaded(threads: Int = preferredKittenThreads()) {
        updateBusyMessage(getString(R.string.loading_kitten))
        withContext(Dispatchers.IO) {
            kittenVoice.load(
                repository.kittenModelFile,
                repository.kittenVoicesFile,
                repository.kittenRulesFile,
                repository.kittenListFile,
                threads,
            )
        }
    }

    private fun preferredKittenThreads(): Int =
        preferences.getInt(PREFERRED_KITTEN_THREADS, DEFAULT_TTS_THREADS).coerceIn(1, 8)

    private fun switchVoiceBackend() {
        workJob = lifecycleScope.launch {
            assistantReady = false
            setBusy(getString(R.string.loading_voice))
            runCatching {
                when (selectedVoiceBackend) {
                    VoiceBackend.KITTEN_CPU -> ensureKittenLoaded()
                    VoiceBackend.QWEN_QMX -> {
                        withContext(Dispatchers.IO) { kittenVoice.close() }
                        if (lastVoiceProof?.inferenceConfirmed != true) {
                            lastLlmAcceleration = llm.accelerationInfo()
                            withContext(Dispatchers.IO) { llm.cleanUp() }
                            delay(MODEL_HANDOFF_DELAY_MS)
                            proveVoiceQmxOnce()
                            loadLlmWithHistory()
                        }
                    }
                }
            }.onFailure { error ->
                showError(getString(R.string.load_failed, error.message ?: error.javaClass.simpleName))
            }.onSuccess {
                assistantReady = true
                showReady()
                statusText.text = getString(R.string.voice_backend_ready, selectedVoiceBackend.displayName)
            }
            progress.visibility = View.GONE
            setControlsEnabled(true)
        }
    }

    private suspend fun proveVoiceQmxOnce() {
        if (lastVoiceProof?.inferenceConfirmed == true) return
        voice.activateTelemetry()
        try {
            updateBusyMessage(getString(R.string.loading_voice))
            voice.load(
                repository.voiceBackboneFile,
                repository.voiceProjectorFile,
                threads = 1,
                qmxLayers = VOICE_QMX_PROBE_LAYERS,
            )
            updateBusyMessage(getString(R.string.warming_voice))
            val result = voice.synthesize(
                text = VOICE_WARMUP_TEXT,
                output = File(cacheDir, "voice-warmup.wav"),
                threads = 1,
                maxFrames = VOICE_WARMUP_FRAMES,
            )
            lastVoiceProof = result.qmx
            check(result.qmx.inferenceConfirmed) {
                "Voice QMX execution was not proven (GEMM=${result.qmx.gemmExecuted}, GEMV=${result.qmx.gemvExecuted})"
            }
        } finally {
            if (voice.isLoaded) withContext(Dispatchers.IO) { voice.unload() }
            delay(MODEL_HANDOFF_DELAY_MS)
        }
        lastVoicePlan = calculateVoiceQmxPlan()
    }

    private fun calculateVoiceQmxPlan(): VoiceQmxPlan {
        val proof = checkNotNull(lastVoiceProof) { "Voice QMX calibration is unavailable" }
        val memory = ActivityManager.MemoryInfo()
        (getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(memory)
        return planVoiceQmxLayers(
            availableBytes = memory.availMem,
            lowMemoryThresholdBytes = memory.threshold,
            modelBytes = repository.voiceBackboneFile.length() + repository.voiceProjectorFile.length(),
            probeLayers = proof.layers,
            probeBufferMiB = proof.bufferMiB,
            totalLayers = proof.totalLayers,
        ).also { lastVoicePlan = it }
    }

    private suspend fun loadLlmWithHistory() {
        if (!::llm.isInitialized) llm = AiChat.getInferenceEngine(applicationContext)
        var state = llm.state.first {
            it is InferenceEngine.State.Initialized ||
                it is InferenceEngine.State.ModelReady ||
                it is InferenceEngine.State.Error
        }
        if (state is InferenceEngine.State.Error) {
            withContext(Dispatchers.IO) { llm.cleanUp() }
            state = llm.state.value
        }
        if (state is InferenceEngine.State.ModelReady) {
            lastLlmAcceleration = llm.accelerationInfo()
            return
        }
        check(state is InferenceEngine.State.Initialized) { "LLM runtime is not ready" }

        updateBusyMessage(getString(R.string.loading_llm))
        llm.activateTelemetry()
        Os.setenv(LLM_QMX_LAYERS_VARIABLE, LLM_QMX_LAYERS.toString(), true)
        llm.loadModel(repository.gemmaFile.absolutePath)
        llm.setSystemPrompt(systemPromptWithHistory())
        check(llm.state.value is InferenceEngine.State.ModelReady) { "Gemma did not become ready" }
        lastLlmAcceleration = llm.accelerationInfo()
    }

    private fun systemPromptWithHistory(): String {
        val history = session.conversationContext(MAX_HISTORY_CHARACTERS)
        return if (history.isBlank()) SYSTEM_PROMPT else
            "$SYSTEM_PROMPT\n\nConversation so far:\n$history\nContinue this conversation consistently."
    }

    private fun runPrompt() {
        if (!assistantReady || workJob?.isActive == true) return
        val prompt = promptInput.text?.toString()?.trim().orEmpty()
        if (prompt.isBlank()) return
        promptInput.text?.clear()
        session.addUser(prompt)
        val assistantIndex = session.beginAssistant(getString(R.string.assistant_placeholder))
        renderTranscript()

        workJob = lifecycleScope.launch {
            setBusy(getString(R.string.thinking))
            val started = SystemClock.elapsedRealtime()
            var firstTokenAt = 0L
            val response = StringBuilder()
            runCatching {
                llm.sendUserPrompt(prompt, predictLength = MAX_RESPONSE_TOKENS).collect { token ->
                    if (firstTokenAt == 0L) firstTokenAt = SystemClock.elapsedRealtime()
                    response.append(token)
                    session.updateAssistant(assistantIndex, response.toString())
                    renderTranscript()
                }
                val finalText = response.toString().trim().ifEmpty { "No response was generated." }
                session.updateAssistant(assistantIndex, finalText)
                renderTranscript()
                val llmFinished = SystemClock.elapsedRealtime()

                val voiceLatency = if (speakSwitch.isChecked) {
                    synthesizeWithSelectedVoice(finalText)
                } else null
                showTurnMetrics(started, firstTokenAt, llmFinished, voiceLatency)
                statusBadge.text = getString(R.string.ready_badge)
                statusText.text = getString(
                    if (voiceLatency == null) R.string.answer_text_ready else R.string.answer_ready,
                )
            }.onFailure { error ->
                if (error !is CancellationException) {
                    session.updateAssistant(
                        assistantIndex,
                        getString(R.string.inference_failed, error.message ?: error.javaClass.simpleName),
                    )
                    renderTranscript()
                    showError(getString(R.string.inference_failed, error.message ?: error.javaClass.simpleName))
                }
            }
            setControlsEnabled(true)
            progress.visibility = View.GONE
        }
    }

    private suspend fun synthesizeWithSelectedVoice(text: String): SpeechLatency =
        when (selectedVoiceBackend) {
            VoiceBackend.QWEN_QMX -> synthesizeWithModelHandoff(text)
            VoiceBackend.KITTEN_CPU -> synthesizeWithKitten(text)
        }

    private suspend fun synthesizeWithKitten(text: String): SpeechLatency {
        updateBusyMessage(getString(R.string.speaking_kitten))
        ensureKittenLoaded()
        val output = File(cacheDir, "assistant-answer-kitten.wav")
        val result = withContext(Dispatchers.IO) {
            kittenVoice.synthesize(
                text = text.take(MAX_SPOKEN_CHARACTERS),
                output = output,
            )
        }
        session.lastAudioPath = output.absolutePath
        playAudio(output)
        return SpeechLatency(
            backend = VoiceBackend.KITTEN_CPU,
            firstPcmMs = result.firstPcmMs,
            totalMs = result.totalMs,
            audioMs = result.audioMs,
            realTimeFactor = result.realTimeFactor,
            threads = result.threads,
            runtime = result.runtime,
        )
    }

    private suspend fun synthesizeWithModelHandoff(text: String): SpeechLatency {
        updateBusyMessage(getString(R.string.unloading_llm))
        assistantReady = false
        lastLlmAcceleration = llm.accelerationInfo()
        withContext(Dispatchers.IO) { llm.cleanUp() }
        delay(MODEL_HANDOFF_DELAY_MS)

        var result: VoiceLatency? = null
        var voiceFailure: Throwable? = null
        try {
            updateBusyMessage(getString(R.string.speaking))
            voice.activateTelemetry()
            val plan = calculateVoiceQmxPlan()
            val threads = preferences.getInt(PREFERRED_THREADS, DEFAULT_TTS_THREADS)
            voice.load(
                repository.voiceBackboneFile,
                repository.voiceProjectorFile,
                threads = threads,
                qmxLayers = plan.layers,
            )
            val output = File(cacheDir, "assistant-answer.wav")
            pcmPlayer?.close()
            val streamingPlayer = StreamingPcmPlayer().also { pcmPlayer = it }
            result = voice.synthesize(
                text = text.take(MAX_SPOKEN_CHARACTERS),
                output = output,
                threads = threads,
                maxFrames = MAX_VOICE_FRAMES,
                pcmChunkListener = streamingPlayer,
            )
            lastVoiceProof = result.qmx
            session.lastAudioPath = output.absolutePath
        } catch (error: Throwable) {
            pcmPlayer?.close()
            pcmPlayer = null
            voiceFailure = error
        } finally {
            updateBusyMessage(getString(R.string.restoring_llm))
            if (voice.isLoaded) withContext(Dispatchers.IO) { voice.unload() }
            delay(MODEL_HANDOFF_DELAY_MS)
            loadLlmWithHistory()
            assistantReady = true
        }
        voiceFailure?.let { throw it }
        return checkNotNull(result).let { latency ->
            SpeechLatency(
                backend = VoiceBackend.QWEN_QMX,
                firstPcmMs = latency.firstPcmMs,
                totalMs = latency.totalMs,
                audioMs = latency.audioMs,
                realTimeFactor = latency.realTimeFactor,
                threads = latency.threads,
                qmxLayers = latency.qmx.layers,
                runtime = "llama.cpp · Qualcomm KleidiAI QMX/SME CPU",
            )
        }
    }

    private fun runVoiceBenchmark() {
        if (!assistantReady || workJob?.isActive == true) return
        if (selectedVoiceBackend == VoiceBackend.KITTEN_CPU) {
            runKittenVoiceBenchmark()
            return
        }
        workJob = lifecycleScope.launch {
            setBusy(getString(R.string.benchmark_preparing))
            assistantReady = false
            lastLlmAcceleration = llm.accelerationInfo()
            var benchmarkSummary: String? = null
            runCatching {
                withContext(Dispatchers.IO) { llm.cleanUp() }
                delay(MODEL_HANDOFF_DELAY_MS)
                val plan = calculateVoiceQmxPlan()
                voice.activateTelemetry()
                voice.load(
                    repository.voiceBackboneFile,
                    repository.voiceProjectorFile,
                    threads = 4,
                    qmxLayers = plan.layers,
                )
                updateBusyMessage(getString(R.string.benchmark_one_thread))
                val one = voice.synthesize(
                    VOICE_BENCHMARK_TEXT,
                    File(cacheDir, "assistant-benchmark-1t.wav"),
                    threads = 1,
                    maxFrames = VOICE_BENCHMARK_FRAMES,
                )
                updateBusyMessage(getString(R.string.benchmark_four_threads))
                val four = voice.synthesize(
                    VOICE_BENCHMARK_TEXT,
                    File(cacheDir, "assistant-benchmark-4t.wav"),
                    threads = 4,
                    maxFrames = VOICE_BENCHMARK_FRAMES,
                )
                val selected = fasterTtsThreads(one.realTimeFactor, four.realTimeFactor)
                preferences.edit { putInt(PREFERRED_THREADS, selected) }
                lastVoiceProof = if (selected == 4) four.qmx else one.qmx
                getString(
                    R.string.benchmark_result,
                    one.totalMs / 1000.0,
                    one.realTimeFactor,
                    four.totalMs / 1000.0,
                    four.realTimeFactor,
                    one.realTimeFactor / four.realTimeFactor,
                    selected,
                )
            }.onFailure {
                showError(getString(R.string.inference_failed, it.message ?: it.javaClass.simpleName))
            }.onSuccess { benchmarkSummary = it }
            updateBusyMessage(getString(R.string.restoring_llm))
            if (voice.isLoaded) withContext(Dispatchers.IO) { voice.unload() }
            delay(MODEL_HANDOFF_DELAY_MS)
            runCatching { loadLlmWithHistory() }
                .onSuccess {
                    assistantReady = true
                    statusBadge.text = getString(R.string.ready_badge)
                    statusText.text = getString(
                        if (benchmarkSummary == null) R.string.ready else R.string.benchmark_complete,
                    )
                    benchmarkSummary?.let { metricsText.text = it }
                }
                .onFailure { showError(getString(R.string.load_failed, it.message ?: it.javaClass.simpleName)) }
            progress.visibility = View.GONE
            setControlsEnabled(true)
        }
    }

    private fun runKittenVoiceBenchmark() {
        workJob = lifecycleScope.launch {
            assistantReady = false
            setBusy(getString(R.string.benchmark_preparing))
            var benchmarkSummary: String? = null
            runCatching {
                suspend fun measure(threads: Int, suffix: String): KittenLatency {
                    ensureKittenLoaded(threads)
                    withContext(Dispatchers.IO) {
                        kittenVoice.synthesize(
                            VOICE_WARMUP_TEXT,
                            File(cacheDir, "kitten-warmup-${suffix}.wav"),
                        )
                    }
                    updateBusyMessage(
                        getString(
                            if (threads == 1) R.string.benchmark_one_thread
                            else R.string.benchmark_four_threads,
                        ),
                    )
                    return withContext(Dispatchers.IO) {
                        kittenVoice.synthesize(
                            VOICE_BENCHMARK_TEXT,
                            File(cacheDir, "kitten-benchmark-${suffix}.wav"),
                        )
                    }
                }

                val one = measure(1, "1t")
                val four = measure(4, "4t")
                val selected = fasterTtsThreads(one.realTimeFactor, four.realTimeFactor)
                preferences.edit { putInt(PREFERRED_KITTEN_THREADS, selected) }
                ensureKittenLoaded(selected)
                getString(
                    R.string.benchmark_result,
                    one.totalMs / 1000.0,
                    one.realTimeFactor,
                    four.totalMs / 1000.0,
                    four.realTimeFactor,
                    one.realTimeFactor / four.realTimeFactor,
                    selected,
                )
            }.onFailure {
                showError(getString(R.string.inference_failed, it.message ?: it.javaClass.simpleName))
            }.onSuccess { benchmarkSummary = it }

            assistantReady = true
            progress.visibility = View.GONE
            statusBadge.text = getString(R.string.ready_badge)
            statusText.text = getString(
                if (benchmarkSummary == null) R.string.ready else R.string.benchmark_complete,
            )
            benchmarkSummary?.let { metricsText.text = it }
            setControlsEnabled(true)
        }
    }

    private fun clearConversation() {
        if (!assistantReady || workJob?.isActive == true) return
        workJob = lifecycleScope.launch {
            setBusy(getString(R.string.new_chat))
            runCatching {
                assistantReady = false
                withContext(Dispatchers.IO) { llm.cleanUp() }
                session.clear()
                renderTranscript()
                delay(MODEL_HANDOFF_DELAY_MS)
                loadLlmWithHistory()
                assistantReady = true
                metricsText.text = ""
                statusText.text = getString(R.string.conversation_cleared)
                statusBadge.text = getString(R.string.ready_badge)
            }.onFailure {
                showError(getString(R.string.inference_failed, it.message ?: it.javaClass.simpleName))
            }
            setControlsEnabled(true)
            progress.visibility = View.GONE
        }
    }

    private fun showReady() {
        progress.visibility = View.GONE
        statusBadge.text = getString(R.string.ready_badge)
        statusText.text = getString(R.string.ready)
        downloadButton.visibility = View.GONE
        setControlsEnabled(true)
        metricsText.text = when (selectedVoiceBackend) {
            VoiceBackend.KITTEN_CPU -> getString(
                R.string.kitten_runtime_metrics,
                kittenVoice.runtimeInfo(),
            )
            VoiceBackend.QWEN_QMX -> {
                val voiceStatus = lastVoiceProof ?: voice.qmxStatus()
                getString(
                    R.string.runtime_metrics,
                    lastLlmAcceleration.ifBlank { llm.accelerationInfo() },
                    if (voiceStatus.inferenceConfirmed) "proven" else "not proven",
                    voiceStatus.bufferMiB,
                    voiceStatus.layers,
                    preferences.getInt(PREFERRED_THREADS, DEFAULT_TTS_THREADS),
                    lastVoicePlan?.layers ?: voiceStatus.layers,
                    voiceStatus.totalLayers,
                )
            }
        }
        playButton.isEnabled = session.lastAudioPath?.let(::File)?.isFile == true
    }

    private fun showDownloadReady() {
        assistantReady = false
        progress.visibility = View.GONE
        statusBadge.text = getString(R.string.waiting_badge)
        statusText.text = getString(R.string.models_needed)
        modelText.text = getString(R.string.model_summary)
        downloadButton.visibility = View.VISIBLE
        downloadButton.text = getString(R.string.download_models)
        downloadButton.isEnabled = true
        setControlsEnabled(false)
        voiceBackendSpinner.isEnabled = true
    }

    private fun showTurnMetrics(
        started: Long,
        firstTokenAt: Long,
        llmFinished: Long,
        voiceLatency: SpeechLatency?,
    ) {
        val ttft = (firstTokenAt.takeIf { it > 0 } ?: llmFinished) - started
        metricsText.text = if (voiceLatency == null) {
            "LLM TTFT $ttft ms · total ${llmFinished - started} ms"
        } else if (voiceLatency.backend == VoiceBackend.KITTEN_CPU) {
            getString(
                R.string.kitten_turn_metrics,
                ttft,
                llmFinished - started,
                voiceLatency.firstPcmMs,
                voiceLatency.totalMs / 1000.0,
                voiceLatency.realTimeFactor,
                voiceLatency.threads,
            )
        } else {
            getString(
                R.string.turn_metrics,
                ttft,
                llmFinished - started,
                voiceLatency.firstPcmMs,
                voiceLatency.totalMs / 1000.0,
                voiceLatency.realTimeFactor,
                voiceLatency.threads,
                voiceLatency.qmxLayers,
            )
        }
    }

    private fun renderTranscript() {
        transcriptText.text = if (session.turns.isEmpty()) {
            getString(R.string.welcome)
        } else {
            session.turns.joinToString("\n\n") { turn ->
                val speaker = if (turn.speaker == Speaker.USER) "YOU" else "ASSISTANT"
                "$speaker\n${turn.text}"
            }
        }
        transcriptScroll.post { transcriptScroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun playLastAudio() {
        session.lastAudioPath?.let(::File)?.takeIf(File::isFile)?.let(::playAudio)
    }

    private fun playAudio(file: File) {
        pcmPlayer?.close()
        pcmPlayer = null
        player?.release()
        player = MediaPlayer().apply {
            setDataSource(file.absolutePath)
            setOnCompletionListener { completed ->
                completed.release()
                player = null
            }
            prepare()
            start()
        }
        playButton.isEnabled = true
    }

    private fun setBusy(message: String) {
        progress.visibility = View.VISIBLE
        progress.isIndeterminate = true
        updateBusyMessage(message)
        setControlsEnabled(false)
    }

    private fun updateBusyMessage(message: String) {
        statusText.text = message
    }

    private fun setControlsEnabled(enabled: Boolean) {
        promptInput.isEnabled = enabled && assistantReady
        sendButton.isEnabled = enabled && assistantReady
        newChatButton.isEnabled = enabled && assistantReady
        speakSwitch.isEnabled = enabled && assistantReady
        playButton.isEnabled = enabled && session.lastAudioPath?.let(::File)?.isFile == true
        benchmarkButton.isEnabled = enabled && assistantReady
        voiceBackendSpinner.isEnabled = enabled
    }

    private fun showError(message: String) {
        progress.visibility = View.GONE
        statusBadge.text = getString(R.string.error_badge)
        statusText.text = message
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    override fun onDestroy() {
        monitorJob?.cancel()
        workJob?.cancel()
        player?.release()
        pcmPlayer?.close()
        kittenVoice.close()
        super.onDestroy()
    }

    companion object {
        private const val QMX_SWITCH = "GGML_KLEIDIAI_SME"
        private const val LLM_QMX_LAYERS_VARIABLE = "QMX_ACCELERATED_LAYERS"
        private const val LLM_QMX_LAYERS = 2
        private const val VOICE_QMX_PROBE_LAYERS = 4
        private const val VOICE_WARMUP_FRAMES = 24
        private const val VOICE_BENCHMARK_FRAMES = 48
        private const val DEFAULT_TTS_THREADS = 4
        private const val MAX_RESPONSE_TOKENS = 96
        private const val MAX_VOICE_FRAMES = 160
        private const val MAX_SPOKEN_CHARACTERS = 600
        private const val MAX_HISTORY_CHARACTERS = 5_500
        private const val DOWNLOAD_POLL_MS = 750L
        private const val MODEL_HANDOFF_DELAY_MS = 350L
        private const val VOICE_WARMUP_TEXT = "Voice assistant ready."
        private const val VOICE_BENCHMARK_TEXT = "This is a matched voice latency test."
        private const val VOICE_PREFERENCES = "voice-performance"
        private const val PREFERRED_THREADS = "preferred-threads"
        private const val PREFERRED_KITTEN_THREADS = "preferred-kitten-threads"
        private const val PREFERRED_VOICE_BACKEND = "preferred-voice-backend"
        private const val SYSTEM_PROMPT =
            "You are a concise, helpful voice assistant running privately on this phone. " +
                "Use short spoken-friendly answers unless the user requests detail."

        @Volatile
        private var assistantReady = false
        @Volatile
        private var lastVoiceProof: VoiceQmxStatus? = null
        @Volatile
        private var lastVoicePlan: VoiceQmxPlan? = null
        @Volatile
        private var lastLlmAcceleration = ""
        private val modelLoadMutex = Mutex()
    }
}
