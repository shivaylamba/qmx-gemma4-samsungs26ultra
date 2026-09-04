package com.example.qmxvoice

import android.media.MediaPlayer
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.ProgressBar
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.arm.aichat.VoiceInferenceEngine
import com.arm.aichat.VoiceLatency
import com.arm.aichat.VoiceQmxStatus
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var engine: VoiceInferenceEngine
    private lateinit var models: VoiceModelDownload
    private lateinit var qmxBadge: TextView
    private lateinit var statusText: TextView
    private lateinit var modelText: TextView
    private lateinit var resultText: TextView
    private lateinit var proofText: TextView
    private lateinit var progress: ProgressBar
    private lateinit var downloadButton: Button
    private lateinit var synthesizeButton: Button
    private lateinit var compareButton: Button
    private lateinit var playButton: Button
    private lateinit var inputText: TextInputEditText
    private lateinit var threadGroup: RadioGroup

    private var monitorJob: Job? = null
    private var workJob: Job? = null
    private var player: MediaPlayer? = null
    private var modelLoaded = false
    private var lastAudio: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        qmxBadge = findViewById(R.id.qmxBadge)
        statusText = findViewById(R.id.statusText)
        modelText = findViewById(R.id.modelText)
        resultText = findViewById(R.id.resultText)
        proofText = findViewById(R.id.proofText)
        progress = findViewById(R.id.progress)
        downloadButton = findViewById(R.id.downloadButton)
        synthesizeButton = findViewById(R.id.synthesizeButton)
        compareButton = findViewById(R.id.compareButton)
        playButton = findViewById(R.id.playButton)
        inputText = findViewById(R.id.inputText)
        threadGroup = findViewById(R.id.threadGroup)

        engine = VoiceInferenceEngine.getInstance(applicationContext)
        models = VoiceModelDownload(applicationContext)

        downloadButton.setOnClickListener { startDownload() }
        synthesizeButton.setOnClickListener { runSingle() }
        compareButton.setOnClickListener { runComparison() }
        playButton.setOnClickListener { playLastAudio() }

        lifecycleScope.launch {
            runCatching { engine.initialize() }
                .onFailure { showError(getString(R.string.runtime_initialization_failed), it) }
                .onSuccess {
                    if (engine.isLoaded) restoreLoadedModel() else resumeModelSetup()
                }
        }
    }

    private suspend fun resumeModelSetup() {
        val state = models.query()
        when (voiceStartupAction(state, models.isVerified(), models.hasCompleteFiles())) {
            VoiceStartupAction.MONITOR_DOWNLOAD -> {
                val active = state as VoiceModelDownload.State.Active
                showActiveDownload(active)
                monitorDownloads()
            }
            VoiceStartupAction.LOAD_MODEL -> loadModels()
            VoiceStartupAction.VERIFY_FILES -> {
                setBusy(true, getString(R.string.verifying_existing_models))
                val valid = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                    models.verifyAndRemember()
                }
                if (valid) loadModels() else showDownloadReady()
            }
            VoiceStartupAction.SHOW_DOWNLOAD -> showDownloadReady()
            VoiceStartupAction.SHOW_FAILURE -> {
                showDownloadReady()
                showError(
                    getString(R.string.model_download_failed),
                    IllegalStateException((state as VoiceModelDownload.State.Failed).reason),
                )
            }
        }
    }

    private fun showActiveDownload(state: VoiceModelDownload.State.Active) {
        val total = state.totalBytes.takeIf { it > 0 } ?: models.totalBytes
        val fraction = (state.downloadedBytes.toDouble() / total).coerceIn(0.0, 1.0)
        downloadButton.isEnabled = false
        progress.visibility = View.VISIBLE
        progress.isIndeterminate = false
        progress.max = 1000
        progress.progress = (fraction * progress.max).toInt()
        statusText.text = getString(R.string.downloading_percent, (fraction * 100).toInt())
        modelText.text = getString(
            R.string.download_progress,
            VoiceModelDownload.formatBytes(state.downloadedBytes),
            VoiceModelDownload.formatBytes(total),
        )
    }

    private fun showDownloadReady() {
        progress.visibility = View.GONE
        statusText.text = getString(R.string.voice_runtime_ready)
        modelText.text = getString(
            R.string.model_files_description,
            VoiceModelDownload.formatBytes(models.totalBytes),
        )
        downloadButton.isEnabled = true
        downloadButton.text = getString(R.string.download_models)
    }

    private fun startDownload() {
        runCatching { models.enqueue() }
            .onFailure { showError(getString(R.string.could_not_start_download), it) }
            .onSuccess {
                downloadButton.isEnabled = false
                progress.visibility = View.VISIBLE
                progress.isIndeterminate = false
                progress.max = 1000
                statusText.text = getString(R.string.downloading_from_hugging_face)
                monitorDownloads()
            }
    }

    private fun monitorDownloads() {
        monitorJob?.cancel()
        monitorJob = lifecycleScope.launch {
            while (isActive) {
                when (val state = models.query()) {
                    VoiceModelDownload.State.Idle -> {
                        showDownloadReady()
                        return@launch
                    }
                    is VoiceModelDownload.State.Active -> {
                        showActiveDownload(state)
                    }
                    VoiceModelDownload.State.Complete -> {
                        val valid = if (models.isVerified()) true else {
                            statusText.text = getString(R.string.verifying_checksums)
                            progress.isIndeterminate = true
                            kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                                models.verifyAndRemember()
                            }
                        }
                        if (!valid) {
                            models.cancel()
                            showDownloadReady()
                            showError(
                                getString(R.string.model_verification_failed),
                                IllegalStateException(getString(R.string.checksum_mismatch)),
                            )
                            return@launch
                        }
                        loadModels()
                        return@launch
                    }
                    is VoiceModelDownload.State.Failed -> {
                        showDownloadReady()
                        showError(
                            getString(R.string.model_download_failed),
                            IllegalStateException(state.reason),
                        )
                        return@launch
                    }
                }
                delay(750)
            }
        }.also { job ->
            job.invokeOnCompletion { error ->
                if (error != null && error !is kotlinx.coroutines.CancellationException) {
                    runOnUiThread { showError(getString(R.string.model_download_failed), error) }
                }
            }
        }
    }

    private suspend fun loadModels() {
        setBusy(true, getString(R.string.loading_qmx_model))
        runCatching { engine.load(models.backboneFile, models.mmprojFile, 1) }
            .onFailure {
                setBusy(false)
                showError(getString(R.string.model_load_failed), it)
            }
            .onSuccess {
                showLoadedModel()
            }
    }

    private fun restoreLoadedModel() {
        showLoadedModel()
        statusText.text = getString(R.string.model_restored)
    }

    private fun showLoadedModel() {
        modelLoaded = true
        setBusy(false)
        val qmx = engine.qmxStatus()
        statusText.text = when {
            qmx.inferenceConfirmed -> getString(R.string.qmx_already_proven)
            qmx.selected -> getString(R.string.qmx_selected)
            else -> getString(R.string.qmx_not_observed)
        }
        modelText.text = getString(R.string.loaded_model_name)
        updateProof(qmx)
        qmxBadge.text = when {
            qmx.inferenceConfirmed -> getString(R.string.qmx_proven_badge)
            qmx.selected -> getString(R.string.qmx_ready_badge)
            else -> getString(R.string.fallback_badge)
        }
        qmxBadge.setBackgroundResource(
            if (qmx.selected) R.drawable.badge_ready else R.drawable.badge_waiting,
        )
    }

    private fun runSingle() {
        val text = inputText.text?.toString().orEmpty().trim()
        if (text.isBlank()) return
        val threads = if (threadGroup.checkedRadioButtonId == R.id.fourThreads) 4 else 1
        val output = File(cacheDir, "qmx-voice-${threads}t.wav")
        workJob = lifecycleScope.launch {
            setBusy(true, getString(R.string.synthesizing_threads, threads))
            runCatching { engine.synthesize(text, output, threads, maxFrames = 120) }
                .onFailure {
                    setBusy(false)
                    showError(getString(R.string.voice_synthesis_failed), it)
                }
                .onSuccess { latency ->
                    lastAudio = output
                    setBusy(false)
                    resultText.text = formatLatency(latency)
                    updateProof(latency.qmx)
                    updateQmxBadge(latency.qmx)
                    playButton.isEnabled = true
                    statusText.text = getString(R.string.synthesis_complete)
                }
        }
    }

    private fun runComparison() {
        val text = inputText.text?.toString().orEmpty().trim()
        if (text.isBlank()) return
        workJob = lifecycleScope.launch {
            setBusy(true, getString(R.string.running_one_thread))
            val one = runCatching {
                engine.synthesize(text, File(cacheDir, "qmx-voice-1t.wav"), 1, maxFrames = 120)
            }.getOrElse {
                setBusy(false)
                showError(getString(R.string.one_thread_failed), it)
                return@launch
            }
            statusText.text = getString(R.string.running_four_threads)
            val fourOutput = File(cacheDir, "qmx-voice-4t.wav")
            val four = runCatching {
                engine.synthesize(text, fourOutput, 4, maxFrames = 120)
            }.getOrElse {
                setBusy(false)
                showError(getString(R.string.four_threads_failed), it)
                return@launch
            }
            lastAudio = fourOutput
            setBusy(false)
            resultText.text = formatComparison(one, four)
            updateProof(four.qmx)
            updateQmxBadge(four.qmx)
            playButton.isEnabled = true
            statusText.text = getString(R.string.comparison_complete)
        }
    }

    private fun formatLatency(value: VoiceLatency): String = buildString {
        appendLine("${value.threads} thread result")
        appendLine("Prompt prefill        ${ms(value.promptMs)}")
        appendLine("First audio frame*    ${ms(value.firstFrameMs)}")
        appendLine("Full WAV ready        ${ms(value.totalMs)}")
        appendLine("Audio duration        ${ms(value.audioMs)}")
        appendLine("Real-time factor      ${"%.3f".format(value.realTimeFactor)}")
        append("${value.frames} frames at ${value.sampleRate} Hz")
    }

    private fun formatComparison(one: VoiceLatency, four: VoiceLatency): String = buildString {
        appendLine("Matched synthesis comparison")
        appendLine("                         1 thread       4 threads")
        appendLine("Prompt prefill          ${pad(ms(one.promptMs))}${ms(four.promptMs)}")
        appendLine("First audio frame*      ${pad(ms(one.firstFrameMs))}${ms(four.firstFrameMs)}")
        appendLine("Full WAV ready          ${pad(ms(one.totalMs))}${ms(four.totalMs)}")
        appendLine("Real-time factor        ${pad("%.3f".format(one.realTimeFactor))}${"%.3f".format(four.realTimeFactor)}")
        append("Total speedup, 4T vs 1T: ${"%.3f".format(one.totalMs / four.totalMs)}x")
    }

    private fun updateProof(qmx: VoiceQmxStatus) {
        proofText.text = buildString {
            appendLine(getString(R.string.proof_selection, yesNo(qmx.selected)))
            appendLine(getString(R.string.proof_gemm, yesNo(qmx.gemmExecuted)))
            appendLine(getString(R.string.proof_gemv, yesNo(qmx.gemvExecuted)))
            append(getString(R.string.proof_buffer, qmx.bufferMiB, qmx.layers))
        }
    }

    private fun updateQmxBadge(qmx: VoiceQmxStatus) {
        qmxBadge.text = getString(
            if (qmx.inferenceConfirmed) R.string.qmx_proven_badge else R.string.partial_badge,
        )
        qmxBadge.setBackgroundResource(if (qmx.inferenceConfirmed) R.drawable.badge_ready else R.drawable.badge_waiting)
    }

    private fun setBusy(busy: Boolean, message: String? = null) {
        progress.visibility = if (busy) View.VISIBLE else View.GONE
        progress.isIndeterminate = true
        downloadButton.isEnabled = !busy && !models.isVerified()
        synthesizeButton.isEnabled = !busy && modelLoaded
        compareButton.isEnabled = !busy && modelLoaded
        inputText.isEnabled = !busy
        if (message != null) statusText.text = message
    }

    private fun playLastAudio() {
        val audio = lastAudio?.takeIf(File::isFile) ?: return
        player?.release()
        player = MediaPlayer().apply {
            setDataSource(audio.absolutePath)
            setOnCompletionListener { completed -> completed.release(); player = null }
            prepare()
            start()
        }
    }

    private fun showError(prefix: String, error: Throwable) {
        statusText.text = getString(
            R.string.error_with_detail,
            prefix,
            error.message ?: error.javaClass.simpleName,
        )
        qmxBadge.text = getString(R.string.error_badge)
        qmxBadge.setBackgroundResource(R.drawable.badge_waiting)
        Toast.makeText(this, statusText.text, Toast.LENGTH_LONG).show()
    }

    override fun onDestroy() {
        monitorJob?.cancel()
        workJob?.cancel()
        player?.release()
        super.onDestroy()
    }

    private fun ms(value: Double) = if (value >= 1000) "%.2f s".format(value / 1000.0) else "%.0f ms".format(value)
    private fun pad(value: String) = value.padEnd(15)
    private fun yesNo(value: Boolean) = getString(
        if (value) R.string.confirmed else R.string.not_observed,
    )
}
