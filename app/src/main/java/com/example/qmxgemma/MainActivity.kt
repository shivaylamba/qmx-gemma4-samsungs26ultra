package com.example.qmxgemma

import android.app.DownloadManager
import android.database.Cursor
import android.net.Uri
import android.os.Bundle
import android.os.StatFs
import android.provider.OpenableColumns
import android.system.Os
import android.util.Log
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.google.android.material.textfield.TextInputEditText
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
import java.io.FileOutputStream
import kotlin.math.max
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {
    private lateinit var statusText: TextView
    private lateinit var qmxBadge: TextView
    private lateinit var modelText: TextView
    private lateinit var transcriptText: TextView
    private lateinit var transcriptScroll: ScrollView
    private lateinit var promptInput: TextInputEditText
    private lateinit var modelButton: Button
    private lateinit var downloadModelButton: Button
    private lateinit var sendButton: Button

    private lateinit var engine: InferenceEngine
    private lateinit var modelDownload: HuggingFaceModelDownload
    private var generationJob: Job? = null
    private var downloadMonitorJob: Job? = null
    private val modelLoadMutex = Mutex()
    @Volatile
    private var modelReady = false
    private var hasConversation = false
    private var benchmarkStarted = false
    @Volatile
    private var runtimeReady = false

    private val chooseModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(::importAndLoadModel)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        qmxBadge = findViewById(R.id.qmxBadge)
        modelText = findViewById(R.id.modelText)
        transcriptText = findViewById(R.id.transcriptText)
        transcriptScroll = findViewById(R.id.transcriptScroll)
        promptInput = findViewById(R.id.promptInput)
        modelButton = findViewById(R.id.modelButton)
        downloadModelButton = findViewById(R.id.downloadModelButton)
        sendButton = findViewById(R.id.sendButton)
        modelDownload = HuggingFaceModelDownload(applicationContext)
        applySystemInsets(findViewById(R.id.root))
        modelButton.isEnabled = false
        downloadModelButton.isEnabled = false

        // Qualcomm's documented runtime switch. ADB may override it for controlled A/B tests.
        // KleidiAI still checks the CPU capability,
        // so unsupported devices safely use another compiled CPU backend.
        val qmxMode = intent.getStringExtra(EXTRA_QMX_MODE)?.takeIf { it == "0" || it == "1" } ?: "1"
        val qmxLayers = intent.getIntExtra(EXTRA_QMX_LAYERS, DEFAULT_QMX_LAYERS)
            .coerceIn(1, MAX_QMX_LAYERS)
        Os.setenv(QMX_ENVIRONMENT_VARIABLE, qmxMode, true)
        Os.setenv(QMX_LAYERS_ENVIRONMENT_VARIABLE, qmxLayers.toString(), true)
        Log.i(TAG, "QMX_BENCH_CONFIG sme=$qmxMode layers=$qmxLayers")

        lifecycleScope.launch(Dispatchers.Default) {
            engine = AiChat.getInferenceEngine(applicationContext)
            engine.state.first {
                it is InferenceEngine.State.Initialized || it is InferenceEngine.State.Error
            }
            if (engine.state.value is InferenceEngine.State.Initialized) {
                runtimeReady = true
                val modelsDir = File(filesDir, "models")
                val requestedModelName = intent.getStringExtra(EXTRA_MODEL_NAME)
                    ?.takeIf { it == File(it).name }
                val importedModel = if (requestedModelName != null) {
                    File(modelsDir, requestedModelName).takeIf {
                        it.isFile && it.extension.equals("gguf", ignoreCase = true)
                    }
                } else {
                    modelsDir.listFiles()
                        ?.filter { it.isFile && it.extension.equals("gguf", ignoreCase = true) }
                        ?.maxByOrNull(File::lastModified)
                }
                val downloadedModel = if (
                    importedModel == null &&
                    (requestedModelName == null || requestedModelName == HuggingFaceModelDownload.MODEL_FILENAME)
                ) {
                    verifiedDownloadedModel()
                } else {
                    null
                }
                val existingModel = importedModel ?: downloadedModel
                Log.i(TAG, "QMX_MODEL_SELECTION requested=$requestedModelName selected=${existingModel?.name}")
                if (existingModel != null) {
                    try {
                        loadModelFile(existingModel)
                    } catch (error: Exception) {
                        Log.e(TAG, "Could not reload the imported model", error)
                        withContext(Dispatchers.Main) {
                            showError(error.message ?: "Could not reload the imported model")
                        }
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Runtime ready · choose a model"
                        modelButton.isEnabled = true
                        downloadModelButton.isEnabled = true
                    }
                }
                withContext(Dispatchers.Main) {
                    refreshDownloadButton()
                    monitorModelDownload()
                }
            } else {
                withContext(Dispatchers.Main) {
                    statusText.text = "Native runtime failed to start"
                }
            }
        }

        modelButton.setOnClickListener {
            if (generationJob?.isActive == true) return@setOnClickListener
            chooseModel.launch(arrayOf("application/octet-stream", "*/*"))
        }
        downloadModelButton.setOnClickListener {
            when (modelDownload.query()) {
                is HuggingFaceModelDownload.State.Active -> cancelModelDownload()
                HuggingFaceModelDownload.State.Complete -> monitorModelDownload()
                is HuggingFaceModelDownload.State.Failed -> {
                    modelDownload.clearTracking()
                    showModelDownloadConfirmation()
                }
                HuggingFaceModelDownload.State.Idle -> {
                    if (modelDownload.isVerified()) {
                        loadVerifiedDownloadedModel()
                    } else {
                        showModelDownloadConfirmation()
                    }
                }
            }
        }
        sendButton.setOnClickListener {
            if (generationJob?.isActive == true) generationJob?.cancel() else sendPrompt()
        }
        promptInput.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                sendPrompt()
                true
            } else {
                false
            }
        }
        findViewById<Button>(R.id.clearButton).setOnClickListener {
            if (!modelReady || generationJob?.isActive == true) return@setOnClickListener
            setControlsEnabled(false)
            statusText.text = "Clearing conversation…"
            generationJob = lifecycleScope.launch {
                runCatching { engine.resetConversation() }
                    .onSuccess {
                        transcriptText.text = "Conversation cleared. Ask a new question."
                        hasConversation = false
                        showAccelerationStatus()
                    }
                    .onFailure { error ->
                        showError(error.message ?: "Could not clear the conversation")
                    }
                setControlsEnabled(true)
            }
        }
    }

    private fun importAndLoadModel(uri: Uri) {
        modelReady = false
        setControlsEnabled(false)
        val displayName = queryDisplayName(uri)
        if (!displayName.endsWith(".gguf", ignoreCase = true)) {
            showError("Please choose a GGUF model file.")
            return
        }
        modelText.text = displayName

        lifecycleScope.launch(Dispatchers.IO) {
            runCatching {
                engine.state.first {
                    it is InferenceEngine.State.Initialized || it is InferenceEngine.State.Error
                }
                check(engine.state.value is InferenceEngine.State.Initialized) {
                    "Native runtime is unavailable"
                }

                val modelFile = copyModelToPrivateStorage(uri, displayName)
                loadModelFile(modelFile)
            }.onFailure { error ->
                Log.e(TAG, "Model import/load failed", error)
                withContext(Dispatchers.Main) {
                    showError(error.message ?: "Could not load this model")
                }
            }
        }
    }

    private fun showModelDownloadConfirmation() {
        AlertDialog.Builder(this)
            .setTitle("Download Gemma 4 E2B Q8_0?")
            .setMessage(
                "This downloads ${formatBytes(HuggingFaceModelDownload.MODEL_SIZE_BYTES)} " +
                    "from the ggml-org repository on Hugging Face. Wi-Fi is recommended. " +
                    "The model stays in this app's storage and is removed if the app is uninstalled.",
            )
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton("Download") { _, _ -> startModelDownload() }
            .show()
    }

    private fun startModelDownload() {
        runCatching { modelDownload.enqueue() }
            .onSuccess {
                modelText.text = HuggingFaceModelDownload.MODEL_FILENAME
                transcriptText.text =
                    "Downloading Gemma from Hugging Face. You can leave the app; Android will continue the transfer."
                refreshDownloadButton()
                monitorModelDownload()
            }
            .onFailure { error -> showDownloadError(error.message ?: "Could not start the model download") }
    }

    private fun cancelModelDownload() {
        modelDownload.cancel()
        downloadMonitorJob?.cancel()
        statusText.text = if (modelReady) engine.accelerationInfo() else "Model download cancelled"
        if (!modelReady) modelText.text = "No model selected"
        refreshDownloadButton()
    }

    private fun monitorModelDownload() {
        downloadMonitorJob?.cancel()
        refreshDownloadButton()
        if (!modelDownload.hasTrackedDownload) return

        downloadMonitorJob = lifecycleScope.launch {
            while (isActive && modelDownload.hasTrackedDownload) {
                when (val state = withContext(Dispatchers.IO) { modelDownload.query() }) {
                    is HuggingFaceModelDownload.State.Active -> showDownloadProgress(state)
                    HuggingFaceModelDownload.State.Complete -> {
                        finishModelDownload()
                        break
                    }
                    is HuggingFaceModelDownload.State.Failed -> {
                        withContext(Dispatchers.IO) { modelDownload.discardInvalidFile() }
                        showDownloadError("Model download failed (${downloadFailureMessage(state.reason)}).")
                        refreshDownloadButton()
                        break
                    }
                    HuggingFaceModelDownload.State.Idle -> {
                        modelDownload.clearTracking()
                        refreshDownloadButton()
                        break
                    }
                }
                delay(DOWNLOAD_POLL_INTERVAL_MS)
            }
        }
    }

    private fun showDownloadProgress(state: HuggingFaceModelDownload.State.Active) {
        val total = state.totalBytes.takeIf { it > 0 } ?: HuggingFaceModelDownload.MODEL_SIZE_BYTES
        val percent = ((state.downloadedBytes * 100) / total).coerceIn(0, 100)
        val phase = when (state.status) {
            DownloadManager.STATUS_PAUSED -> "Download paused"
            DownloadManager.STATUS_PENDING -> "Download queued"
            else -> "Downloading model"
        }
        statusText.text = "$phase · $percent% · ${formatBytes(state.downloadedBytes)} / ${formatBytes(total)}"
        modelText.text = HuggingFaceModelDownload.MODEL_FILENAME
    }

    private suspend fun finishModelDownload() {
        updateStatus("Verifying downloaded model…")
        val verified = withContext(Dispatchers.IO) { modelDownload.verifyAndRemember() }
        if (!verified) {
            withContext(Dispatchers.IO) { modelDownload.discardInvalidFile() }
            showDownloadError("The downloaded model failed its size or SHA-256 check and was removed.")
            refreshDownloadButton()
            return
        }

        modelDownload.clearTracking()
        refreshDownloadButton()
        if (runtimeReady && !modelReady) {
            withContext(Dispatchers.IO) { loadModelFile(modelDownload.destinationFile) }
        } else {
            statusText.text = if (modelReady) engine.accelerationInfo() else "Model downloaded and verified"
            Toast.makeText(this, "Gemma model downloaded and verified", Toast.LENGTH_LONG).show()
        }
    }

    private suspend fun verifiedDownloadedModel(): File? {
        val file = modelDownload.destinationFile
        if (!file.isFile || file.length() != HuggingFaceModelDownload.MODEL_SIZE_BYTES) return null
        val state = withContext(Dispatchers.IO) { modelDownload.query() }
        if (state is HuggingFaceModelDownload.State.Active || state is HuggingFaceModelDownload.State.Failed) {
            return null
        }
        if (!modelDownload.isVerified()) {
            updateStatus("Verifying downloaded model…")
            if (!withContext(Dispatchers.IO) { modelDownload.verifyAndRemember() }) {
                withContext(Dispatchers.IO) { modelDownload.discardInvalidFile() }
                return null
            }
        }
        modelDownload.clearTracking()
        return file
    }

    private fun loadVerifiedDownloadedModel() {
        if (!runtimeReady || modelReady) return
        lifecycleScope.launch(Dispatchers.IO) {
            runCatching {
                verifiedDownloadedModel()?.let { loadModelFile(it) }
                    ?: error("The downloaded model is incomplete or invalid")
            }.onFailure { error ->
                Log.e(TAG, "Downloaded model load failed", error)
                withContext(Dispatchers.Main) {
                    showError(error.message ?: "Could not load the downloaded model")
                }
            }
        }
    }

    private fun refreshDownloadButton() {
        downloadModelButton.text = when {
            modelDownload.hasTrackedDownload -> getString(R.string.cancel_download)
            modelDownload.isVerified() -> getString(R.string.model_downloaded)
            else -> getString(R.string.download_model)
        }
        downloadModelButton.isEnabled = runtimeReady || modelDownload.hasTrackedDownload
    }

    private fun downloadFailureMessage(reason: Int): String = when (reason) {
        DownloadManager.ERROR_INSUFFICIENT_SPACE -> "not enough storage"
        DownloadManager.ERROR_CANNOT_RESUME -> "the transfer could not resume"
        DownloadManager.ERROR_DEVICE_NOT_FOUND -> "storage is unavailable"
        DownloadManager.ERROR_FILE_ERROR -> "file storage error"
        DownloadManager.ERROR_HTTP_DATA_ERROR -> "Hugging Face network error"
        DownloadManager.ERROR_TOO_MANY_REDIRECTS -> "too many redirects"
        else -> "reason $reason"
    }

    private fun showDownloadError(message: String) {
        statusText.text = if (modelReady) engine.accelerationInfo() else "Model download failed"
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        refreshDownloadButton()
    }

    private suspend fun loadModelFile(modelFile: File) = modelLoadMutex.withLock {
        if (!modelReady) {
            updateStatus("Loading ${modelFile.nameWithoutExtension} into memory…")
            engine.loadModel(modelFile.absolutePath)
            engine.setSystemPrompt(
                "You are Gemma, a concise and helpful assistant running privately on this phone."
            )
            modelReady = true
        }
        withContext(Dispatchers.Main) {
            showAccelerationStatus()
            modelText.text = "${modelFile.name} · ${formatBytes(modelFile.length())}"
            transcriptText.text = "Gemma is ready. Ask anything — inference is fully offline."
            hasConversation = false
            setControlsEnabled(true)
        }
        runRequestedBenchmark()
    }

    private suspend fun runRequestedBenchmark() {
        if (!intent.getBooleanExtra(EXTRA_RUN_BENCHMARK, false) || benchmarkStarted) return
        benchmarkStarted = true
        val threads = intent.getIntExtra(EXTRA_BENCH_THREADS, 4).coerceIn(1, 4)
        val runs = intent.getIntExtra(EXTRA_BENCH_RUNS, 5).coerceIn(1, 20)
        val sme = System.getenv(QMX_ENVIRONMENT_VARIABLE) ?: "unset"
        updateStatus("Benchmarking SME=$sme with $threads thread(s)…")
        val result = engine.bench(
            pp = BENCH_PROMPT_TOKENS,
            tg = BENCH_DECODE_TOKENS,
            pl = 1,
            nr = runs,
            threads = threads,
        )
        Log.i(TAG, "QMX_BENCH_RESULT sme=$sme threads=$threads runs=$runs\n$result")
        withContext(Dispatchers.Main) {
            transcriptText.text = result
            statusText.text = "Benchmark complete · SME=$sme · $threads thread(s)"
        }
    }

    private suspend fun copyModelToPrivateStorage(uri: Uri, displayName: String): File {
        val safeName = displayName.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val modelsDir = File(filesDir, "models").apply { mkdirs() }
        val destination = File(modelsDir, safeName)
        val expectedSize = contentResolver.openFileDescriptor(uri, "r")?.use { it.statSize } ?: -1L

        if (destination.isFile && expectedSize > 0 && destination.length() == expectedSize) {
            updateStatus("Using the imported model…")
            return destination
        }
        if (expectedSize > 0 && StatFs(filesDir.absolutePath).availableBytes < expectedSize + FREE_SPACE_HEADROOM) {
            error("Not enough free space. The model needs ${formatBytes(expectedSize)} plus working space.")
        }

        updateStatus("Importing ${formatBytes(expectedSize)} model…")
        val temporary = File(modelsDir, "$safeName.partial")
        contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Cannot open the selected file" }
            FileOutputStream(temporary).use { output ->
                val buffer = ByteArray(COPY_BUFFER_BYTES)
                var copied = 0L
                var lastPercent = -1
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    output.write(buffer, 0, count)
                    copied += count
                    if (expectedSize > 0) {
                        val percent = ((copied * 100) / expectedSize).toInt()
                        if (percent != lastPercent) {
                            lastPercent = percent
                            updateStatus("Importing model… $percent%")
                        }
                    }
                }
            }
        }
        if (destination.exists()) destination.delete()
        check(temporary.renameTo(destination)) { "Could not finalize the imported model" }
        return destination
    }

    private fun sendPrompt() {
        if (!modelReady || generationJob?.isActive == true) return
        val prompt = promptInput.text?.toString()?.trim().orEmpty()
        if (prompt.isEmpty()) return

        promptInput.text?.clear()
        if (!hasConversation) transcriptText.text = ""
        hasConversation = true
        transcriptText.append("\n\nYou\n$prompt\n\nGemma\n")
        scrollTranscript()
        setControlsEnabled(false)
        sendButton.isEnabled = true
        sendButton.text = getString(R.string.stop)
        statusText.text = "Generating on device…"

        generationJob = lifecycleScope.launch {
            runCatching {
                engine.sendUserPrompt(prompt, predictLength = 256).collect { token ->
                    transcriptText.append(token)
                    scrollTranscript()
                }
            }.onFailure { error ->
                if (error !is kotlinx.coroutines.CancellationException) {
                    transcriptText.append("\n\n[Error: ${error.message}]")
                    Log.e(TAG, "Generation failed", error)
                }
            }
            showAccelerationStatus()
            sendButton.text = getString(R.string.send)
            setControlsEnabled(true)
        }
    }

    private fun setControlsEnabled(enabled: Boolean) {
        promptInput.isEnabled = enabled && modelReady
        sendButton.isEnabled = enabled && modelReady
        modelButton.isEnabled = enabled
        downloadModelButton.isEnabled = enabled || modelDownload.hasTrackedDownload
    }

    private suspend fun updateStatus(message: String) = withContext(Dispatchers.Main) {
        statusText.text = message
    }

    private fun queryDisplayName(uri: Uri): String {
        var name = "gemma-4-E2B-it-Q8_0.gguf"
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor: Cursor ->
            if (cursor.moveToFirst()) {
                val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (index >= 0) name = cursor.getString(index)
            }
        }
        return name
    }

    private fun scrollTranscript() = transcriptScroll.post {
        transcriptScroll.fullScroll(ScrollView.FOCUS_DOWN)
    }

    private fun applySystemInsets(root: View) {
        val horizontal = (20 * resources.displayMetrics.density).roundToInt()
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, windowInsets ->
            val bars = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars())
            val ime = windowInsets.getInsets(WindowInsetsCompat.Type.ime())
            view.setPadding(
                horizontal + bars.left,
                bars.top,
                horizontal + bars.right,
                max(bars.bottom, ime.bottom),
            )
            if (ime.bottom > 0) scrollTranscript()
            windowInsets
        }
        ViewCompat.requestApplyInsets(root)
    }

    private fun showAccelerationStatus() {
        val acceleration = engine.accelerationInfo()
        qmxBadge.text = if (acceleration.startsWith("QMX active")) "QMX ACTIVE" else "CPU FALLBACK"
        statusText.text = acceleration
    }

    private fun showError(message: String) {
        statusText.text = "Model not loaded"
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        modelButton.isEnabled = true
        downloadModelButton.isEnabled = true
    }

    override fun onDestroy() {
        generationJob?.cancel()
        downloadMonitorJob?.cancel()
        // The engine is process-scoped. Android owns its native memory when the process exits;
        // destroying it for an Activity recreation would invalidate the singleton for the new UI.
        super.onDestroy()
    }

    companion object {
        private const val TAG = "QmxGemma"
        private const val QMX_ENVIRONMENT_VARIABLE = "GGML_KLEIDIAI_SME"
        private const val QMX_LAYERS_ENVIRONMENT_VARIABLE = "QMX_ACCELERATED_LAYERS"
        private const val EXTRA_QMX_MODE = "qmx_mode"
        private const val EXTRA_QMX_LAYERS = "qmx_layers"
        private const val EXTRA_RUN_BENCHMARK = "run_benchmark"
        private const val EXTRA_MODEL_NAME = "model_name"
        private const val EXTRA_BENCH_THREADS = "bench_threads"
        private const val EXTRA_BENCH_RUNS = "bench_runs"
        private const val BENCH_PROMPT_TOKENS = 128
        private const val BENCH_DECODE_TOKENS = 128
        private const val DEFAULT_QMX_LAYERS = 6
        private const val MAX_QMX_LAYERS = 64
        private const val COPY_BUFFER_BYTES = 4 * 1024 * 1024
        private const val FREE_SPACE_HEADROOM = 512L * 1024 * 1024
        private const val DOWNLOAD_POLL_INTERVAL_MS = 1_000L

        private fun formatBytes(bytes: Long): String = when {
            bytes < 0 -> "unknown size"
            bytes >= 1_000_000_000 -> "%.2f GB".format(bytes / 1_000_000_000.0)
            bytes >= 1_000_000 -> "%.1f MB".format(bytes / 1_000_000.0)
            else -> "$bytes B"
        }
    }
}
