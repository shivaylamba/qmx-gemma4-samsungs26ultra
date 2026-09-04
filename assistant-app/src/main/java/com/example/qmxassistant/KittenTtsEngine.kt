package com.example.qmxassistant

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.SystemClock
import android.util.Log
import java.io.BufferedOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.zip.ZipFile
import kotlin.math.roundToInt

internal data class KittenLatency(
    val firstPcmMs: Double,
    val inferenceMs: Double,
    val wavMs: Double,
    val totalMs: Double,
    val audioMs: Double,
    val realTimeFactor: Double,
    val threads: Int,
    val runtime: String,
)

/**
 * KittenTTS Nano 0.8 inference using ONNX Runtime's CPU execution provider.
 *
 * No NNAPI, QNN, GPU, or XNNPACK provider is registered on this session. The
 * explicit addCPU call is intentional: this backend measures the Snapdragon
 * CPU through ONNX Runtime and must not be confused with the llama.cpp QMX
 * path used by Qwen3-TTS.
 */
internal class KittenTtsEngine {
    private val environment = OrtEnvironment.getEnvironment()
    private val phonemizer = KittenPhonemizer()
    private var session: OrtSession? = null
    private var voices: Map<String, VoiceEmbedding> = emptyMap()
    private var loadedThreads = 0

    val isLoaded: Boolean
        @Synchronized get() = session != null

    @Synchronized
    fun load(
        model: File,
        voicesFile: File,
        rulesFile: File,
        listFile: File,
        threads: Int,
    ) {
        require(model.isFile) { "KittenTTS ONNX model not found" }
        require(voicesFile.isFile) { "KittenTTS voice embeddings not found" }
        require(rulesFile.isFile && listFile.isFile) { "KittenTTS phonemizer data not found" }
        require(threads > 0) { "ONNX CPU thread count must be positive" }
        if (session != null && loadedThreads == threads) return

        closeSession()
        val loadedVoices = loadVoiceEmbeddings(voicesFile)
        check(loadedVoices.containsKey(DEFAULT_VOICE)) { "Jasper voice embedding is missing" }
        phonemizer.load(rulesFile, listFile)

        val options = OrtSession.SessionOptions()
        try {
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            options.setIntraOpNumThreads(threads)
            options.setInterOpNumThreads(1)
            options.setMemoryPatternOptimization(true)
            options.setCPUArenaAllocator(true)
            options.setLoggerId("KittenTTS-CPU")
            options.addCPU(true)
            val created = environment.createSession(model.absolutePath, options)
            check(created.inputNames == EXPECTED_INPUTS) {
                "Unexpected KittenTTS inputs: ${created.inputNames}"
            }
            check("waveform" in created.outputNames) {
                "KittenTTS waveform output is missing: ${created.outputNames}"
            }
            session = created
            voices = loadedVoices
            loadedThreads = threads
            Log.i(
                TAG,
                "Loaded KittenTTS with explicit CPUExecutionProvider; " +
                    "ORT=${environment.version}; threads=$threads; " +
                    "inputs=${created.inputNames}; outputs=${created.outputNames}",
            )
        } catch (error: Throwable) {
            phonemizer.close()
            throw error
        } finally {
            options.close()
        }
    }

    @Synchronized
    fun synthesize(
        text: String,
        output: File,
        voice: String = DEFAULT_VOICE,
        speed: Float = DEFAULT_SPEED,
    ): KittenLatency {
        val activeSession = checkNotNull(session) { "Load KittenTTS before synthesizing" }
        val embedding = checkNotNull(voices[voice]) { "Voice embedding $voice is unavailable" }
        require(text.isNotBlank()) { "Enter text to speak" }

        val started = SystemClock.elapsedRealtimeNanos()
        val phonemes = phonemizer.phonemize(text.trim())
        check(phonemes.isNotBlank()) { "The phonemizer produced no speech tokens" }
        val chunks = splitIntoChunks(encodePhonemes(phonemes))
        val audioChunks = ArrayList<FloatArray>(chunks.size)
        for (chunk in chunks) {
            val bodyLength = (chunk.size - 3).coerceAtLeast(0)
            audioChunks += runChunk(activeSession, chunk, embedding, bodyLength, speed)
        }
        val inferenceFinished = SystemClock.elapsedRealtimeNanos()
        val samples = concatenate(audioChunks)
        check(samples.isNotEmpty()) { "KittenTTS produced an empty waveform" }
        output.parentFile?.mkdirs()
        writePcm16Wav(output, samples, SAMPLE_RATE)
        val finished = SystemClock.elapsedRealtimeNanos()

        val inferenceMs = nanosToMs(inferenceFinished - started)
        val wavMs = nanosToMs(finished - inferenceFinished)
        val totalMs = nanosToMs(finished - started)
        val audioMs = samples.size * 1000.0 / SAMPLE_RATE
        return KittenLatency(
            // Kitten's ONNX graph returns a complete waveform rather than
            // streaming frames, so PCM becomes playable after the WAV is final.
            firstPcmMs = totalMs,
            inferenceMs = inferenceMs,
            wavMs = wavMs,
            totalMs = totalMs,
            audioMs = audioMs,
            realTimeFactor = totalMs / audioMs,
            threads = loadedThreads,
            runtime = runtimeInfo(),
        )
    }

    @Synchronized
    fun runtimeInfo(): String =
        "ONNX Runtime ${environment.version} · CPUExecutionProvider (explicit) · ${loadedThreads}T"

    @Synchronized
    fun close() {
        closeSession()
        phonemizer.close()
    }

    private fun closeSession() {
        session?.close()
        session = null
        voices = emptyMap()
        loadedThreads = 0
    }

    private fun runChunk(
        activeSession: OrtSession,
        tokens: LongArray,
        embedding: VoiceEmbedding,
        bodyLength: Int,
        speed: Float,
    ): FloatArray {
        val row = bodyLength.coerceIn(0, embedding.rows - 1)
        val style = embedding.data.copyOfRange(row * embedding.cols, (row + 1) * embedding.cols)
        OnnxTensor.createTensor(
            environment,
            LongBuffer.wrap(tokens),
            longArrayOf(1, tokens.size.toLong()),
        ).use { inputIds ->
            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(style),
                longArrayOf(1, style.size.toLong()),
            ).use { styleTensor ->
                OnnxTensor.createTensor(
                    environment,
                    FloatBuffer.wrap(floatArrayOf(speed)),
                    longArrayOf(1),
                ).use { speedTensor ->
                    activeSession.run(
                        mapOf(
                            "input_ids" to inputIds,
                            "style" to styleTensor,
                            "speed" to speedTensor,
                        ),
                    ).use { result ->
                        val waveform = (result.get("waveform").orElse(result[0]) as OnnxTensor)
                            .floatBuffer
                            .duplicate()
                        waveform.rewind()
                        val samples = FloatArray(waveform.remaining())
                        waveform.get(samples)
                        return trimTrailingSilence(samples)
                    }
                }
            }
        }
    }

    private fun trimTrailingSilence(samples: FloatArray): FloatArray {
        val maxTrim = minOf(samples.size, SAMPLE_RATE * MAX_SILENCE_TRIM_MS / 1000)
        var trim = 0
        while (trim < maxTrim && samples[samples.lastIndex - trim].let { kotlin.math.abs(it) } <= SILENCE_THRESHOLD) {
            trim++
        }
        return if (trim == 0 || trim >= samples.size) samples else samples.copyOf(samples.size - trim)
    }

    private fun concatenate(chunks: List<FloatArray>): FloatArray {
        val result = FloatArray(chunks.sumOf(FloatArray::size))
        var offset = 0
        chunks.forEach { chunk ->
            chunk.copyInto(result, offset)
            offset += chunk.size
        }
        return result
    }

    private fun splitIntoChunks(tokens: LongArray): List<LongArray> {
        val body = tokens.copyOfRange(1, tokens.size - 2)
        val maxBody = MAX_TOKENS_PER_CHUNK - 3
        if (body.size <= maxBody) return listOf(tokens)
        return body.asList().chunked(maxBody).map { chunk ->
            longArrayOf(START_TOKEN) + chunk.toLongArray() + longArrayOf(END_TOKEN, PAD_TOKEN)
        }
    }

    private fun encodePhonemes(phonemes: String): LongArray {
        val tokens = ArrayList<Long>(phonemes.length + 3)
        tokens += START_TOKEN
        phonemes.codePoints().forEach { codePoint -> SYMBOL_INDEX[codePoint]?.let(tokens::add) }
        tokens += END_TOKEN
        tokens += PAD_TOKEN
        return tokens.toLongArray()
    }

    private fun writePcm16Wav(output: File, samples: FloatArray, sampleRate: Int) {
        val dataBytes = samples.size * 2
        BufferedOutputStream(output.outputStream()).use { stream ->
            fun ascii(value: String) = stream.write(value.toByteArray(Charsets.US_ASCII))
            fun little16(value: Int) {
                stream.write(value and 0xff)
                stream.write((value ushr 8) and 0xff)
            }
            fun little32(value: Int) {
                stream.write(value and 0xff)
                stream.write((value ushr 8) and 0xff)
                stream.write((value ushr 16) and 0xff)
                stream.write((value ushr 24) and 0xff)
            }

            ascii("RIFF")
            little32(36 + dataBytes)
            ascii("WAVEfmt ")
            little32(16)
            little16(1)
            little16(1)
            little32(sampleRate)
            little32(sampleRate * 2)
            little16(2)
            little16(16)
            ascii("data")
            little32(dataBytes)
            samples.forEach { sample ->
                val pcm = (sample.coerceIn(-1f, 1f) * Short.MAX_VALUE).roundToInt()
                little16(pcm)
            }
        }
    }

    private data class VoiceEmbedding(
        val rows: Int,
        val cols: Int,
        val data: FloatArray,
    )

    private fun loadVoiceEmbeddings(file: File): Map<String, VoiceEmbedding> {
        val result = linkedMapOf<String, VoiceEmbedding>()
        ZipFile(file).use { archive ->
            val entries = archive.entries()
            while (entries.hasMoreElements()) {
                val entry = entries.nextElement()
                if (entry.isDirectory || !entry.name.endsWith(".npy")) continue
                val bytes = archive.getInputStream(entry).use { it.readBytes() }
                result[entry.name.substringBeforeLast(".npy")] = parseNpy(bytes, entry.name)
            }
        }
        check(result.isNotEmpty()) { "No voice embeddings were found in ${file.name}" }
        return result
    }

    private fun parseNpy(bytes: ByteArray, name: String): VoiceEmbedding {
        check(bytes.size >= 12 && bytes.copyOfRange(0, 6).contentEquals(NPY_MAGIC)) {
            "$name is not an NPY array"
        }
        val major = bytes[6].toInt() and 0xff
        val headerOffset = if (major >= 2) 12 else 10
        val headerLength = if (major >= 2) {
            readLittleInt(bytes, 8)
        } else {
            (bytes[8].toInt() and 0xff) or ((bytes[9].toInt() and 0xff) shl 8)
        }
        check(headerOffset + headerLength <= bytes.size) { "$name has an invalid NPY header" }
        val header = bytes.copyOfRange(headerOffset, headerOffset + headerLength).toString(Charsets.US_ASCII)
        check("'fortran_order': False" in header || "\"fortran_order\": False" in header) {
            "$name uses unsupported Fortran array ordering"
        }
        val descriptor = DESCRIPTOR_REGEX.find(header)?.groupValues?.get(1)
        check(descriptor == "<f4" || descriptor == "|f4" || descriptor == ">f4") {
            "$name uses unsupported NPY type $descriptor"
        }
        val dimensions = SHAPE_REGEX.find(header)?.groupValues?.get(1)
            ?.split(',')
            ?.map(String::trim)
            ?.filter(String::isNotEmpty)
            ?.map(String::toInt)
            .orEmpty()
        check(dimensions.size == 2 && dimensions.all { it > 0 }) { "$name has unsupported shape $dimensions" }
        val count = dimensions[0] * dimensions[1]
        val dataOffset = headerOffset + headerLength
        check(bytes.size - dataOffset >= count * 4) { "$name has truncated float data" }
        val order = if (descriptor == ">f4") ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN
        val buffer = ByteBuffer.wrap(bytes, dataOffset, count * 4).slice().order(order).asFloatBuffer()
        val floats = FloatArray(count)
        buffer.get(floats)
        return VoiceEmbedding(dimensions[0], dimensions[1], floats)
    }

    private fun readLittleInt(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xff) or
            ((bytes[offset + 1].toInt() and 0xff) shl 8) or
            ((bytes[offset + 2].toInt() and 0xff) shl 16) or
            ((bytes[offset + 3].toInt() and 0xff) shl 24)

    private fun nanosToMs(nanos: Long): Double = nanos / 1_000_000.0

    companion object {
        private const val TAG = "KittenTtsCpu"
        private const val SAMPLE_RATE = 24_000
        private const val MAX_TOKENS_PER_CHUNK = 400
        private const val MAX_SILENCE_TRIM_MS = 250
        private const val SILENCE_THRESHOLD = 0.005f
        private const val DEFAULT_VOICE = "expr-voice-2-m" // Jasper
        private const val DEFAULT_SPEED = 0.8f
        private const val START_TOKEN = 0L
        private const val END_TOKEN = 10L
        private const val PAD_TOKEN = 0L
        private val EXPECTED_INPUTS = setOf("input_ids", "style", "speed")
        private val NPY_MAGIC = byteArrayOf(0x93.toByte(), 0x4e, 0x55, 0x4d, 0x50, 0x59)
        private val DESCRIPTOR_REGEX = Regex("['\\\"]descr['\\\"]\\s*:\\s*['\\\"]([^'\\\"]+)['\\\"]")
        private val SHAPE_REGEX = Regex("['\\\"]shape['\\\"]\\s*:\\s*\\(([^)]*)\\)")

        private const val SYMBOLS =
            "$;:,.!?¡¿—…\"«»“” ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" +
                "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘’̩‘ᵻ"
        private val SYMBOL_INDEX: Map<Int, Long> = buildMap {
            SYMBOLS.codePoints().forEachOrdered { codePoint -> put(codePoint, size.toLong()) }
        }
    }
}

internal class KittenPhonemizer : AutoCloseable {
    private var handle = 0L

    init {
        System.loadLibrary("ai-chat")
    }

    @Synchronized
    fun load(rules: File, list: File) {
        close()
        handle = nativeCreate(rules.absolutePath, list.absolutePath)
        check(handle != 0L) { "Could not load KittenTTS phonemizer" }
    }

    @Synchronized
    fun phonemize(text: String): String {
        check(handle != 0L) { "KittenTTS phonemizer is not loaded" }
        return checkNotNull(nativePhonemize(handle, text)) { "KittenTTS phonemization failed" }
    }

    @Synchronized
    override fun close() {
        if (handle != 0L) nativeDestroy(handle)
        handle = 0L
    }

    private external fun nativeCreate(rulesPath: String, listPath: String): Long
    private external fun nativePhonemize(handle: Long, text: String): String?
    private external fun nativeDestroy(handle: Long)
}
