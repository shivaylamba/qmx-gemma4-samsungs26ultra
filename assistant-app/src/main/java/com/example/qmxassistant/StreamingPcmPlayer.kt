package com.example.qmxassistant

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import com.arm.aichat.PcmChunkListener
import java.io.Closeable
import kotlin.math.max

internal class StreamingPcmPlayer : PcmChunkListener, Closeable {
    private val lock = Any()
    private var track: AudioTrack? = null
    private var sampleRate = 0

    override fun onPcmChunk(samples: ShortArray, sampleRate: Int, isFinal: Boolean) {
        synchronized(lock) {
            if (samples.isNotEmpty()) {
                val audioTrack = obtainTrack(sampleRate)
                if (audioTrack.playState != AudioTrack.PLAYSTATE_PLAYING) audioTrack.play()
                val written = audioTrack.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
                check(written == samples.size) { "AudioTrack accepted $written of ${samples.size} PCM samples" }
            }
            if (isFinal && samples.isEmpty() && track == null) return
        }
    }

    private fun obtainTrack(requestedSampleRate: Int): AudioTrack {
        track?.let {
            check(sampleRate == requestedSampleRate) { "PCM sample rate changed during synthesis" }
            return it
        }
        val minimum = AudioTrack.getMinBufferSize(
            requestedSampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        check(minimum > 0) { "Android rejected the PCM output format" }
        val created = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANT)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(requestedSampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
            )
            .setBufferSizeInBytes(max(minimum, requestedSampleRate * Short.SIZE_BYTES))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
            .build()
        check(created.state == AudioTrack.STATE_INITIALIZED) { "Could not initialize PCM playback" }
        sampleRate = requestedSampleRate
        track = created
        return created
    }

    override fun close() {
        synchronized(lock) {
            track?.let {
                runCatching { it.pause() }
                runCatching { it.flush() }
                it.release()
            }
            track = null
            sampleRate = 0
        }
    }
}
