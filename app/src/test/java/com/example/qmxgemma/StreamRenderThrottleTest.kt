package com.example.qmxgemma

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class StreamRenderThrottleTest {
    @Test
    fun rendersFirstChunkImmediatelyAndBatchesFollowingChunks() {
        val throttle = StreamRenderThrottle(intervalMs = 48)

        assertTrue(throttle.shouldRender(nowMs = 1_000))
        assertFalse(throttle.shouldRender(nowMs = 1_010))
        assertFalse(throttle.shouldRender(nowMs = 1_047))
        assertTrue(throttle.shouldRender(nowMs = 1_048))
    }
}
