package com.example.qmxgemma

/**
 * Coalesces native token chunks into UI-sized frames so RecyclerView is not
 * rebound faster than the display can present the changes.
 */
internal class StreamRenderThrottle(
    private val intervalMs: Long,
) {
    private var lastRenderAt: Long? = null

    init {
        require(intervalMs > 0) { "Render interval must be positive" }
    }

    fun shouldRender(nowMs: Long): Boolean {
        val previous = lastRenderAt
        if (previous != null && nowMs - previous < intervalMs) return false
        lastRenderAt = nowMs
        return true
    }
}
