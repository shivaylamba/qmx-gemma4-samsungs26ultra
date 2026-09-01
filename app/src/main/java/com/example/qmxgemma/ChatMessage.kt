package com.example.qmxgemma

import java.util.concurrent.atomic.AtomicLong

internal data class ChatMessage(
    var text: String,
    val isUser: Boolean,
    val id: Long = nextId.getAndIncrement(),
) {
    private companion object {
        val nextId = AtomicLong(1)
    }
}
