package com.example.qmxgemma

/**
 * Process-scoped presentation state for the process-scoped native conversation.
 *
 * This intentionally stores only plain message data. Activities and Views must never be retained
 * here. Android discards this state naturally if it terminates the application process, at which
 * point the native model and conversation are also discarded.
 */
internal class ChatSessionState {
    private val mutableMessages = mutableListOf<ChatMessage>()

    val messages: List<ChatMessage>
        get() = mutableMessages

    var hasConversation: Boolean = false
        private set

    fun beginConversation() {
        hasConversation = true
    }

    fun markConversationCleared() {
        hasConversation = false
    }

    fun append(message: ChatMessage): Int {
        val index = mutableMessages.size
        mutableMessages.add(message)
        return index
    }

    fun replaceWith(message: ChatMessage): Int {
        val previousCount = mutableMessages.size
        mutableMessages.clear()
        mutableMessages.add(message)
        return previousCount
    }

    fun clear(): Int {
        val previousCount = mutableMessages.size
        mutableMessages.clear()
        return previousCount
    }

    companion object {
        val process = ChatSessionState()
    }
}
