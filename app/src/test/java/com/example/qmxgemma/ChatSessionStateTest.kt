package com.example.qmxgemma

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatSessionStateTest {
    @Test
    fun `recreated UI observes the same transcript and conversation state`() {
        val state = ChatSessionState()
        val firstUiMessages = state.messages

        state.append(ChatMessage("First question", isUser = true))
        state.append(ChatMessage("First answer", isUser = false))
        state.beginConversation()

        val recreatedUiMessages = state.messages

        assertSame(firstUiMessages, recreatedUiMessages)
        assertEquals(listOf("First question", "First answer"), recreatedUiMessages.map { it.text })
        assertTrue(state.hasConversation)
    }

    @Test
    fun `clearing a conversation resets both transcript and conversation flag`() {
        val state = ChatSessionState()
        state.append(ChatMessage("Question", isUser = true))
        state.beginConversation()

        assertEquals(1, state.clear())
        state.markConversationCleared()

        assertTrue(state.messages.isEmpty())
        assertFalse(state.hasConversation)
    }
}
