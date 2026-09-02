package com.example.qmxassistant

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class AssistantSessionTest {
    @Test
    fun `assistant placeholder is replaced without losing the user turn`() {
        val session = AssistantSession()
        session.addUser("What is QMX?")
        val assistant = session.beginAssistant("Preparing")

        session.updateAssistant(assistant, "QMX accelerates matrix operations on the CPU.")

        assertEquals(2, session.turns.size)
        assertEquals(Speaker.USER, session.turns[0].speaker)
        assertEquals("QMX accelerates matrix operations on the CPU.", session.turns[1].text)
    }

    @Test
    fun `clear removes conversation and generated audio`() {
        val session = AssistantSession()
        session.addUser("Hello")
        session.lastAudioPath = "/tmp/answer.wav"

        session.clear()

        assertEquals(emptyList<AssistantTurn>(), session.turns)
        assertNull(session.lastAudioPath)
    }

    @Test
    fun `conversation context preserves roles and most recent turns`() {
        val session = AssistantSession()
        session.addUser("first question")
        session.beginAssistant("first answer")
        session.addUser("follow up")

        assertEquals(
            "Assistant: first answer\nUser: follow up",
            session.conversationContext(40),
        )
    }
}
