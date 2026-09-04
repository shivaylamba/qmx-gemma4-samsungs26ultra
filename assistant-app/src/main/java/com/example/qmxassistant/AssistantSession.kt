package com.example.qmxassistant

internal enum class Speaker { USER, ASSISTANT }

internal data class AssistantTurn(val speaker: Speaker, var text: String)

internal class AssistantSession {
    private val mutableTurns = mutableListOf<AssistantTurn>()

    val turns: List<AssistantTurn>
        get() = mutableTurns

    var lastAudioPath: String? = null

    fun addUser(text: String) {
        mutableTurns.add(AssistantTurn(Speaker.USER, text))
    }

    fun beginAssistant(placeholder: String): Int {
        mutableTurns.add(AssistantTurn(Speaker.ASSISTANT, placeholder))
        return mutableTurns.lastIndex
    }

    fun updateAssistant(index: Int, text: String) {
        require(mutableTurns[index].speaker == Speaker.ASSISTANT)
        mutableTurns[index].text = text
    }

    fun clear() {
        mutableTurns.clear()
        lastAudioPath = null
    }

    fun conversationContext(maxCharacters: Int): String {
        require(maxCharacters > 0)
        val selected = ArrayDeque<String>()
        var used = 0
        for (turn in mutableTurns.asReversed()) {
            val label = if (turn.speaker == Speaker.USER) "User" else "Assistant"
            val line = "$label: ${turn.text}"
            if (selected.isNotEmpty() && used + line.length + 1 > maxCharacters) break
            selected.addFirst(line.take(maxCharacters))
            used += line.length + 1
        }
        return selected.joinToString("\n")
    }

    companion object {
        val process = AssistantSession()
    }
}
