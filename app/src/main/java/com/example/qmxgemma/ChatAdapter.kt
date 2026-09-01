package com.example.qmxgemma

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

internal class ChatAdapter(
    private val messages: List<ChatMessage>,
) : RecyclerView.Adapter<ChatAdapter.MessageViewHolder>() {
    override fun getItemViewType(position: Int): Int = if (messages[position].isUser) USER else ASSISTANT

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MessageViewHolder {
        val layout = if (viewType == USER) {
            R.layout.item_message_user
        } else {
            R.layout.item_message_assistant
        }
        val view = LayoutInflater.from(parent.context).inflate(layout, parent, false)
        val textId = if (viewType == USER) R.id.userMessageText else R.id.assistantMessageText
        return MessageViewHolder(view, textId)
    }

    override fun onBindViewHolder(holder: MessageViewHolder, position: Int) {
        holder.messageText.text = messages[position].text
    }

    override fun getItemCount(): Int = messages.size

    class MessageViewHolder(itemView: View, textId: Int) : RecyclerView.ViewHolder(itemView) {
        val messageText: TextView = itemView.findViewById(textId)
    }

    private companion object {
        const val USER = 1
        const val ASSISTANT = 2
    }
}
