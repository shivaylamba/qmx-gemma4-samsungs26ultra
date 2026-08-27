package com.example.qmxgemma

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import android.os.Environment
import android.os.StatFs
import java.io.File
import java.security.MessageDigest

class HuggingFaceModelDownload(private val context: Context) {
    sealed interface State {
        data object Idle : State
        data class Active(val status: Int, val downloadedBytes: Long, val totalBytes: Long) : State
        data object Complete : State
        data class Failed(val reason: Int) : State
    }

    private val downloadManager = context.getSystemService(DownloadManager::class.java)
    private val preferences = context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)

    val destinationFile: File
        get() = File(
            requireNotNull(context.getExternalFilesDir(null)) { "External app storage is unavailable" },
            "models/$MODEL_FILENAME",
        )

    val hasTrackedDownload: Boolean
        get() = trackedDownloadId() != NO_DOWNLOAD_ID

    fun enqueue(): Long {
        val destination = destinationFile
        destination.parentFile?.mkdirs()
        val availableBytes = StatFs(destination.parentFile!!.absolutePath).availableBytes
        check(availableBytes >= MODEL_SIZE_BYTES + FREE_SPACE_HEADROOM_BYTES) {
            "Not enough free space. Downloading this model requires at least " +
                "${formatBytes(MODEL_SIZE_BYTES + FREE_SPACE_HEADROOM_BYTES)} free."
        }

        if (destination.exists()) {
            check(destination.delete()) { "Could not replace the incomplete model download" }
        }
        clearVerification()

        val request = DownloadManager.Request(Uri.parse(MODEL_DOWNLOAD_URL))
            .setTitle("Gemma 4 E2B Q8_0")
            .setDescription("Downloading the on-device GGUF model from Hugging Face")
            .setMimeType("application/octet-stream")
            .setAllowedOverMetered(true)
            .setAllowedOverRoaming(false)
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            .setDestinationUri(Uri.fromFile(destination))

        return downloadManager.enqueue(request).also { downloadId ->
            preferences.edit().putLong(KEY_DOWNLOAD_ID, downloadId).apply()
        }
    }

    fun query(): State {
        val downloadId = trackedDownloadId()
        if (downloadId == NO_DOWNLOAD_ID) return State.Idle

        val query = DownloadManager.Query().setFilterById(downloadId)
        downloadManager.query(query)?.use { cursor ->
            if (!cursor.moveToFirst()) return State.Idle
            val status = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
            val downloaded = cursor.getLong(
                cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR),
            )
            val total = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
            return when (status) {
                DownloadManager.STATUS_SUCCESSFUL -> State.Complete
                DownloadManager.STATUS_FAILED -> State.Failed(
                    cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON)),
                )
                else -> State.Active(status, downloaded, total)
            }
        }
        return State.Idle
    }

    fun cancel() {
        val downloadId = trackedDownloadId()
        if (downloadId != NO_DOWNLOAD_ID) downloadManager.remove(downloadId)
        clearTracking()
        val destination = destinationFile
        if (destination.exists()) destination.delete()
        clearVerification()
    }

    fun clearTracking() {
        preferences.edit().remove(KEY_DOWNLOAD_ID).apply()
    }

    fun isVerified(): Boolean {
        val file = destinationFile
        return file.isFile &&
            file.length() == MODEL_SIZE_BYTES &&
            preferences.getLong(KEY_VERIFIED_LENGTH, -1L) == file.length() &&
            preferences.getLong(KEY_VERIFIED_MODIFIED, -1L) == file.lastModified()
    }

    fun verifyAndRemember(): Boolean {
        val file = destinationFile
        if (!file.isFile || file.length() != MODEL_SIZE_BYTES) return false

        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered(HASH_BUFFER_BYTES).use { input ->
            val buffer = ByteArray(HASH_BUFFER_BYTES)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        val actual = digest.digest().joinToString("") { byte -> "%02x".format(byte) }
        if (!actual.equals(MODEL_SHA256, ignoreCase = true)) return false

        preferences.edit()
            .putLong(KEY_VERIFIED_LENGTH, file.length())
            .putLong(KEY_VERIFIED_MODIFIED, file.lastModified())
            .apply()
        return true
    }

    fun discardInvalidFile() {
        val destination = destinationFile
        if (destination.exists()) destination.delete()
        clearVerification()
        clearTracking()
    }

    private fun trackedDownloadId(): Long = preferences.getLong(KEY_DOWNLOAD_ID, NO_DOWNLOAD_ID)

    private fun clearVerification() {
        preferences.edit()
            .remove(KEY_VERIFIED_LENGTH)
            .remove(KEY_VERIFIED_MODIFIED)
            .apply()
    }

    companion object {
        const val MODEL_FILENAME = "gemma-4-E2B-it-Q8_0.gguf"
        const val MODEL_SIZE_BYTES = 4_967_497_152L
        const val MODEL_SHA256 = "996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a"
        const val MODEL_DOWNLOAD_URL =
            "https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/" +
                "$MODEL_FILENAME?download=true"

        private const val PREFERENCES_NAME = "hugging_face_model_download"
        private const val KEY_DOWNLOAD_ID = "download_id"
        private const val KEY_VERIFIED_LENGTH = "verified_length"
        private const val KEY_VERIFIED_MODIFIED = "verified_modified"
        private const val NO_DOWNLOAD_ID = -1L
        private const val HASH_BUFFER_BYTES = 4 * 1024 * 1024
        private const val FREE_SPACE_HEADROOM_BYTES = 512L * 1024 * 1024

        fun formatBytes(bytes: Long): String = when {
            bytes < 0 -> "unknown size"
            bytes >= 1_000_000_000 -> "%.2f GB".format(bytes / 1_000_000_000.0)
            bytes >= 1_000_000 -> "%.1f MB".format(bytes / 1_000_000.0)
            else -> "$bytes B"
        }
    }
}
