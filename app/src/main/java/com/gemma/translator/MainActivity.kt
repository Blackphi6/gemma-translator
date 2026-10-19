package com.gemma.translator

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.view.View
import android.view.WindowManager
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.gemma.translator.databinding.ActivityMainBinding
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * Gemmaモデルをローカル推論させる常駐型翻訳スタンド。
 * 時計表示・翻訳・輝度調整をひとつの画面でまとめて提供する。
 */
class MainActivity : AppCompatActivity() {

    companion object {
        // 読み込むモデルファイル名。実機に配置するファイル名と一致させること。
        // litert-community/Gemma3-1B-IT (Hugging Face) の .task 形式モデルを使用。
        private const val MODEL_FILE_NAME = "Gemma3-1B-IT_multi-prefill-seq_q4_ekv2048.task"
        private const val REQUEST_STORAGE_PERMISSION = 1001
    }

    private lateinit var binding: ActivityMainBinding

    private var llmInference: LlmInference? = null
    private var isModelReady = false
    private var isTranslating = false
    private var clockJob: Job? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 常時表示のスタンドとして画面が消灯しないようにする
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setupBrightnessSlider()
        binding.btnTranslate.setOnClickListener { onTranslateClicked() }
        binding.btnGrantPermission.setOnClickListener { requestStoragePermission() }

        startClock()
        tryLoadModel()
    }

    override fun onResume() {
        super.onResume()
        // ストレージ設定画面などから戻ってきたときに、未読み込みなら再トライする
        if (!isModelReady) {
            tryLoadModel()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        clockJob?.cancel()
        llmInference?.close()
    }

    // ------------------------------------------------------------------
    // 時計表示
    // ------------------------------------------------------------------

    private fun startClock() {
        val formatter = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
        clockJob = lifecycleScope.launch {
            while (isActive) {
                binding.tvClock.text = formatter.format(System.currentTimeMillis())
                delay(1000)
            }
        }
    }

    // ------------------------------------------------------------------
    // 画面輝度調整(古い端末のバッテリー・焼き付き対策)
    // ------------------------------------------------------------------

    private fun setupBrightnessSlider() {
        val currentBrightness = try {
            Settings.System.getInt(contentResolver, Settings.System.SCREEN_BRIGHTNESS)
        } catch (e: Settings.SettingNotFoundException) {
            128
        }
        binding.seekBrightness.progress = (currentBrightness * 100 / 255)

        binding.seekBrightness.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (!fromUser) return
                val safeProgress = progress.coerceIn(5, 100)
                val params = window.attributes
                params.screenBrightness = safeProgress / 100f
                window.attributes = params
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    // ------------------------------------------------------------------
    // ストレージ権限
    // ------------------------------------------------------------------

    private fun hasStoragePermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            Environment.isExternalStorageManager()
        } else {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.READ_EXTERNAL_STORAGE
            ) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requestStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            try {
                val intent = Intent(
                    Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    Uri.parse("package:$packageName")
                )
                startActivity(intent)
            } catch (e: Exception) {
                startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
            }
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                REQUEST_STORAGE_PERMISSION
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_STORAGE_PERMISSION) {
            tryLoadModel()
        }
    }

    // ------------------------------------------------------------------
    // モデル読み込み
    // ------------------------------------------------------------------

    private fun tryLoadModel() {
        if (isModelReady || llmInference != null) return

        binding.tvStatus.text = getString(R.string.status_model_loading)
        binding.btnGrantPermission.visibility = View.GONE
        binding.btnTranslate.isEnabled = false

        lifecycleScope.launch(Dispatchers.IO) {
            val modelPath = resolveModelPath()
            if (modelPath == null) {
                withContext(Dispatchers.Main) { onModelMissing() }
                return@launch
            }
            try {
                val options = LlmInference.LlmInferenceOptions.builder()
                    .setModelPath(modelPath)
                    .setMaxTokens(512)
                    .setTopK(40)
                    .setTemperature(0.3f)
                    .setRandomSeed(0)
                    .build()
                val inference = LlmInference.createFromOptions(this@MainActivity, options)
                llmInference = inference
                isModelReady = true
                withContext(Dispatchers.Main) { onModelReady() }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { onModelError(e) }
            }
        }
    }

    /**
     * モデルファイルの場所を優先順位をつけて探す。
     * 1. アプリ専用の外部ストレージ領域(権限不要・推奨)
     * 2. /sdcard/Download(要ストレージ権限、仕様に挙げられた配置例)
     * 3. assets内(アプリに同梱した場合)
     */
    private fun resolveModelPath(): String? {
        getExternalFilesDir(null)?.let { dir ->
            val file = File(dir, MODEL_FILE_NAME)
            if (file.exists()) return file.absolutePath
        }

        if (hasStoragePermission()) {
            val downloadFile = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
                MODEL_FILE_NAME
            )
            if (downloadFile.exists()) return downloadFile.absolutePath
        }

        if (assets.list("")?.contains(MODEL_FILE_NAME) == true) {
            return copyAssetToFilesDir(MODEL_FILE_NAME)
        }

        return null
    }

    private fun copyAssetToFilesDir(assetName: String): String {
        val outFile = File(filesDir, assetName)
        if (!outFile.exists()) {
            assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return outFile.absolutePath
    }

    private fun onModelReady() {
        binding.tvStatus.text = getString(R.string.status_model_ready)
        binding.btnGrantPermission.visibility = View.GONE
        binding.btnTranslate.isEnabled = true
    }

    private fun onModelMissing() {
        val appDir = getExternalFilesDir(null)?.absolutePath ?: "(取得不可)"
        binding.tvStatus.text = getString(R.string.status_model_missing, appDir, MODEL_FILE_NAME)
        binding.btnTranslate.isEnabled = false
        if (!hasStoragePermission()) {
            binding.btnGrantPermission.visibility = View.VISIBLE
        }
    }

    private fun onModelError(e: Exception) {
        binding.tvStatus.text = getString(R.string.status_model_error, e.message ?: e.toString())
        binding.btnTranslate.isEnabled = false
    }

    // ------------------------------------------------------------------
    // 翻訳実行
    // ------------------------------------------------------------------

    private fun onTranslateClicked() {
        if (isTranslating) return
        val inference = llmInference
        if (inference == null || !isModelReady) return

        val inputText = binding.etInput.text?.toString()?.trim().orEmpty()
        if (inputText.isEmpty()) {
            Toast.makeText(this, getString(R.string.toast_empty_input), Toast.LENGTH_SHORT).show()
            return
        }

        isTranslating = true
        binding.btnTranslate.isEnabled = false
        binding.progressBar.visibility = View.VISIBLE
        binding.tvStatus.text = getString(R.string.status_translating)

        lifecycleScope.launch(Dispatchers.Default) {
            val prompt = buildPrompt(inputText)
            try {
                val result = inference.generateResponse(prompt)
                withContext(Dispatchers.Main) { onTranslateFinished(result.trim()) }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { onTranslateError(e) }
            }
        }
    }

    private fun onTranslateFinished(result: String) {
        binding.tvResult.text = result
        binding.progressBar.visibility = View.GONE
        binding.tvStatus.text = getString(R.string.status_model_ready)
        binding.btnTranslate.isEnabled = true
        isTranslating = false
    }

    private fun onTranslateError(e: Exception) {
        binding.progressBar.visibility = View.GONE
        binding.tvStatus.text = getString(R.string.status_translate_error, e.message ?: e.toString())
        binding.btnTranslate.isEnabled = true
        isTranslating = false
    }

    /**
     * 入力に日本語が含まれていれば英訳、それ以外は和訳するようにターゲット言語を切り替える。
     * システムプロンプトでLLMの出力を翻訳結果のみに限定する。
     */
    private fun buildPrompt(inputText: String): String {
        val targetLanguage = if (containsJapanese(inputText)) "English" else "Japanese"
        return "You are an expert translator. Translate the following text into $targetLanguage. " +
            "Output ONLY the translation result, with no explanation, notes, or quotation marks.\n\n" +
            "Text: $inputText\nTranslation:"
    }

    private fun containsJapanese(text: String): Boolean {
        return text.any { ch ->
            ch in '぀'..'ゟ' || // ひらがな
                ch in '゠'..'ヿ' || // カタカナ
                ch in '一'..'鿿' // 漢字
        }
    }
}
