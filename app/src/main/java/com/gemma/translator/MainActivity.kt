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
import android.widget.ArrayAdapter
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.gemma.translator.databinding.ActivityMainBinding
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
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

private data class Language(val code: String, val displayName: String, val englishName: String)

// Google翻訳のように選べる言語一覧。englishNameはLLMへのプロンプトで使う。
private val SUPPORTED_LANGUAGES = listOf(
    Language("ja", "日本語", "Japanese"),
    Language("en", "英語", "English"),
    Language("zh", "中国語(簡体字)", "Simplified Chinese"),
    Language("zh-TW", "中国語(繁体字)", "Traditional Chinese"),
    Language("ko", "韓国語", "Korean"),
    Language("fr", "フランス語", "French"),
    Language("de", "ドイツ語", "German"),
    Language("es", "スペイン語", "Spanish"),
    Language("pt", "ポルトガル語", "Portuguese"),
    Language("it", "イタリア語", "Italian"),
    Language("ru", "ロシア語", "Russian"),
    Language("ar", "アラビア語", "Arabic"),
    Language("hi", "ヒンディー語", "Hindi"),
    Language("th", "タイ語", "Thai"),
    Language("vi", "ベトナム語", "Vietnamese"),
    Language("id", "インドネシア語", "Indonesian"),
)

/**
 * Gemmaモデルをローカル推論させる常駐型翻訳スタンド。
 * 時計表示・翻訳・輝度調整をひとつの画面でまとめて提供する。
 * 推論ランタイムはLiteRT-LM(MediaPipe LLM Inference APIの後継)を使用する。
 */
class MainActivity : AppCompatActivity() {

    companion object {
        // 読み込むモデルファイル名。実機に配置するファイル名と一致させること。
        // litert-community/Gemma3-1B-IT (Hugging Face) の .litertlm 形式モデルを使用。
        private const val MODEL_FILE_NAME = "Gemma3-1B-IT_multi-prefill-seq_q4_ekv4096.litertlm"
        private const val REQUEST_STORAGE_PERMISSION = 1001
    }

    private lateinit var binding: ActivityMainBinding

    private var engine: Engine? = null
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
        setupLanguageSpinners()
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
        engine?.close()
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
    // 翻訳元/翻訳先言語の選択(Google翻訳風)
    // ------------------------------------------------------------------

    private fun setupLanguageSpinners() {
        val autoLabel = getString(R.string.label_auto_detect)
        val sourceNames = listOf(autoLabel) + SUPPORTED_LANGUAGES.map { it.displayName }
        val targetNames = SUPPORTED_LANGUAGES.map { it.displayName }

        binding.spinnerSourceLang.adapter =
            ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, sourceNames)
        binding.spinnerTargetLang.adapter =
            ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, targetNames)

        binding.spinnerSourceLang.setSelection(SUPPORTED_LANGUAGES.indexOfFirst { it.code == "ja" } + 1)
        binding.spinnerTargetLang.setSelection(SUPPORTED_LANGUAGES.indexOfFirst { it.code == "en" })

        binding.btnSwapLang.setOnClickListener {
            val sourcePos = binding.spinnerSourceLang.selectedItemPosition
            if (sourcePos == 0) return@setOnClickListener // 検出した言語は入れ替え不可
            val targetPos = binding.spinnerTargetLang.selectedItemPosition
            binding.spinnerSourceLang.setSelection(targetPos + 1)
            binding.spinnerTargetLang.setSelection(sourcePos - 1)
        }
    }

    /** nullは「検出した言語」(自動判定)を表す。 */
    private fun selectedSourceLanguage(): Language? {
        val pos = binding.spinnerSourceLang.selectedItemPosition
        return if (pos <= 0) null else SUPPORTED_LANGUAGES[pos - 1]
    }

    private fun selectedTargetLanguage(): Language {
        val pos = binding.spinnerTargetLang.selectedItemPosition.coerceIn(0, SUPPORTED_LANGUAGES.size - 1)
        return SUPPORTED_LANGUAGES[pos]
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
        if (isModelReady || engine != null) return

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
                val newEngine = Engine(
                    EngineConfig(
                        modelPath = modelPath,
                        cacheDir = cacheDir.path
                    )
                )
                newEngine.initialize()
                engine = newEngine
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
        val currentEngine = engine
        if (currentEngine == null || !isModelReady) return

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
            try {
                val systemInstruction = buildSystemInstruction()
                val conversationConfig = ConversationConfig(
                    systemInstruction = Contents.of(systemInstruction),
                    samplerConfig = SamplerConfig(topK = 40, topP = 0.95, temperature = 0.3),
                )
                val result = currentEngine.createConversation(conversationConfig).use { conversation ->
                    conversation.sendMessage(inputText).toString()
                }
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
     * スピナーで選択された翻訳元/翻訳先言語をもとにシステムプロンプトを組み立てる。
     * 翻訳元が「検出した言語」の場合は言語判定もLLMに任せる。
     */
    private fun buildSystemInstruction(): String {
        val target = selectedTargetLanguage()
        val source = selectedSourceLanguage()
        return if (source == null) {
            "You are an expert translator. Detect the language of the user's text and translate it into ${target.englishName}. " +
                "Output ONLY the translation result, with no explanation, notes, or quotation marks."
        } else {
            "You are an expert translator. Translate the user's text from ${source.englishName} to ${target.englishName}. " +
                "Output ONLY the translation result, with no explanation, notes, or quotation marks."
        }
    }
}
