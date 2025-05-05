package com.nhatnguyenba.tensorflow

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil3.compose.rememberAsyncImagePainter
import com.nhatnguyenba.tensorflow.ui.theme.TensorflowliteandroidexampleTheme
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.classifier.Classifications
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import java.io.IOException

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            TensorflowliteandroidexampleTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { padding ->
                    ImageClassifierApp(padding)
                }
            }
        }
    }
}

@Composable
fun ImageClassifierApp(
    paddingValues: PaddingValues
) {
    val context = LocalContext.current
    val bitmapState = remember { mutableStateOf<Bitmap?>(null) }
    val resultsState = remember { mutableStateOf<List<Classifications>?>(null) }
    val errorState = remember { mutableStateOf<String?>(null) }

    // Khởi tạo ImageClassifier
    val classifier = remember {
        try {
            ImageClassifier.createFromFile(context, "mobilenet_v1_1.0_224_quantized_1_metadata_1.tflite")
        } catch (e: IOException) {
            errorState.value = "Lỗi tải model: ${e.message}"
            null
        }
    }

    // Launcher để chọn ảnh
    val photoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            bitmapState.value = uri.toBitmap(context)
            classifyImage(bitmapState.value!!, classifier, resultsState, errorState)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(paddingValues)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Nút chọn ảnh
        Button(onClick = {
            photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }) {
            Text("Chọn ảnh")
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Hiển thị ảnh
        bitmapState.value?.let { bitmap ->
            Image(
                painter = rememberAsyncImagePainter(bitmap),
                contentDescription = "Selected image",
                modifier = Modifier.size(250.dp)
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Hiển thị kết quả
        when {
            errorState.value != null -> {
                Text("Lỗi: ${errorState.value}", color = Color.Red)
            }

            resultsState.value != null -> {
                ClassificationResults(resultsState.value!!)
            }

            classifier == null -> {
                CircularProgressIndicator()
                Text("Đang tải model...")
            }
        }
    }
}

@Composable
fun ClassificationResults(classifications: List<Classifications>) {
    Column {
        Text("Kết quả nhận dạng:", fontSize = 20.sp)

        classifications.first().categories.forEach { category ->
            Text(
                text = "${category.label}: ${"%.2f".format(category.score * 100)}%",
                fontSize = 16.sp,
                modifier = Modifier.padding(vertical = 4.dp)
            )
        }
    }
}

// Extension function chuyển Uri thành Bitmap
fun Uri.toBitmap(context: Context): Bitmap? {
    return try {
        MediaStore.Images.Media.getBitmap(context.contentResolver, this)
    } catch (e: IOException) {
        null
    }
}

// Hàm xử lý phân loại ảnh
fun classifyImage(
    bitmap: Bitmap,
    classifier: ImageClassifier?,
    resultsState: MutableState<List<Classifications>?>,
    errorState: MutableState<String?>
) {
    if (classifier == null) {
        errorState.value = "Model chưa được khởi tạo"
        return
    }

    try {
        // Tiền xử lý ảnh
        val image = TensorImage.fromBitmap(bitmap)

        // Phân loại
        val results = classifier.classify(image)

        // Cập nhật kết quả
        resultsState.value = results
        errorState.value = null
    } catch (e: Exception) {
        errorState.value = "Lỗi phân loại: ${e.message}"
    }
}