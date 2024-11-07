package com.example.drleaf

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.annotation.Nullable
import android.media.ThumbnailUtils
import android.util.Log
import com.example.drleaf.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var textViewResult: TextView
    private lateinit var model: Model

    private val imageSize = 128

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        textViewResult = findViewById(R.id.textViewResult)
        val buttonCaptureImage: Button = findViewById(R.id.buttonCaptureImage)
        val buttonSelectImage: Button = findViewById(R.id.buttonSelectImage)

        buttonCaptureImage.setOnClickListener {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(intent, 3)
        }

        buttonSelectImage.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, 4)
        }

        try {
            model = Model.newInstance(this)
        } catch (e: IOException) {
            e.printStackTrace()
            // TODO: Handle the exception
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == Activity.RESULT_OK) {
            when (requestCode) {
                3 -> {
                    val image = data?.extras?.get("data") as? Bitmap
                    image?.let {
                        val dimension = minOf(it.width, it.height)
                        val thumbnail = ThumbnailUtils.extractThumbnail(it, dimension, dimension)
                        imageView.setImageBitmap(thumbnail)

                        val resizedImage = Bitmap.createScaledBitmap(thumbnail, imageSize, imageSize, false)
                        classifyImage(resizedImage)
                    }
                }
                4 -> {
                    val imageUri = data?.data
                    imageUri?.let {
                        try {
                            val image = MediaStore.Images.Media.getBitmap(this.contentResolver, it)
                            image?.let { img ->
                                imageView.setImageBitmap(img)

                                val resizedImage = Bitmap.createScaledBitmap(img, imageSize, imageSize, false)
                                classifyImage(resizedImage)
                            }
                        } catch (e: IOException) {
                            e.printStackTrace()
                            // TODO: Handle the exception
                        }
                    }
                }
            }
        }
    }

    private fun classifyImage(image: Bitmap) {
        try {
            val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3).apply {
                order(ByteOrder.nativeOrder())
            }

            val resizedImage = Bitmap.createScaledBitmap(image, imageSize, imageSize, true)
            val intValues = IntArray(imageSize * imageSize)
            resizedImage.getPixels(intValues, 0, resizedImage.width, 0, 0, resizedImage.width, resizedImage.height)

            intValues.forEach { value ->
                byteBuffer.putFloat(((value shr 16) and 0xFF) / 255.0f)
                byteBuffer.putFloat(((value shr 8) and 0xFF) / 255.0f)
                byteBuffer.putFloat((value and 0xFF) / 255.0f)
            }

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, imageSize, imageSize, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(byteBuffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            val confidences = outputFeature0.floatArray
            Log.d("accuracy", confidences.contentToString())
            val maxPos = confidences.filter { it >= 0.85  }.indices.maxByOrNull { confidences[it] } ?: 5



            val classes = arrayOf("Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back", "Gall Midge", "Healthy Leaf","Powdery Mildew", "Sooty Mould" )
            textViewResult.text = classes[maxPos]

        } catch (e: Exception) {
            e.printStackTrace()
            // TODO: Handle the exception
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }
}
