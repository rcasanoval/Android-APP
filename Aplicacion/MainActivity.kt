package com.example.finalharapp

import android.content.res.AssetManager
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.pow
import kotlin.concurrent.timer

class MainActivity : AppCompatActivity(), SensorEventListener {

    private var sensorManager: SensorManager? = null
    private var accelerometerSensor: Sensor? = null
    private var gyroscopeSensor: Sensor? = null
    private var magnetometerSensor: Sensor? = null
    private var accelerationTextView: TextView? = null
    private var gyroscopeTextView: TextView? = null
    private var magnetometerTextView: TextView? = null
    private var outputTextView: TextView? = null

    // TensorFlow Lite
    private lateinit var tflite: Interpreter
    private lateinit var inputBuffer: ByteBuffer

    // Lista de TextViews y nombres de actividades
    private val activityTextViews = mutableListOf<TextView>()
    private val activityNames = mutableListOf<String>()

    private var timer: Timer? = null

    //Ficheros
    private var sensorDataFile: File? = null
    private var modelInputDataFile: File? = null
    private var sensorDataWriter: BufferedWriter? = null
    private var modelInputDataWriter: BufferedWriter? = null
    private var measurementsCount = 0


    // Variables para almacenar los valores de cada sensor
    private var accelerometerValues: FloatArray? = null
    private var gyroscopeValues: FloatArray? = null
    private var magnetometerValues: FloatArray? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Obtener referencias a los TextViews y nombres de actividades
        activityTextViews.apply {
            add(findViewById(R.id.textViewActivity1))
            add(findViewById(R.id.textViewActivity2))
            add(findViewById(R.id.textViewActivity3))
            add(findViewById(R.id.textViewActivity4))
            add(findViewById(R.id.textViewActivity5))
            add(findViewById(R.id.textViewActivity6))
        }

        activityNames.apply {
            add("Caminando")
            add("Metro")
            add("Bus")
            add("Conducir coche")
            add("Subir o bajar escaleras")
            add("Estar detenido")
        }

        // Inicializar el SensorManager
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        // Obtener sensores
        accelerometerSensor = sensorManager!!.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscopeSensor = sensorManager!!.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        magnetometerSensor = sensorManager!!.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

        // Configurar el modelo TensorFlow Lite
        try {
            val modelBuffer = loadModelFile()
            tflite = Interpreter(modelBuffer)
        } catch (e: IOException) {
            e.printStackTrace()
        }

        // Configurar el buffer de entrada
        inputBuffer = ByteBuffer.allocateDirect(4 * 1 * 1 * 24) // Tamaño ajustado según las necesidades del modelo
        inputBuffer.order(ByteOrder.nativeOrder())

        // Crear archivos y escritores para guardar datos
        createFiles()

        // Inicializar el temporizador para realizar mediciones cada segundo
        timer = Timer()
        timer?.scheduleAtFixedRate(object : TimerTask() {
            override fun run() {
                // Registrar listeners para los sensores con SENSOR_DELAY_NORMAL (500 milisegundos)
                sensorManager!!.registerListener(
                    this@MainActivity,
                    accelerometerSensor,
                    SensorManager.SENSOR_DELAY_NORMAL
                )
                sensorManager!!.registerListener(
                    this@MainActivity,
                    gyroscopeSensor,
                    SensorManager.SENSOR_DELAY_NORMAL
                )
                sensorManager!!.registerListener(
                    this@MainActivity,
                    magnetometerSensor,
                    SensorManager.SENSOR_DELAY_NORMAL
                )
            }
        }, 0, 1000) // 1000 milisegundos (1 segundo)
    }


    override fun onSensorChanged(event: SensorEvent) {
        // Manejar cambios en los valores de los sensores
        if (event.sensor == accelerometerSensor) {
            accelerometerValues = event.values
            accelerationTextView?.text = """
        Aceleración:
        X: ${accelerometerValues!![0]}
        Y: ${accelerometerValues!![1]}
        Z: ${accelerometerValues!![2]}
    """.trimIndent()

            // Agregar registro de depuración
            Log.d("SensorValues", "Acelerómetro: ${accelerometerValues!!.contentToString()}")

        } else if (event.sensor == gyroscopeSensor) {
            gyroscopeValues = event.values
            gyroscopeTextView?.text = """
        Giroscopio:
        X: ${gyroscopeValues!![0]}
        Y: ${gyroscopeValues!![1]}
        Z: ${gyroscopeValues!![2]}
    """.trimIndent()

            Log.d("SensorValues", "Giroscopio: ${gyroscopeValues!!.contentToString()}")

        } else if (event.sensor == magnetometerSensor) {
            magnetometerValues = event.values
            magnetometerTextView?.text = """
        Magnetómetro:
        X: ${magnetometerValues!![3]}
        Y: ${magnetometerValues!![4]}
        Z: ${magnetometerValues!![5]}
    """.trimIndent()

            // Agregar registro de depuración
            Log.d("SensorValues", "Magnetómetro: ${magnetometerValues!!.contentToString()}")

        }

        // Verificar si todos los sensores han enviado nuevos valores
        if (accelerometerValues != null && gyroscopeValues != null && magnetometerValues != null) {

            // Registrar valores en el archivo CSV
            writeSensorDataToCsv(System.currentTimeMillis(), accelerometerValues!!, gyroscopeValues!!, magnetometerValues!!)

            // Configurar los valores de entrada para el modelo
            configureInputBuffer(accelerometerValues!!, gyroscopeValues!!, magnetometerValues!!)

            // Realizar la inferencia
            runInference()

            // Resetear los valores de los sensores
            accelerometerValues = null
            gyroscopeValues = null
            magnetometerValues = null
        }

        // Verificar si se alcanzó el límite de 100 mediciones
        if (measurementsCount >= 100) {
            // Detener el temporizador
            timer?.cancel()
            timer?.purge()

            // Cerrar los escritores
            closeWriters()
        }
    }


    fun configureInputBuffer(accelerometerValues: FloatArray?, gyroscopeValues: FloatArray, magnetometerValues: FloatArray) {
        // Aquí configuras los valores de entrada para el modelo
        // asumiendo que se deben combinar con los valores actuales del acelerómetro
        // Verificar si accelerometerValues no es nulo antes de acceder a sus elementos

        // Factor de conversión de m/s^2 a g
        val g_conversion = 1 / 9.81

        val Ax = (accelerometerValues?.get(0) ?: 0.0f) * g_conversion
        val Ay = (accelerometerValues?.get(1) ?: 0.0f) * g_conversion
        val Az = (accelerometerValues?.get(2) ?: 0.0f) * g_conversion


        if (gyroscopeValues != null) {
            gyroscopeValues[0] = gyroscopeValues[0].toDegreesPerSecond()
            gyroscopeValues[1] = gyroscopeValues[1].toDegreesPerSecond()
            gyroscopeValues[2] = gyroscopeValues[2].toDegreesPerSecond()
        }

        val AyGx = kotlin.math.sqrt((Ay.pow(2))  + (gyroscopeValues?.get(0)?.pow(2) ?: 0.0f)).toFloat()
        val AxGx = kotlin.math.sqrt((Ax.pow(2)) + (gyroscopeValues?.get(0)?.pow(2) ?: 0.0f)).toFloat() // 'AxGx'
        val AyGy = kotlin.math.sqrt((Ay.pow(2)) + (gyroscopeValues?.get(1)?.pow(2) ?: 0.0f)).toFloat() // 'AyGy'
        val AxGy = kotlin.math.sqrt((Ax.pow(2)) + (gyroscopeValues?.get(1)?.pow(2) ?: 0.0f)).toFloat() // 'AxGy'
        val AzGx = kotlin.math.sqrt(Az.pow(2) + (gyroscopeValues?.get(0)?.pow(2) ?: 0.0f)).toFloat() // 'AzGx'
        val Mxz = kotlin.math.sqrt((magnetometerValues?.get(0)?.pow(2) ?: 0.0f) + (magnetometerValues?.get(2)?.pow(2) ?: 0.0f)).toFloat() //'Mxz'
        val MyGx = kotlin.math.sqrt((magnetometerValues?.get(1)?.pow(2) ?: 0.0f) + (gyroscopeValues?.get(0)?.pow(2) ?: 0.0f)).toFloat() //'MyGz'
        val AyMz = kotlin.math.sqrt((Ay.pow(2)) + (magnetometerValues?.get(2)?.pow(2) ?: 0.0f)).toFloat() //'AyMz'
        val Ayz = kotlin.math.sqrt((Ay.pow(2)) + (Az.pow(2))).toFloat() //'Ayz'
        val Mz = magnetometerValues[2] // 'Mz'
        val My = magnetometerValues[1] // 'My'
        val Gzy = kotlin.math.sqrt((gyroscopeValues?.get(2)?.pow(2) ?: 0.0f) + (gyroscopeValues?.get(1)?.pow(2) ?: 0.0f)).toFloat() // 'Gzy'
        val MxGx = kotlin.math.sqrt((magnetometerValues?.get(0)?.pow(2) ?: 0.0f) + (gyroscopeValues?.get(0)?.pow(2) ?: 0.0f)).toFloat() // 'MxGx'
        val MxGy = kotlin.math.sqrt((magnetometerValues?.get(0)?.pow(2) ?: 0.0f) + (gyroscopeValues?.get(1)?.pow(2) ?: 0.0f)).toFloat() // 'MxGy'
        val Myz = kotlin.math.sqrt((magnetometerValues?.get(1)?.pow(2) ?: 0.0f) + (magnetometerValues?.get(2)?.pow(2) ?: 0.0f)).toFloat() // 'Myz'
        val Mx = magnetometerValues[0] // 'Mx'
        val Mxy = kotlin.math.sqrt((magnetometerValues?.get(0)?.pow(2) ?: 0.0f) + (magnetometerValues?.get(1)?.pow(2) ?: 0.0f)).toFloat() // 'Mxy'
        val AyMx = kotlin.math.sqrt((Ay.pow(2)) + (magnetometerValues?.get(0)?.pow(2) ?: 0.0f)).toFloat() // 'AyMx'
        val AzMx = kotlin.math.sqrt((Az.pow(2)) + (magnetometerValues?.get(0)?.pow(2) ?: 0.0f)).toFloat() // 'AzMx'
        val AyMy = kotlin.math.sqrt((Ay.pow(2)) + (magnetometerValues?.get(1)?.pow(2) ?: 0.0f)).toFloat() // 'AyMy'
        val Gz = gyroscopeValues[2] // 'Gz'


        val modelInput = floatArrayOf(
            Az.toFloat(), // 'Az'
            AyGx, // 'AyGx'
            AxGx,// 'AxGx'
            AyGy, // 'AyGy'
            AxGy,// 'AxGy'
            AzGx, // 'AzGx'
            Mxz, //'Mxz'
            MyGx, //'MyGx'
            AyMz, //'AyMz'
            Ayz, //'Ayz'
            Mz,// 'Mz'
            Ax.toFloat(), // 'Ax'
            My, // 'My'
            Gzy, // 'Gzy'
            Ay.toFloat(), // 'Ay'
            MxGx, // 'MxGx'
            MxGy, // 'MxGy'
            Myz, // 'Myz'
            Mx, // 'Mx'
            Mxy, // 'Mxy'
            AyMx, // 'AyMx'
            AzMx, // 'AzMx'
            AyMy,// 'AyMy'
            Gz // 'Gz'
        )

        // Llamada a la función para escribir en el archivo de valores de entrada del modelo
        writeModelInputDataToCsv(modelInput)

        // Preprocesar y configurar los valores de entrada según las necesidades del modelo
        inputBuffer.rewind()

        // Asegúrate de que el búfer tenga suficiente capacidad
        if (inputBuffer.capacity() >= modelInput.size * 4) {
            for (value in modelInput) {
                inputBuffer.putFloat(value)
            }
        } else {
            Log.e("BufferOverflow", "El búfer no tiene capacidad suficiente para los datos.")
        }

        // Imprimir por pantalla los valores de entrada del modelo
        Log.d("ModelInput", "Valores de entrada del modelo: ${modelInput.joinToString(", ")}")
    }

    // Función de extensión para convertir radianes por segundo a grados por segundo
    fun Float.toDegreesPerSecond(): Float {
        return Math.toDegrees(this.toDouble()).toFloat()
    }

    fun runInference() {
        // Realizar la inferencia
        val outputBuffer = ByteBuffer.allocateDirect(1 * 6 * 4)
        tflite.run(inputBuffer, outputBuffer)

        // Mostrar los resultados sin escalar en el Log
        val rawResults = FloatArray(6)
        outputBuffer.rewind()
        for (i in 0 until 6) {
            rawResults[i] = outputBuffer.float
        }
        Log.d("RawResults", "Valores de salida sin escalar: ${rawResults.joinToString(", ")}")

        // Procesar y mostrar los resultados normalizados en el Log
        val normalizedResults = softmax(rawResults)
        Log.d("NormalizedResults", "Valores de salida normalizados: ${normalizedResults.joinToString(", ")}")

        // Actualizar la interfaz de usuario con los resultados de salida
        updateUIWithResults(normalizedResults)
    }

    // Función para aplicar la función softmax a un array de números
    fun softmax(x: FloatArray): FloatArray {
        val expX = x.map { kotlin.math.exp(it.toDouble()) }.toFloatArray()
        val sumExpX = expX.sum()
        return FloatArray(x.size) { index -> expX[index] / sumExpX }
    }

    fun List<Double>.toFloatArray(): FloatArray {
        return this.map { it.toFloat() }.toFloatArray()
    }


    private fun updateUIWithResults(results: FloatArray) {
        // Redondear y escalar los resultados a porcentajes (0% a 100%)
        val scaledResults = results.map { (it * 100).coerceIn(0f, 100f) }

        // Encontrar el índice de la actividad con el porcentaje más alto
        val maxIndex = scaledResults.indexOf(scaledResults.maxOrNull())

        // Actualizar cada TextView con su respectivo porcentaje y color de fondo
        for ((index, textView) in activityTextViews.withIndex()) {
            // Verificar si el porcentaje es NaN y asignar 0% en ese caso
            val scaledResult = if (scaledResults[index].isNaN()) 0f else scaledResults[index]

            // Formatear la cadena para mostrar solo 2 decimales
            val formattedResult = String.format("%.2f", scaledResult)
            textView.text = "${activityNames[index]}: $formattedResult%"

            // Cambiar el color de fondo solo si es la actividad con el porcentaje más alto
            if (index == maxIndex) {
                textView.setBackgroundColor(resources.getColor(android.R.color.holo_green_light, null))
            } else {
                // Restaurar el color de fondo predeterminado
                textView.setBackgroundColor(Color.TRANSPARENT)
            }
        }
    }




    fun procesarResultados(outputBuffer: ByteBuffer): FloatArray {
        // Supongamos que la salida del modelo es un conjunto de 6 valores flotantes
        val resultArray = FloatArray(6)

        // Leer los valores de salida del buffer
        outputBuffer.rewind()
        for (i in 0 until 6) {
            resultArray[i] = outputBuffer.float
        }

        // Escalar los valores a porcentajes (0% a 100%)
        val scaledResults = resultArray.map { it * 100 }.toFloatArray()

        // Devolver el arreglo de resultados
        return scaledResults
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        // Manejar cambios en la precisión de los sensores si es necesario
    }

    override fun onPause() {
        super.onPause()
        // Detener el temporizador
        timer?.cancel()
        timer?.purge()

        // Detener la lectura de los sensores cuando la actividad está en pausa
        sensorManager?.unregisterListener(this)


        //////////////////
        // Cerrar los escritores de archivos
        closeWriters()
    }

    override fun onResume() {
        super.onResume()
        // Registrar nuevamente los listeners al reanudar la actividad (el temporizador se reiniciará en `onCreate`)
        // No es necesario registrar los listeners aquí, ya que el temporizador se encarga de eso cada segundo.

        ///////////////////
        // Crear archivos y escritores para guardar datos
        createFiles()

    }

    private fun loadModelFile(): ByteBuffer {
        val assetManager: AssetManager = assets
        val inputStream: InputStream = assetManager.open("model_HAR_UPM_LSTM.tflite")
        val modelBuffer = ByteArray(inputStream.available())
        inputStream.read(modelBuffer)
        inputStream.close()

        val directBuffer = ByteBuffer.allocateDirect(modelBuffer.size)
        directBuffer.order(ByteOrder.nativeOrder())
        directBuffer.put(modelBuffer)
        directBuffer.rewind()

        return directBuffer
    }

    // Método para crear archivos y escritores
    private fun createFiles() {
        // Obtener la ruta del directorio de documentos
        val documentsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)

        // Crear la carpeta de la aplicación si no existe
        val appFolder = File(documentsDir, "HAR")
        if (!appFolder.exists()) {
            appFolder.mkdirs()
        }

        // Definir los archivos en la carpeta de la aplicación
        sensorDataFile = File(appFolder, "sensor_data.csv")
        modelInputDataFile = File(appFolder, "model_input_data.csv")

        // En el método createFiles()
        Log.d("FilePath", "Sensor Data File Path: ${sensorDataFile?.absolutePath}")
        Log.d("FilePath", "Model Input Data File Path: ${modelInputDataFile?.absolutePath}")

        // Crear escritores
        try {
            sensorDataWriter = BufferedWriter(FileWriter(sensorDataFile))
            modelInputDataWriter = BufferedWriter(FileWriter(modelInputDataFile))
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    // Método para escribir una línea en el archivo CSV de sensores
    private fun writeSensorDataToCsv(timestamp: Long, accelerometerValues: FloatArray?, gyroscopeValues: FloatArray, magnetometerValues: FloatArray) {
        val line = "$timestamp," +
                "${accelerometerValues?.joinToString(",") ?: "0,0,0"}," +
                "${gyroscopeValues.joinToString(",")}," +
                "${magnetometerValues.joinToString(",")}"

        // Verificar si el archivo está vacío y escribir las cabeceras si es necesario
        if (sensorDataFile?.length() == 0L) {
            val header = "Timestamp,AccX,AccY,AccZ,GyroX,GyroY,GyroZ,MagX,MagY,MagZ"
            writeCsvLine(sensorDataWriter, header)
        }

        writeCsvLine(sensorDataWriter, line)

        // Log para verificar si se está escribiendo correctamente
        Log.d("WriteSensorData", "Línea escrita en sensor_data.csv: $line")
    }
    // Método para escribir una línea en el archivo CSV de valores de entrada del modelo
    private fun writeModelInputDataToCsv(modelInput: FloatArray) {
        val line = "${modelInput.joinToString(",")}"

        // Log para verificar si se está escribiendo correctamente
        Log.d("WriteModelInputData", "Línea escrita en model_input_data.csv: $line")

        writeCsvLine(modelInputDataWriter, line)
    }
    // Método para escribir una línea en un archivo CSV
    private fun writeCsvLine(writer: BufferedWriter?, line: String) {
        try {
            writer?.write(line)
            writer?.newLine()
            writer?.flush()

            // Log para verificar si se está escribiendo correctamente
            Log.d("WriteCsvLine", "Línea escrita: $line")
        } catch (e: IOException) {
            e.printStackTrace()
            Log.e("FileWriteError", "Error al escribir en el archivo CSV: ${e.message}")
        }
    }
    // Método para cerrar los escritores al finalizar
    private fun closeWriters() {
        try {
            sensorDataWriter?.let {
                it.flush()
                it.close()
                sensorDataWriter = null
            }

            modelInputDataWriter?.let {
                it.flush()
                it.close()
                modelInputDataWriter = null
            }
        } catch (e: IOException) {
            e.printStackTrace()
            Log.e("FileWriteError", "Error al cerrar los escritores: ${e.message}")
        }
    }
}
