import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Accelerometer, Gyroscope, Magnetometer } from 'expo-sensors';
import * as tf from '@tensorflow/tfjs';
//import * as FileSystem from 'expo-file-system';
import '@tensorflow/tfjs-react-native';
import { load } from 'react-native-fast-tflite';

// Ruta donde se encuentra el modelo TensorFlow Lite
const modeloTFLiteRuta = 'assets/modelo_tf_lite/model.tflite';

// Función para preparar los datos de entrada
const prepareInputData = (accelerometerValues, gyroscopeValues, magnetometerValues) => {
  // Calcular los valores necesarios
  const Ax = accelerometerValues?.[0] || 0.0;
  const Ay = accelerometerValues?.[1] || 0.0;
  const Az = accelerometerValues?.[2] || 0.0;

  const AyGx = Math.sqrt(Ay**2 + (gyroscopeValues?.[0] || 0.0)**2);
  const AxGx = Math.sqrt(Ax**2 + (gyroscopeValues?.[0] || 0.0)**2);
  const AyGy = Math.sqrt(Ay**2 + (gyroscopeValues?.[1] || 0.0)**2);
  const AxGy = Math.sqrt(Ax**2 + (gyroscopeValues?.[1] || 0.0)**2);
  const AzGx = Math.sqrt(Az**2 + (gyroscopeValues?.[0] || 0.0)**2);
  const Mxz = Math.sqrt((magnetometerValues?.[0] || 0.0)**2 + (magnetometerValues?.[2] || 0.0)**2);
  const MyGx = Math.sqrt((magnetometerValues?.[1] || 0.0)**2 + (gyroscopeValues?.[0] || 0.0)**2);
  const AyMz = Math.sqrt(Ay**2 + (magnetometerValues?.[2] || 0.0)**2);
  const Ayz = Math.sqrt(Ay**2 + Az**2);
  const Mz = magnetometerValues?.[2] || 0.0;
  const My = magnetometerValues?.[1] || 0.0;
  const Gzy = Math.sqrt((gyroscopeValues?.[2] || 0.0)**2 + (gyroscopeValues?.[1] || 0.0)**2);
  const MxGx = Math.sqrt((magnetometerValues?.[0] || 0.0)**2 + (gyroscopeValues?.[0] || 0.0)**2);
  const MxGy = Math.sqrt((magnetometerValues?.[0] || 0.0)**2 + (gyroscopeValues?.[1] || 0.0)**2);
  const Myz = Math.sqrt((magnetometerValues?.[1] || 0.0)**2 + (magnetometerValues?.[2] || 0.0)**2);
  const Mx = magnetometerValues?.[0] || 0.0;
  const Mxy = Math.sqrt((magnetometerValues?.[0] || 0.0)**2 + (magnetometerValues?.[1] || 0.0)**2);
  const AyMx = Math.sqrt(Ay**2 + (magnetometerValues?.[0] || 0.0)**2);
  const AzMx = Math.sqrt(Az**2 + (magnetometerValues?.[0] || 0.0)**2);
  const AyMy = Math.sqrt(Ay**2 + (magnetometerValues?.[1] || 0.0)**2);
  const Gz = gyroscopeValues?.[2] || 0.0;

  // Devolver un array con los valores preparados
  return [
    Az, AyGx, AxGx, AyGy, AxGy, AzGx, Mxz, MyGx, AyMz, Ayz, Mz, Ax, My, Gzy, Ay, MxGx, MxGy, Myz, Mx, Mxy, AyMx, AzMx, AyMy, Gz
  ];
};

const modeloURL = 'https://drive.google.com/uc?id=1OFCNV4qp1ihImNQb0zCOuvy7j4Ni9z_N';
//const pesoURL = 'https://drive.google.com/uc?id=1aUG-mkKdnssUtV9mWEyp4GVlsdN8iUP';

const App = () => {
  const [model, setModel] = useState(null);
  const [predictedClass, setPredictedClass] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        // Descarga del modelo JSON
        const response = await fetch(modeloURL);
        const modelJSON= await response.json();

        // Cargar el modelo desde un archivo local con react-native-fast-tflite
        const model = await tf.loadLayersModel(modelJSON);

        // Utilizar el modelo
        setModel(model);

        console.log('Modelo cargado correctamente');
      } catch (error) {
        console.error('Error cargando el modelo:', error);
      }
    };

    loadModel();
  }, []);

  useEffect(() => {
    // Función para manejar los datos del acelerómetro
    const handleAccelerometerData = ({ x, y, z }) => {
      const sensorData = prepareInputData([x, y, z], null, null);
      runModel(sensorData);
    };

    // Función para manejar los datos del giroscopio
    const handleGyroscopeData = ({ x, y, z }) => {
      const sensorData = prepareInputData(null, [x, y, z], null);
      runModel(sensorData);
    };

    // Función para manejar los datos del magnetómetro
    const handleMagnetometerData = ({ x, y, z }) => {
      const sensorData = prepareInputData(null, null, [x, y, z]);
      runModel(sensorData);
    };

    // Función para ejecutar el modelo
    const runModel = (sensorData) => {
      if (model) {
        // Utilizar el modelo para la inferencia
        model.predict(tf.tensor2d([sensorData])).array()
          .then((res) => {
            const confidence = Math.max(...res[0]);
            const predictedClass = res[0].indexOf(confidence) + 1;
            console.log(`Predicted Class: ${predictedClass}, Confidence: ${confidence * 100}%`);

            // Update the state with the predicted class and trigger a re-render
            setPredictedClass({ class: predictedClass, confidence: confidence * 100 });
          })
          .catch((error) => {
            console.error('Error al ejecutar el modelo:', error);
          });
      }
    };

    // Suscribirse a los sensores
    const subscribeToSensors = async () => {
      await Accelerometer.setUpdateInterval(1000);  // Frecuencia de actualización en milisegundos
      await Gyroscope.setUpdateInterval(1000);
      await Magnetometer.setUpdateInterval(1000);

      Accelerometer.addListener(handleAccelerometerData);
      Gyroscope.addListener(handleGyroscopeData);
      Magnetometer.addListener(handleMagnetometerData);
    };

    subscribeToSensors();

    return () => {
      // Desuscribirse de los sensores al desmontar el componente
      Accelerometer.removeAllListeners();
      Gyroscope.removeAllListeners();
      Magnetometer.removeAllListeners();
    };
  }, [model]);

  return (
    <View style={styles.container}>
      <Text>Real-time Sensor Data</Text>
      {predictedClass && (
        <Text>{`Predicted Class: ${predictedClass.class}, Confidence: ${predictedClass.confidence.toFixed(2)}%`}</Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default App;