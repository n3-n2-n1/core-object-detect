import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';

// Elementos del DOM
const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('main-canvas');
const statusElement = document.getElementById('status');
const detectionsContainer = document.getElementById('detections-container'); // Contenedor
const detectionsListElement = document.getElementById('detections-list'); // Lista de detecciones
let model;

// Estado del contenedor
let containerExpanded = false;

// Umbral de confianza
const confidenceThreshold = 0.6;

// Cargar modelo COCO-SSD
async function loadModel() {
    try {
        statusElement.innerText = 'Loading model...';
        model = await cocoSsd.load({ base: 'mobilenet_v2' }); // Modelo más preciso
        statusElement.innerText = 'Model loaded. Starting webcam...';
        enableWebcam();
    } catch (error) {
        statusElement.innerText = 'Failed to load the model.';
        console.error('Error loading the model:', error);
    }
}

// Habilitar la webcam
async function enableWebcam() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        statusElement.innerText = 'Webcam not supported in this browser.';
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } // Resolución optimizada
        });
        webcamElement.srcObject = stream;

        webcamElement.addEventListener('loadeddata', () => {
            statusElement.innerText = '';
            predictWebcam();
        });
    } catch (error) {
        statusElement.innerText = 'Error accessing the webcam.';
        console.error('Webcam error:', error);
    }
}

// Predecir objetos en el feed de la webcam
function predictWebcam() {
    if (!model) {
        console.warn('Model not loaded yet.');
        return;
    }

    requestAnimationFrame(async () => {
        const predictions = await model.detect(webcamElement);

        // Filtrar predicciones por umbral de confianza
        const filteredPredictions = predictions.filter(pred => pred.score > confidenceThreshold);

        // Dibujar en el canvas
        drawPredictions(filteredPredictions);

        // Mostrar los datos detectados en el HTML
        handleDetections(filteredPredictions);

        // Continuar prediciendo
        predictWebcam();
    });
}

// Dibujar las predicciones en el canvas
function drawPredictions(predictions) {
    const context = canvasElement.getContext('2d');

    // Ajustar el tamaño del canvas al tamaño del video
    if (canvasElement.width !== webcamElement.videoWidth || canvasElement.height !== webcamElement.videoHeight) {
        canvasElement.width = webcamElement.videoWidth;
        canvasElement.height = webcamElement.videoHeight;
    }

    // Dibujar el cuadro del video en el canvas
    context.clearRect(0, 0, canvasElement.width, canvasElement.height); // Limpiar el canvas
    context.drawImage(webcamElement, 0, 0, canvasElement.width, canvasElement.height);

    // Dibujar las cajas delimitadoras y etiquetas
    predictions.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;

        context.strokeStyle = 'red';
        context.lineWidth = 2;
        context.font = '16px Roboto';
        context.fillStyle = 'red';

        // Dibujar el rectángulo
        context.strokeRect(x, y, width, height);

        // Etiqueta con el nombre y confianza
        context.fillText(
            `${prediction.class} (${(prediction.score * 100).toFixed(1)}%)`,
            x,
            y > 10 ? y - 5 : 10
        );
    });
}

// Manejar las detecciones
function handleDetections(predictions) {
    // Limpiar la lista anterior
    detectionsListElement.innerHTML = '';

    // Si hay predicciones, expande el contenedor
    if (predictions.length > 0) {
        if (!containerExpanded) {
            detectionsContainer.style.transition = 'max-height 0.5s ease-in-out';
            detectionsContainer.style.maxHeight = '400px'; // Expande el contenedor
            containerExpanded = true;
        }

        // Añadir cada detección como un elemento de la lista
        predictions.forEach(prediction => {
            const listItem = document.createElement('li');
            listItem.style.margin = '10px 0';
            listItem.style.fontSize = '1rem';
            listItem.style.lineHeight = '1.5';

            listItem.textContent = `
                Tipo: ${prediction.class}, 
                Confidencia: ${(prediction.score * 100).toFixed(1)}%, 
                Coordenadas: [${prediction.bbox.map(coord => coord.toFixed(1)).join(', ')}]
            `;

            detectionsListElement.appendChild(listItem);
        });
    } else {
        // Si no hay predicciones, colapsa el contenedor
        if (containerExpanded) {
            detectionsContainer.style.transition = 'max-height 0.5s ease-in-out';
            detectionsContainer.style.maxHeight = '0'; // Colapsa el contenedor
            containerExpanded = false;
        }
    }
}

// Configurar backend para WebGL
async function configureBackend() {
    const tf = await import('@tensorflow/tfjs');
    await tf.setBackend('webgl'); // Usa WebGL para mayor rendimiento
    console.log('Backend configurado a:', tf.getBackend());
}

// Inicializar la aplicación
configureBackend().then(loadModel);
