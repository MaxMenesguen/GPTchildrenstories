// Define the character set used in your model training
const chars = ['\n', ' ', '!', "'", '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '—', '`', '’', '“', '”'];
const stoi = {};
chars.forEach((char, index) => stoi[char] = index);
const sequence_length = 64;

// Encode function
function encode(str) {
    console.log("Encoding input text...");
    return str.split('').map(char => stoi[char] || 0);
}


function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scaled = logits.map(logit => Math.exp(logit - maxLogit));
    const total = scaled.reduce((acc, val) => acc + val, 0);
    return scaled.map(val => val / total);
}

function sampleIndex(probabilities) {
    const rnd = Math.random();
    let cumSum = 0;
    for (let i = 0; i < probabilities.length; i++) {
        cumSum += probabilities[i];
        if (rnd < cumSum) return i;
    }
    return probabilities.length - 1;
}

// Global variable for the ONNX session
let session;

// Load the ONNX model using ONNX Runtime Web
async function loadModel() {
    console.log('Model loading...');
    try {
        session = await ort.InferenceSession.create("model.onnx", { executionProviders: ['cpu', 'wasm'] });
        console.log('Model loaded');
        console.log('Session:', session); // Ajouter un log pour inspecter l'objet session
        document.getElementById('generateButton').disabled = false;
    } catch (error) {
        console.error('Error loading model:', error);
    }
}
async function testModel() {
    if (!session) {
        console.error('Model not loaded yet');
        return;
    }

    let inputText = document.getElementById('inputContext').value;
    let generatedText = inputText; // Initialiser avec le texte d'entrée
    let encodedInput = encode(inputText);
    let tokenInput = document.getElementById('tokenInput').value;
    

    // Déclaration de outputElement à l'intérieur de testModel
    let outputElement = document.getElementById('outputText');
    outputElement.innerText = inputText;

    for (let i = 0; i < tokenInput; i++) { 
        while (encodedInput.length < sequence_length) {
            encodedInput.push(0);
        }
        encodedInput = encodedInput.slice(-sequence_length);

        const bigIntEncodedInput = encodedInput.map(num => BigInt(num));
        let inputTensor = new ort.Tensor("int64", new BigInt64Array(bigIntEncodedInput), [1, sequence_length]);

        try {
            const outputMap = await session.run({ input: inputTensor });
            const outputTensor = outputMap.output;
            
            const lastLogits = outputTensor.data.slice(-83);
            const probabilities = softmax(lastLogits);
            const nextTokenIndex = sampleIndex(probabilities);

            generatedText += chars[nextTokenIndex]; // Mettre à jour le texte généré
            outputElement.innerText = generatedText; // Mettre à jour le texte généré sur la page

            encodedInput.push(nextTokenIndex); // Mettre à jour la séquence d'entrée
        } catch (error) {
            console.error('Error during model run:', error);
            break; // Arrêter la boucle en cas d'erreur
        }
    }

    console.log("Generated text:", generatedText);
}


// Event listener for the test button
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('generateButton').addEventListener('click', testModel);
    loadModel();
});
