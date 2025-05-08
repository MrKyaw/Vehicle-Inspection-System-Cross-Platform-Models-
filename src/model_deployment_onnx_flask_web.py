
# Just sample code to demonstrate how to deploy a model using Flask and ONNX Runtime
from flask import Flask, request, jsonify, render_template_string
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Load the ONNX model
onnx_model_path = "resnet18.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Preprocess function
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes))
    input_tensor = transform(image).unsqueeze(0)
    return image, input_tensor.numpy()  # Return both the image and numpy array

# Function to convert image to base64 for embedding in HTML
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Define the root route to handle the base URL
@app.route("/")
def home():
    return render_template_string("""
        <html>
            <body>
                <h1>Upload an image for prediction</h1>
                <form action="/predict" method="POST" enctype="multipart/form-data">
                    <input type="file" name="file">
                    <button type="submit">Submit</button>
                </form>
            </body>
        </html>
    """)

# Define the favicon route to handle favicon.ico requests
@app.route("/favicon.ico")
def favicon():
    return "", 204  # No content response

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get the image bytes from the request
    image_bytes = file.read()
    image, input_numpy = preprocess_image(image_bytes)

    # Run inference with ONNX Runtime
    inputs = {ort_session.get_inputs()[0].name: input_numpy}
    outputs = ort_session.run(None, inputs)

    # Get the predicted class (assuming single output)
    predicted_class = np.argmax(outputs[0])

    # Convert the image to base64 for embedding in HTML
    image_base64 = image_to_base64(image)

    # Return HTML response with the image and prediction result
    return render_template_string("""
        <html>
            <body>
                <h1>Uploaded Image using Onnx and Flask API</h1>
                <img src="data:image/jpeg;base64,{{ image_base64 }}" alt="Uploaded Image" style="max-width: 300px;">
                <h2>Prediction Result:</h2>
                <p>Predicted Class: {{ predicted_class }}</p>
                <br><br>
                <a href="/">Upload another image</a>
            </body>
        </html>
    """, image_base64=image_base64, predicted_class=predicted_class)

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Use a different port
