

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import io
import base64

# Initialize FastAPI app
app = FastAPI()

# Load the ONNX model
onnx_model_path = "resnet18.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Preprocess function (same as Flask)
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes))
    input_tensor = transform(image).unsqueeze(0)
    return image, input_tensor.numpy()  # Return both image and numpy array

# Convert image to base64 (same as Flask)
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Serve HTML at root (FastAPI uses Jinja2 for templating)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return """
    <html>
        <body>
            <h1>Upload an image for Prediction with Onnx and Fast API</h1>
            <form action="/predict" method="POST" enctype="multipart/form-data">
                <input type="file" name="file">
                <button type="submit">Submit</button>
            </form>
        </body>
    </html>
    """

# Prediction endpoint (FastAPI handles file uploads differently)
@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    if not file:
        return {"error": "No file uploaded"}, 400

    # Read image bytes
    image_bytes = await file.read()
    image, input_numpy = preprocess_image(image_bytes)

    # Run inference
    inputs = {ort_session.get_inputs()[0].name: input_numpy}
    outputs = ort_session.run(None, inputs)

    # Get predicted class
    predicted_class = np.argmax(outputs[0])

    # Convert image to base64
    image_base64 = image_to_base64(image)

    # Return HTML response
    return f"""
    <html>
        <body>
            <h1>Uploaded Image</h1>
            <img src="data:image/jpeg;base64,{image_base64}" alt="Uploaded Image" style="max-width: 300px;">
            <h2>Prediction Result:</h2>
            <p>Predicted Class: {predicted_class}</p>
            <br><br>
            <a href="/">Upload another image</a>
        </body>
    </html>
    """

# Run with Uvicorn (FastAPI's recommended ASGI server)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)