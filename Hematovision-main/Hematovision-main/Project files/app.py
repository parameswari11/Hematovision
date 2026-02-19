import os
import hashlib
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Flask app setup
app = Flask(__name__, template_folder='templates', static_folder='static')

# Ensure upload directory exists inside static/images
UPLOAD_DIR = os.path.join(app.static_folder, 'images')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Model loading with fallback to a mock model
MODEL_PATH = os.path.join(app.template_folder, 'blood_cell_classifier_mobilenetv2 (1).h5')

CLASS_NAMES = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
IMG_SIZE = (224, 224)


class MockModel:
    """Deterministic mock model to allow the app to run without a real trained model."""
    def predict(self, x):
        # x is a numpy array of shape (batch, 224, 224, 3) with floats in [0, 1]
        batch = x.shape[0]
        out = np.zeros((batch, len(CLASS_NAMES)), dtype=np.float32)
        for i in range(batch):
            sample = (x[i] * 255).astype(np.uint8)
            # Create a deterministic seed from the content
            h = hashlib.sha1(sample.tobytes()).digest()
            seed = int.from_bytes(h[:4], 'little')
            rng = np.random.default_rng(seed)
            logits = rng.normal(size=(len(CLASS_NAMES),))
            # softmax
            exp = np.exp(logits - np.max(logits))
            out[i] = exp / np.sum(exp)
        return out


# Try to load the real model; fall back to mock on any error
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"Loaded model from: {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"WARNING: Using MockModel because loading failed: {e}")
    model = MockModel()


def preprocess_image(image_path):
    """Load and preprocess image to model input tensor."""
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # rescale like typical training
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    # Save file
    safe_name = file.filename.replace(' ', '_')
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(save_path)

    # Preprocess and predict
    input_tensor = preprocess_image(save_path)
    preds = model.predict(input_tensor)
    pred_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))

    # Map to class name (fallback if out of range)
    if 0 <= pred_idx < len(CLASS_NAMES):
        pred_class = CLASS_NAMES[pred_idx]
    else:
        pred_class = f"Class_{pred_idx}"

    return render_template(
        'result.html',
        prediction=pred_class,
        confidence=f"{confidence*100:.2f}%",
        image_file=safe_name
    )


if __name__ == '__main__':
    # Run the Flask dev server
    app.run(host='127.0.0.1', port=5000, debug=True)
