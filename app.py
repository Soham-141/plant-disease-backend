
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("best_plant_leaf_disease_model.h5")

class_labels = [
    "Apple__Apple_scab", "Apple__Black_rot", "Apple__Cedar_apple_rust", "Apple__Apple_healthy",
    "Grape__Black_rot", "Grape__Esca_(Black_Measles)", "Grape__healthy", "Grape__Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Potato__Early_blight", "Potato__healthy", "Potato__Late_blight", "Unknown"
]

cure_recommendations = {
    "Apple__Apple_scab": "Use resistant varieties. Apply fungicide (Captan, Lime Sulfur).",
    "Apple__Black_rot": "Apply fungicide (Captan, Mancozeb). Remove infected leaves.",
    "Apple__Cedar_apple_rust": "Remove nearby cedar trees. Use fungicide (Myclobutanil, Sulfur).",
    "Apple__Apple_healthy": "No action needed. Maintain proper watering & sunlight.",
    "Grape__Black_rot": "Prune infected leaves. Apply fungicide (Copper, Mancozeb).",
    "Grape__Esca_(Black_Measles)": "No cure available. Remove infected vines.",
    "Grape__healthy": "No action needed. Ensure proper vineyard management.",
    "Grape__Leaf_blight_(Isariopsis_Leaf_Spot)": "Use fungicides (Mancozeb, Copper). Ensure proper air circulation.",
    "Potato__Early_blight": "Apply fungicide (Chlorothalonil, Copper). Remove affected leaves.",
    "Potato__healthy": "No action needed. Ensure proper soil conditions.",
    "Potato__Late_blight": "Use resistant varieties. Apply fungicide (Metalaxyl, Copper).",
    "Unknown": "The disease is not recognized. Consult an expert for further analysis."
}

def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    preprocessed = preprocess_image(img)
    prediction = model.predict(preprocessed)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    class_name = class_labels[class_index]
    recommendation = cure_recommendations[class_name]

    return jsonify({
        "class": class_name,
        "confidence": round(confidence * 100, 2),
        "cure": recommendation
    })

if __name__ == '__main__':
    app.run(debug=True)