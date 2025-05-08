from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)  # Habilitar CORS para todos los dominios

model = load_model("modelo/modelo.h5")
frase_actual = ""

@app.route("/")
def index():
    return send_from_directory('static', 'index.html')

@app.route("/predict", methods=["POST"])
def predict():
    global frase_actual
    if "image" not in request.files:
        return jsonify({"error": "No se proporcionó imagen"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "No se pudo leer la imagen"}), 400

    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)[0]
    confidence = np.max(predictions)
    idx = np.argmax(predictions)
    letra = chr(ord('A') + idx)

    response_data = {
        "letra": letra,
        "frase": frase_actual,
        "confidence": float(confidence),
        "status": "success" if confidence > 0.3 else "low_confidence"
    }

    #if confidence > 0.5 and (not frase_actual or letra != frase_actual[-1]):

    if confidence > 0.5:
        frase_actual += letra
        print(f"\U0001F4AC Letra predicha: {letra} (Confianza: {confidence:.2f})")
        print(f"\U0001F4DD Frase actual: {frase_actual}")
    else:
        print(f"\u26A0\ufe0f Confianza baja ({confidence:.2f}). Letra ignorada: {letra}")

    return jsonify(response_data)

@app.route("/reset", methods=["POST"])
def reset_frase():
    global frase_actual
    frase_actual = ""
    print("\U0001F504 Frase reiniciada")
    return jsonify({"status": "ok", "frase": frase_actual})

@app.route("/frase", methods=["GET"])
def obtener_frase():
    return jsonify({"frase": frase_actual})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')