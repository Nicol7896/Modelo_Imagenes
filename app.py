from ultralytics import YOLO
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Cargar el modelo YOLO entrenado
model = YOLO("runs/detect/yolo_entrenamiento_demo/weights/best.pt")

class_names = ['Pato 🦆', 'Perro 🐶', 'Persona 👨']
target_classes = [0, 1, 2]

@app.route("/")
def home():
    return "API funcionando correctamente"

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No se envió imagen"}), 400

    file = request.files['image']
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file.filename)
    file.save(path)

    results = model(path, conf=0.1)
    result = results[0]

    detecciones = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        if class_id in target_classes:
            label = class_names[class_id]
            detecciones.append({
                "nombre": label,
                "confianza": conf
            })

    if not detecciones:
        return jsonify({
            "resultados": [],
            "mensaje": "No se detectaron ninguno de los tres objetos (pato, perro o persona)."
        })

    return jsonify({"resultados": detecciones})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
