from ultralytics import YOLO

# Variable de control
ya_ejecutado = False

if not ya_ejecutado:
   # Cargar el modelo 
    model = YOLO("runs/detect/yolo_entrenamiento_demo/weights/best.pt")

    # Ruta de la imagen 
    image_path = "https://http2.mlstatic.com/D_NQ_NP_671514-MCO83210850506_042025-O.webp"

    # Ejecutar inferencia 
    results = model(image_path, conf=0.1)

    # Configurar clases 
    class_names = ['pato 🦆', 'perro 🐶', 'persona 👨']
    target_classes = [0, 1, 2]

    # Procesar resultados 
    result = results[0]
    detected_classes = set()

    for box in result.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        if class_id in target_classes and class_id not in detected_classes:
            label = class_names[class_id]
            print(f"Detectado: {label} (Confianza: {conf:.2f})")
            detected_classes.add(class_id)
    