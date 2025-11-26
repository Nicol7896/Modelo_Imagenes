from ultralytics import YOLO 
import os 
import shutil 
print("Iniciando entrenamiento...") 
model = YOLO("yolov8n.pt") 
model.train( data="Imagenes/data.yaml", 
            epochs=70, 
            imgsz=640, 
            batch=2, 
            name="yolo_entrenamiento_demo" ) 
print("Entrenamiento terminado. Revisa la carpeta: runs/detect/yolo_entrenamiento_demo")