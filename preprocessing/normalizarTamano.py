import cv2
import os
import numpy as np

def redimensionar_dataset(ruta_origen, ruta_destino, tamano=(224, 224)):
    if not os.path.exists(ruta_destino):
        os.makedirs(ruta_destino)

    extensiones = ('.jpg', '.jpeg', '.png')
    archivos = [f for f in os.listdir(ruta_origen) if f.lower().endswith(extensiones)]
    
    print(f"Redimensionando {len(archivos)} imágenes a {tamano[0]}x{tamano[1]}...")

    for nombre_archivo in archivos:
        img_path = os.path.join(ruta_origen, nombre_archivo)
        
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            continue

        img_resizada = cv2.resize(img, tamano, interpolation=cv2.INTER_AREA)

        nombre_base = os.path.splitext(nombre_archivo)[0].replace(" ", "_")
        cv2.imwrite(os.path.join(ruta_destino, f"resized_{nombre_base}.jpg"), img_resizada)

    print(f"Ahora, todas las imágenes ahora son de {tamano[0]}x{tamano[1]} en: {ruta_destino}")

ORIGEN = r"C:\Users\scani\Downloads\a\4"
DESTINO = r"C:\Users\scani\Downloads\a\2"

redimensionar_dataset(ORIGEN, DESTINO)