import cv2
import os
import numpy as np

def aplicar_sombras_forzadas(ruta_origen, ruta_destino):
    if not os.path.exists(ruta_destino): os.makedirs(ruta_destino)
    archivos = [f for f in os.listdir(ruta_origen) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for nombre_archivo in archivos:
        img_path = os.path.join(ruta_origen, nombre_archivo)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue

        sombra_mod = cv2.convertScaleAbs(img, alpha=0.6, beta=0) 
        
        sombra_heavy = cv2.convertScaleAbs(img, alpha=0.25, beta=0)

        nombre_base = os.path.splitext(nombre_archivo)[0].replace(" ", "_")
        cv2.imwrite(os.path.join(ruta_destino, f"ORIG_{nombre_base}.jpg"), img)
        cv2.imwrite(os.path.join(ruta_destino, f"MOD_{nombre_base}.jpg"), sombra_mod)
        cv2.imwrite(os.path.join(ruta_destino, f"DARK_{nombre_base}.jpg"), sombra_heavy)

    print("Listop")

ORIGEN = r"C:\Users\scani\Downloads\a\1"
DESTINO = r"C:\Users\scani\Downloads\a\3"

aplicar_sombras_forzadas(ORIGEN, DESTINO)