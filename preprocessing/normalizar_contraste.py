import cv2
import os
import numpy as np

def mejorar_contraste_y_oscurecer(img):
    """
    Aplica CLAHE para resaltar texturas y luego reduce el brillo.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img_contraste = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    hsv = cv2.cvtColor(img_contraste, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    brillo_factor = 0.4  
    v = (v * brillo_factor).astype(np.uint8)
    hsv[:,:,2] = v
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def procesar_solo_contraste(ruta_inicio, ruta_final, tamano=224):
    """
    Toma las fotos y genera la versión de alto contraste oscura.
    """
    if not os.path.exists(ruta_final):
        os.makedirs(ruta_final, exist_ok=True)

    valid_ext = ('.jpg', '.jpeg', '.png', '.webp')
    archivos = [f for f in os.listdir(ruta_inicio) if f.lower().endswith(valid_ext)]

    print(f"Generando versión de CONTRASTE para {len(archivos)} imágenes...")

    contador = 0
    for archivo in archivos:
        ruta_abs = os.path.join(ruta_inicio, archivo)
        
        # Lectura robusta
        img_array = np.fromfile(ruta_abs, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            continue

        img_res = cv2.resize(img, (tamano, tamano), interpolation=cv2.INTER_AREA)
        
        img_final = mejorar_contraste_y_oscurecer(img_res)

        nombre_base = os.path.splitext(archivo)[0]
        nombre_save = f"{nombre_base}_contraste_oscuro.jpg"
        ruta_save = os.path.join(ruta_final, nombre_save)
        
        _, img_encoded = cv2.imencode(".jpg", img_final)
        img_encoded.tofile(ruta_save)

        contador += 1

    print(f"Listo, Se guardaron {contador} imágenes con contraste mejorado en: {ruta_final}")


# RUTAS
RUTA_INPUT = r"C:\Users\scani\Downloads\a\1"
RUTA_OUTPUT = r"C:\Users\scani\Downloads\a\4"

if __name__ == "__main__":
    procesar_solo_contraste(RUTA_INPUT, RUTA_OUTPUT)