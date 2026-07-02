import cv2
import os
import numpy as np

def aumentar_rotacion_unicode(ruta_origen, ruta_destino):
    if not os.path.exists(ruta_destino):
        os.makedirs(ruta_destino)

    extensiones = ('.jpg', '.jpeg', '.png')
    archivos = [f for f in os.listdir(ruta_origen) if f.lower().endswith(extensiones)]
    
    print(f"Procesando {len(archivos)} imágenes...")

    for nombre_archivo in archivos:
        img_path = os.path.join(ruta_origen, nombre_archivo)
        
        stream = open(img_path, "rb")
        bytes_array = bytearray(stream.read())
        numpy_array = np.asarray(bytes_array, dtype=np.uint8)
        img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        stream.close()
        
        if img is None:
            print(f"Error saltando: {nombre_archivo}")
            continue

        h, w = img.shape[:2]
        centro = (w // 2, h // 2)

        nombre_limpio = nombre_archivo.replace(" ", "_") 
        cv2.imwrite(os.path.join(ruta_destino, f"original_{nombre_limpio}"), img)

        for angulo in [15, -15]:
            M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
            img_rotada = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(os.path.join(ruta_destino, f"rot{angulo}_{nombre_limpio}"), img_rotada)

    print(f"Todo guardado en: {ruta_destino}")

aumentar_rotacion_unicode(r"C:\Users\scani\Downloads\a\1", r"C:\Users\scani\Downloads\a\3")