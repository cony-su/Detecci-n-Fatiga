import os
import cv2

def reflejar_imagenes(carpeta_entrada, carpeta_salida):

    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    extensiones = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for archivo in os.listdir(carpeta_entrada):
        if archivo.lower().endswith(extensiones):
            ruta_entrada = os.path.join(carpeta_entrada, archivo)
            imagen = cv2.imread(ruta_entrada)

            if imagen is None:
                print(f"No se pudo leer: {archivo}")
                continue

            imagen_reflejada = cv2.flip(imagen, 1)

            nombre_salida = "flip_" + archivo
            ruta_salida = os.path.join(carpeta_salida, nombre_salida)

            cv2.imwrite(ruta_salida, imagen_reflejada)
            print(f"Guardada: {nombre_salida}")

    print("Proceso terminado")


if __name__ == "__main__":
    carpeta_entrada = r"C:\Users\scani\Downloads\a\2"
    carpeta_salida = r"C:\Users\scani\Downloads\a\4"

    reflejar_imagenes(carpeta_entrada, carpeta_salida)
