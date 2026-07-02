import os
from PIL import Image, ImageOps

def compresion_final_full_res_baja_calidad(carpeta_origen, carpeta_destino):
    """
    - Mantiene TAMAÑO ORIGINAL.
    - Mantiene COLOR ORIGINAL.
    - CALIDAD 15 (Compresión fuerte para notar el cambio de peso).
    - Mejora contraste para no perder detalles por la compresión.
    """
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    for archivo in os.listdir(carpeta_origen):
        if archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            ruta_entrada = os.path.join(carpeta_origen, archivo)
            ruta_salida = os.path.join(carpeta_destino, archivo)

            try:
                with Image.open(ruta_entrada) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    img = ImageOps.autocontrast(img)
                    
                    img.save(ruta_salida, "JPEG", quality=15, optimize=True)
                    
                    ancho, alto = img.size
                    peso_final = os.path.getsize(ruta_salida) / 1024
                    print(f" {archivo} ({ancho}x{alto}) -> {peso_final:.1f} KB")
            except Exception as e:
                print(f"Error en {archivo}: {e}")

ORIGEN = r"C:\Users\scani\Downloads\a\2"
DESTINO = r"C:\Users\scani\Downloads\a\4"

if __name__ == "__main__":
    compresion_final_full_res_baja_calidad(ORIGEN, DESTINO)
    print("\nProceso terminado. Calidad reducida al 15% manteniendo dimensiones.")