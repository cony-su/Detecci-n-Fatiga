import os
import random
import string

DIR_PATH = r"C:\Users\scani\Downloads\a\4"

def generar_codigo_azar(longitud=8):
    """Genera una cadena aleatoria tipo 'a7fb2k92'"""
    caracteres = string.ascii_lowercase + string.digits
    return ''.join(random.choice(caracteres) for _ in range(longitud))

def renombrar_caotico(ruta):
    archivos = [f for f in os.listdir(ruta) if os.path.isfile(os.path.join(ruta, f))]
    
    if not archivos:
        print("No hay archivos en la carpeta.")
        return

    print(f"Renombrando {len(archivos)} archivos con nombres aleatorios no consecutivos...")

    for nombre_original in archivos:
        ext = os.path.splitext(nombre_original)[1]
        
        nuevo_nombre = generar_codigo_azar(10) + ext
        
        antigua_ruta = os.path.join(ruta, nombre_original)
        nueva_ruta = os.path.join(ruta, nuevo_nombre)
        
        while os.path.exists(nueva_ruta):
            nuevo_nombre = generar_codigo_azar(10) + ext
            nueva_ruta = os.path.join(ruta, nuevo_nombre)

        os.rename(antigua_ruta, nueva_ruta)

    print("Listo, Ahora los nombres son aleatorios y no siguen ninguna numeración.")

renombrar_caotico(DIR_PATH)