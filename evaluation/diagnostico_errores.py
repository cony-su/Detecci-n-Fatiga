import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

# Configuración
MODELO_GANADOR = "modelo_vgg_fatiga.h5"
DIR_VAL = r"dataset/val"
DIR_ERRORES = "errores_detectados"

if os.path.exists(DIR_ERRORES): shutil.rmtree(DIR_ERRORES)
os.makedirs(DIR_ERRORES)

# Cargar modelo y datos
model = tf.keras.models.load_model(MODELO_GANADOR)
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
gen = datagen.flow_from_directory(DIR_VAL, target_size=(224,224), batch_size=1, shuffle=False)


preds = model.predict(gen)
indices = gen.class_indices
inv_indices = {v: k for k, v in indices.items()}

print("\n--- Analizando errores ---")
for i, p in enumerate(preds):
    pred_idx = np.argmax(p)
    true_idx = gen.classes[i]
    if pred_idx != true_idx:
        ruta_orig = gen.filepaths[i]
        nombre_foto = os.path.basename(ruta_orig)
        cat_real = inv_indices[true_idx]
        cat_pred = inv_indices[pred_idx]
        
        nuevo_nombre = f"REAL_{cat_real}_DICE_{cat_pred}_{nombre_foto}"
        shutil.copy(ruta_orig, os.path.join(DIR_ERRORES, nuevo_nombre))
        print(f"Error capturado: {nuevo_nombre}")