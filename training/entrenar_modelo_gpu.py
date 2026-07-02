import json
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# 1. Configuración de Hardware
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("--- Hardware: GPU ACTIVADA ---")
    except RuntimeError as e: print(e)

# CONFIGURACIÓN GENERAL
TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/val"
IMG_SIZE = (224, 224)
BATCH = 32 
#anteriormente: MODEL_NAME = "modelo_vgg_fatiga_9_6.h5" # <--- VERSIÓN 9_4
MODEL_NAME = "modelo_vgg_fatiga.h5" # <--- VERSIÓN 9_4

# ===============================
# 2. GENERADORES (Manteniendo Robustez)
# ===============================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=False,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, 
    target_size=IMG_SIZE, 
    batch_size=BATCH, 
    class_mode="categorical",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR, 
    target_size=IMG_SIZE, 
    batch_size=BATCH, 
    class_mode="categorical"
)

with open("labels.json","w") as f:
    json.dump({"class_names": list(train_gen.class_indices.keys())}, f, indent=2)

# ===============================
# 3. CONSTRUCCIÓN DEL MODELO
# ===============================
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x) 
x = BatchNormalization()(x)      
x = Dense(512, activation="relu", kernel_regularizer=l2(0.01))(x) 
x = Dropout(0.5)(x)              
output = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# CALLBACKS
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', save_best_only=True, verbose=1)

# ===============================
# 4. FASE 1: ENTRENAMIENTO CON LABEL SMOOTHING
# ===============================
print("\n--- FASE 1: CLASIFICADOR CON LABEL SMOOTHING ---")
# El Label Smoothing ayuda a que el modelo no sea tan "agresivo" con las predicciones
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss=loss_fn, 
              metrics=["accuracy"])

model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=15, 
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ===============================
# 5. FASE 2: FINE-TUNING PROFUNDO
# ===============================
print("\n--- FASE 2: FINE-TUNING PROFUNDO ---")
base_model.trainable = True

# Mantenemos los bloques 4 y 5 abiertos (capas finales)
for layer in base_model.layers[:-8]:
    layer.trainable = False

# Re-compilar manteniendo el Label Smoothing
model.compile(
    optimizer=Adam(learning_rate=1e-6), 
    loss=loss_fn, 
    metrics=["accuracy"]
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30, 
    callbacks=[early_stop, reduce_lr, checkpoint]
)

print(f"\n✅ Modelo completado: {MODEL_NAME}")