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

# ===============================
# 1. CONFIGURACIÓN DE HARDWARE (UNIVERSAL)
# ===============================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Detectar y configurar GPU/CPU automáticamente
def configurar_hardware():
    """Configura TensorFlow para funcionar en cualquier hardware disponible"""
    
    # Verificar GPUs disponibles
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Configurar cada GPU para crecimiento de memoria
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Mostrar información de la GPU
            print(f"--- Hardware: GPU DETECTADA ({len(gpus)} GPU{'s' if len(gpus)>1 else ''}) ---")
            for i, gpu in enumerate(gpus):
                gpu_name = tf.config.experimental.get_device_details(gpu).get('device_name', 'Desconocida')
                print(f"  GPU {i+1}: {gpu_name}")
            
            # Configurar para usar todas las GPUs disponibles
            strategy = tf.distribute.MirroredStrategy()
            print(f"  Estrategia: {strategy.__class__.__name__}")
            return strategy
            
        except RuntimeError as e:
            print(f"⚠️ Error configurando GPU: {e}")
            print("--- Hardware: CPU (fallback) ---")
            return None
    else:
        print("--- Hardware: CPU DETECTADA ---")
        
        # Optimizaciones para CPU
        # Limitar threads para evitar sobrecarga
        tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count() // 2)
        tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count() // 2)
        
        print(f"  CPU Cores disponibles: {os.cpu_count()}")
        return None

# Aplicar configuración
strategy = configurar_hardware()

# ===============================
# 2. CONFIGURACIÓN GENERAL
# ===============================
TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/val"
IMG_SIZE = (224, 224)
BATCH = 32  # Reducir en CPU si es necesario
MODEL_NAME = "modelo_vgg_fatiga_universal.h5"

# Ajustar batch size para CPU si es necesario
if strategy is None:  # Estamos en CPU
    BATCH = 16  # Batch más pequeño para CPU
    print(f"  Batch size ajustado a {BATCH} para CPU")

# ===============================
# 3. GENERADORES (Manteniendo Robustez)
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
# 4. CONSTRUCCIÓN DEL MODELO
# ===============================
def construir_modelo(num_classes):
    """Construye el modelo VGG16 con las capas personalizadas"""
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False 
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    x = BatchNormalization()(x)      
    x = Dense(512, activation="relu", kernel_regularizer=l2(0.01))(x) 
    x = Dropout(0.5)(x)              
    output = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model

# Construir el modelo
num_classes = train_gen.num_classes
model, base_model = construir_modelo(num_classes)

# Si hay GPU, usar estrategia distribuida
if strategy is not None:
    with strategy.scope():
        model, base_model = construir_modelo(num_classes)
        print("  Modelo construido con estrategia distribuida")

# ===============================
# 5. CALLBACKS
# ===============================
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=8, 
    restore_best_weights=True, 
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=1e-7, 
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_NAME, 
    monitor='val_accuracy', 
    save_best_only=True, 
    verbose=1
)

# ===============================
# 6. FASE 1: ENTRENAMIENTO CON LABEL SMOOTHING
# ===============================
print("\n--- FASE 1: CLASIFICADOR CON LABEL SMOOTHING ---")
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=Adam(learning_rate=1e-4), 
    loss=loss_fn, 
    metrics=["accuracy"]
)

print(f"  Dispositivo: {'GPU' if strategy else 'CPU'}")
print(f"  Batch size: {BATCH}")
print("  Iniciando entrenamiento FASE 1...")

model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=15, 
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ===============================
# 7. FASE 2: FINE-TUNING PROFUNDO
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

print("  Iniciando entrenamiento FASE 2...")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30, 
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

print(f"\n✅ Modelo completado y guardado como: {MODEL_NAME}")

# ===============================
# 8. INFORMACIÓN FINAL
# ===============================
print("\n=== RESUMEN DE ENTRENAMIENTO ===")
print(f"Modelo: {MODEL_NAME}")
print(f"Dispositivo: {'GPU' if strategy else 'CPU'}")
print(f"Clases: {num_classes}")
print(f"Labels: {list(train_gen.class_indices.keys())}")

# Mostrar métricas finales
if history.history['val_accuracy']:
    final_val_acc = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"Precisión validación final: {final_val_acc:.4f}")
    print(f"Pérdida validación final: {final_val_loss:.4f}")

print("\n💡 El modelo está listo para usarse en cualquier máquina (CPU/GPU)")