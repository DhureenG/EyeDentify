import tensorflow as tf
import pickle
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import collections

# Ensure TensorFlow is installed
try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise ModuleNotFoundError("TensorFlow is not installed. Please install it using 'pip install tensorflow'.")

# Set mixed precision
if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Paths
train_path = "./DeepFake Dataset/Training"
val_path = "./DeepFake Dataset/Validation"
test_image_path = "./DSC_8248.jpg"

# Load EfficientNetB0 with pre-trained weights
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model initially

# Model architecture
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),  # Normalize activations
    Dense(256, activation="relu", kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

# Optimizers
optimizer_phase1 = tf.keras.optimizers.Adam(learning_rate=0.0005)
optimizer_phase2 = tf.keras.optimizers.Adam(learning_rate=5e-6)

# Compile model for first phase
model.compile(
    optimizer=optimizer_phase1, 
    loss="binary_crossentropy", 
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(224, 224), batch_size=32, class_mode="binary")
val_generator = val_datagen.flow_from_directory(
    val_path, target_size=(224, 224), batch_size=32, class_mode="binary")

# Compute class distribution
labels = train_generator.classes
counter = collections.Counter(labels)
print(f"Class distribution: {dict(counter)}")

# Compute class weights dynamically
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}  # Ensure correct class mapping
print(f"Computed class weights: {class_weight_dict}")

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=5e-6)
]

# Train with frozen base model
history = model.fit(
    train_generator, epochs=12, validation_data=val_generator,
    class_weight=class_weight_dict, callbacks=callbacks, verbose=1
)

# **Unfreeze more layers**
base_model.trainable = True
for layer in base_model.layers[:-50]:  
    layer.trainable = False

# Recompile with second optimizer
model.compile(
    optimizer=optimizer_phase2, 
    loss="binary_crossentropy", 
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Train for fine-tuning
history_finetune = model.fit(
    train_generator, epochs=12, validation_data=val_generator,
    class_weight=class_weight_dict, callbacks=callbacks, verbose=1
)

# Save model
model.save("deepfake_model_fixed.keras")

# Save model as a pickle file
with open("deepfake_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")

# Function to detect deepfake
def detect_deepfake(image_path, model_path="deepfake_model.pkl"):
    # Load model from pickle file
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    img = tf.expand_dims(img, 0)

    prediction = model.predict(img, verbose=0)[0][0]
    return prediction

# Test model
prediction = detect_deepfake(test_image_path)
print(f"Deepfake probability: {prediction:.4f}")

prediction = detect_deepfake('/Users/dhureengulati/Documents/Pictures ProBook445/Pictures/C31FA79C-1E5C-4F41-8DC7-E936901C5044_1_105_c.jpeg')
print(f"Deepfake probability: {prediction:.4f}")