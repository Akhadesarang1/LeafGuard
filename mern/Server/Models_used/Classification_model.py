"""
This script implements a multi-class image classification pipeline using transfer learning.
It uses a pretrained CNN (ResNet50 / EfficientNet) with a custom classification head.

Key features:
- Automatic removal of corrupted images
- Efficient tf.data pipeline for fast training
- Strong data augmentation and MixUp for better generalization
- Two-phase training: frozen backbone followed by fine-tuning
- Mixed precision training for improved GPU performance
- Evaluation using accuracy, precision, recall, and confusion matrix

The model is trained on a directory-based dataset and saved for deployment.
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

from sklearn.metrics import classification_report, confusion_matrix

import keras_tuner as kt

# Enable mixed precision training for improved performance (if supported)
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Allow loading truncated images (useful for slightly corrupted files)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------------------------------------
# Dataset Directories
# ----------------------------------------------------
train_dir = r'D:\new model\training_set'
test_dir  = r'D:\new model\test_set'

# ----------------------------------------------------
# Clean Directories: Remove Corrupt Images
# ----------------------------------------------------
def clean_directory(directory):
    print(f"Cleaning directory: {directory}")
    removed_files = 0
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                print(f"Removing invalid image: {file_path} | Error: {e}")
                os.remove(file_path)
                removed_files += 1
    print(f"Removed {removed_files} invalid files from {directory}.")

clean_directory(train_dir)
clean_directory(test_dir)

# ----------------------------------------------------
# Create Test Data Generator
# ----------------------------------------------------
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)
num_classes = len(test_data.class_indices)
print(f"Number of classes: {num_classes}")

# ----------------------------------------------------
# Data Augmentation Functions
# ----------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 64

def load_and_preprocess_image(file_path, label):
    """Read an image file, decode, resize, normalize, and convert label to one-hot."""
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, num_classes)
    return image, label

def augment(image, label):
    """Apply random augmentations."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    # Simulate zoom: random crop then resize back
    crop_size = tf.random.uniform([], int(IMG_SIZE[0] * 0.8), IMG_SIZE[0], dtype=tf.int32)
    image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
    image = tf.image.resize(image, IMG_SIZE)
    return image, label

def mixup(batch_images, batch_labels, alpha=0.2):
    """Apply mixup augmentation on a batch."""
    batch_size = tf.shape(batch_images)[0]
    lam = tf.random.uniform([batch_size], 0, 1)
    lam_x = tf.reshape(lam, (batch_size, 1, 1, 1))
    lam_y = tf.reshape(lam, (batch_size, 1))
    index = tf.random.shuffle(tf.range(batch_size))
    mixed_images = batch_images * lam_x + tf.gather(batch_images, index) * (1 - lam_x)
    mixed_labels = batch_labels * lam_y + tf.gather(batch_labels, index) * (1 - lam_y)
    return mixed_images, mixed_labels

def create_dataset(file_paths, labels, training=False, use_mixup=False):
    """Creates a tf.data.Dataset with caching, batching, and (optionally) augmentation."""
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if training:
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch first to ensure full batches, then repeat
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    if training and use_mixup:
        dataset = dataset.map(lambda x, y: mixup(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Repeat the dataset indefinitely after batching
    if training:
        dataset = dataset.repeat()
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ----------------------------------------------------
# Prepare Training Dataset
# ----------------------------------------------------
# Get file paths and labels from the training directory
temp_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
temp_generator = temp_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)
file_paths = [os.path.join(train_dir, fname) for fname in temp_generator.filenames]
labels = np.array(temp_generator.classes)

# Create the training dataset; note the use of .repeat() AFTER batching for infinite data
train_dataset = create_dataset(file_paths, labels, training=True, use_mixup=True)
# Compute steps per epoch based on your dataset size and batch size
steps_per_epoch = len(file_paths) // BATCH_SIZE
print(f"Steps per epoch: {steps_per_epoch}")

# ----------------------------------------------------
# Model Definition with Fine-Tuning
# ----------------------------------------------------
def build_model(hp, freeze_base=True):
    backbone_choice = hp.Choice('backbone', values=['ResNet50', 'EfficientNetB0'], default='ResNet50')
    if backbone_choice == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    if freeze_base:
        base_model.trainable = False
    else:
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        for layer in base_model.layers[-30:]:
            layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    dense_units = hp.Int('units', min_value=128, max_value=512, step=64, default=256)
    l2_rate = hp.Float('l2_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-2)
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_rate))(x)
    x = BatchNormalization()(x)
    dropout_rate = hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1, default=0.5)
    x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_rate))(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    # Attach the base model so we can reference it later during fine-tuning
    model.base_model = base_model
    
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG', default=1e-4)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss_fn,
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

# ----------------------------------------------------
# Training with Two-Phase Fine-Tuning
# ----------------------------------------------------
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    hp = kt.HyperParameters()
    hp.Fixed('units', 256)
    hp.Fixed('dropout_rate', 0.5)
    hp.Fixed('l2_rate', 1e-2)
    hp.Fixed('learning_rate', 1e-4)
    hp.Fixed('backbone', 'ResNet50')
    model_ft = build_model(hp, freeze_base=True)

phase1_callbacks = [
    EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=1e-6),
    TensorBoard(log_dir='logs/phase1', histogram_freq=1)
]

print("Starting Phase 1: Training the classification head (base frozen)")
model_ft.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    callbacks=phase1_callbacks,
    verbose=1
)

# ----------------------------------------------------
# Phase 2: Fine-tuning by unfreezing the last 30 layers
# ----------------------------------------------------
print("Starting Phase 2: Fine-tuning the top layers of the base model")
model_ft.base_model.trainable = True
for layer in model_ft.base_model.layers[:-30]:
    layer.trainable = False
for layer in model_ft.base_model.layers[-30:]:
    layer.trainable = True

# Re-compile the model within the same strategy scope for Phase 2
with strategy.scope():
    model_ft.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                     loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                     metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

phase2_callbacks = [
    EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=1e-6),
    TensorBoard(log_dir='logs/phase2', histogram_freq=1)
]

model_ft.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    callbacks=phase2_callbacks,
    verbose=1
)

# Save the fine-tuned model
model_ft.save(r'D:\pp6v5_finetuned.keras')

# ----------------------------------------------------
# Evaluate Model on Test Set
# ----------------------------------------------------
test_loss, test_acc, test_precision, test_recall = model_ft.evaluate(test_data, verbose=1)
print(f"Test Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")

y_pred_probs = model_ft.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_data.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
