# Advanced Multi‑Class Image Classification with Transfer Learning

This repository implements a **robust, production‑ready multi‑class image classification pipeline** using **TensorFlow / Keras**. The system is designed for high accuracy, strong generalization, and real‑world deployment by combining **transfer learning, advanced data augmentation, mixup, and two‑phase fine‑tuning**.

---

## 📌 Key Highlights

* Multi‑class image classification (single‑label)
* Transfer learning using **ResNet50 / EfficientNetB0**
* Advanced `tf.data` input pipeline
* Strong data augmentation + **MixUp**
* Two‑phase training (frozen base → fine‑tuning)
* Mixed precision training for speed and efficiency
* Precision, Recall, Confusion Matrix & Classification Report
* Scalable and GPU‑optimized

---

## 📂 Dataset Structure

The dataset must follow a **directory‑based class structure**:

```
training_set/
 ├── class_1/
 ├── class_2/
 ├── class_3/

test_set/
 ├── class_1/
 ├── class_2/
 ├── class_3/
```

Each sub‑folder name represents a **class label**. The number of classes is detected automatically.

---

## ⚙️ Environment & Performance Optimization

### Mixed Precision Training

```python
set_global_policy('mixed_float16')
```

* Uses both **float16 and float32** automatically
* Reduces GPU memory usage
* Significantly speeds up training on modern GPUs

---

## 🧹 Dataset Cleaning (Corrupted Image Removal)

Before training, all images are verified and corrupted files are removed.

### Why this is important

* Prevents training crashes
* Ensures dataset integrity
* Avoids silent data corruption

```python
with Image.open(file_path) as img:
    img.verify()
```

This step is applied to both **training** and **test** directories.

---

## 🧪 Test Data Generator (Evaluation Only)

```python
ImageDataGenerator(rescale=1./255)
```

* Only normalization is applied
* No augmentation (to ensure fair evaluation)
* `shuffle=False` ensures correct label alignment during evaluation

---

## 🚀 Training Data Pipeline (`tf.data`)

Instead of `ImageDataGenerator`, the training pipeline uses **TensorFlow’s `tf.data` API** for efficiency and scalability.

### 1️⃣ Image Loading & Preprocessing

```python
load_and_preprocess_image()
```

Operations:

* Read image from disk
* Decode JPEG
* Resize to `224×224`
* Normalize pixel values (0–1)
* Convert label to one‑hot encoding

---

### 2️⃣ Data Augmentation

```python
augment()
```

Applied randomly during training:

* Horizontal flip
* Random brightness
* Random contrast
* Random saturation
* Random rotation
* Random crop + resize (zoom simulation)

**Purpose:**

* Reduce overfitting
* Improve robustness to real‑world variations

---

### 3️⃣ MixUp Augmentation (Advanced)

```python
mixup()
```

MixUp blends two images and their labels:

```
image = λ·image₁ + (1−λ)·image₂
label = λ·label₁ + (1−λ)·label₂
```

**Benefits:**

* Smoother decision boundaries
* Better generalization
* Reduced model overconfidence

---

### 4️⃣ Dataset Creation

```python
create_dataset()
```

Pipeline steps:

* Load file paths & labels
* Apply preprocessing
* Shuffle (training only)
* Apply augmentation
* Batch with fixed size
* Apply MixUp (optional)
* Repeat dataset infinitely
* Prefetch for GPU efficiency

This ensures **continuous, high‑performance training**.

---

## 🧠 Model Architecture (Transfer Learning)

### Backbone Networks

* **ResNet50** (default)
* **EfficientNetB0** (optional)

Pretrained on **ImageNet** and used as feature extractors.

---

### Classification Head

```
Backbone CNN
↓
Global Average Pooling
↓
Batch Normalization
↓
Dense (ReLU + L2 Regularization)
↓
Batch Normalization
↓
Dropout
↓
Dense (Softmax Output)
```

**Design choices:**

* L2 regularization → prevents overfitting
* Dropout → improves generalization
* Softmax → multi‑class probability output

---

## 🎯 Loss Function & Metrics

```python
CategoricalCrossentropy(label_smoothing=0.1)
```

* Suitable for **multi‑class classification**
* Label smoothing stabilizes training

Metrics:

* Accuracy
* Precision
* Recall

---

## 🧩 Distributed Training

```python
tf.distribute.MirroredStrategy()
```

* Enables multi‑GPU training automatically
* Works seamlessly on single‑GPU systems

---

## 🏋️ Two‑Phase Training Strategy

### 🔹 Phase 1: Train Classification Head

* Backbone frozen
* Only top layers are trained
* Faster convergence
* Stable feature learning

Callbacks used:

* EarlyStopping
* ReduceLROnPlateau
* TensorBoard logging

---

### 🔹 Phase 2: Fine‑Tuning

* Last 30 layers of backbone unfrozen
* Lower learning rate
* Learns dataset‑specific features
* Improves final accuracy

---

## 💾 Model Saving

```python
model.save('pp6v5_finetuned.keras')
```

* Saves architecture + weights + optimizer state
* Ready for deployment or inference

---

## 📊 Model Evaluation

The model is evaluated on the **unseen test set** using:

* Test accuracy & loss
* Precision & recall
* Classification report (per class)
* Confusion matrix

This provides a **complete performance analysis**.

---

## ✅ Classification Type

* **Multi‑class** classification
* **Single‑label** per image
* Softmax output layer

---

## 🏁 Conclusion

This pipeline is designed for **high‑quality image classification projects**, suitable for:

* Final‑year academic projects
* Research experiments
* Real‑world deployment
* Production‑level deep learning systems

It combines **modern best practices** in data handling, model training, and evaluation to achieve reliable and scalable results.

---

⭐ If you find this useful, consider starring the repository!
