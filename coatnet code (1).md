```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2

print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)
print("OpenCV version:", cv2.__version__)
print("Matplotlib version:", plt.__version__)
print("Seaborn version:", sns.__version__)
print("Pandas version:", pd.__version__)

```

    C:\Users\Big Data Team\anaconda4\envs\python11\Lib\site-packages\requests\__init__.py:86: RequestsDependencyWarning: Unable to find acceptable character detection dependency (chardet or charset_normalizer).
      warnings.warn(
    

    WARNING:tensorflow:From C:\Users\Big Data Team\anaconda4\envs\python11\Lib\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
    
    NumPy version: 1.24.4
    TensorFlow version: 2.15.0
    OpenCV version: 4.12.0
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[1], line 11
          9 print("TensorFlow version:", tf.__version__)
         10 print("OpenCV version:", cv2.__version__)
    ---> 11 print("Matplotlib version:", plt.__version__)
         12 print("Seaborn version:", sns.__version__)
         13 print("Pandas version:", pd.__version__)
    

    AttributeError: module 'matplotlib.pyplot' has no attribute '__version__'



```python
import numpy as np
import tensorflow as tf
import cv2
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)
print("OpenCV version:", cv2.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Seaborn version:", sns.__version__)
print("Pandas version:", pd.__version__)

```

    NumPy version: 1.24.4
    TensorFlow version: 2.15.0
    OpenCV version: 4.12.0
    Matplotlib version: 3.10.3
    Seaborn version: 0.13.2
    Pandas version: 2.3.1
    


```python
print(pd.__version__)

```

    2.3.1
    


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detected: {gpus[0].name}")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass
else:
    print("⚠️ No GPU detected! Training will be slow.")

# ✅ Dataset path
dataset_dir = "organized_gastric"
IMG_SIZE = 224
BATCH_SIZE = 32

# ✅ Load training and validation datasets from the same folder
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# ✅ Class names
class_names = train_ds.class_names
print(f"✅ Class names: {class_names} (0 → {class_names[0]}, 1 → {class_names[1]})")

# ✅ Prefetch to optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ✅ Build the model using EfficientNetV2B0 as CoAtNet-like backbone
def build_model(img_shape=(IMG_SIZE, IMG_SIZE, 3)):
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False,
        input_shape=img_shape,
        weights="imagenet",
        pooling='avg'
    )
    base_model.trainable = False  # Transfer learning

    inputs = keras.Input(shape=img_shape)
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    return model

# ✅ Compile the model
model = build_model()
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ✅ Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

# ✅ Evaluate on validation set
loss, acc = model.evaluate(val_ds)
print(f"\n✅ Validation Accuracy: {acc:.4f}")

# ✅ Make predictions
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)

# ✅ Convert probabilities to 0 or 1
y_pred_labels = [1 if p > 0.5 else 0 for p in y_pred]

# ✅ Confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred_labels)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"\n📈 Sensitivity (Recall for Malignant): {sensitivity:.4f}")
print(f"📉 Specificity (True Negative Rate for Benign): {specificity:.4f}")
print("\n🔎 Classification Report:\n", classification_report(y_true, y_pred_labels, target_names=class_names))

# ✅ Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

```

    ⚠️ No GPU detected! Training will be slow.
    Found 7774 files belonging to 2 classes.
    Using 6220 files for training.
    Found 7774 files belonging to 2 classes.
    Using 1554 files for validation.
    ✅ Class names: ['benign', 'malignant'] (0 → benign, 1 → malignant)
    WARNING:tensorflow:From C:\Users\Big Data Team\anaconda4\envs\python11\Lib\site-packages\keras\src\backend.py:1398: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.
    
    WARNING:tensorflow:From C:\Users\Big Data Team\anaconda4\envs\python11\Lib\site-packages\keras\src\layers\normalization\batch_normalization.py:979: The name tf.nn.fused_batch_norm is deprecated. Please use tf.compat.v1.nn.fused_batch_norm instead.
    
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/efficientnet_v2/efficientnetv2-b0_notop.h5
    24274472/24274472 [==============================] - 4s 0us/step
    WARNING:tensorflow:From C:\Users\Big Data Team\anaconda4\envs\python11\Lib\site-packages\keras\src\optimizers\__init__.py:309: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_2 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                     
     efficientnetv2-b0 (Functio  (None, 1280)              5919312   
     nal)                                                            
                                                                     
     dropout (Dropout)           (None, 1280)              0         
                                                                     
     dense (Dense)               (None, 1)                 1281      
                                                                     
    =================================================================
    Total params: 5920593 (22.59 MB)
    Trainable params: 1281 (5.00 KB)
    Non-trainable params: 5919312 (22.58 MB)
    _________________________________________________________________
    Epoch 1/15
    WARNING:tensorflow:From C:\Users\Big Data Team\anaconda4\envs\python11\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.
    
    WARNING:tensorflow:From C:\Users\Big Data Team\anaconda4\envs\python11\Lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.
    
    195/195 [==============================] - 75s 348ms/step - loss: 0.5015 - accuracy: 0.7621 - val_loss: 0.4210 - val_accuracy: 0.8250
    Epoch 2/15
    195/195 [==============================] - 61s 313ms/step - loss: 0.4202 - accuracy: 0.8108 - val_loss: 0.3904 - val_accuracy: 0.8378
    Epoch 3/15
    195/195 [==============================] - 62s 319ms/step - loss: 0.3972 - accuracy: 0.8225 - val_loss: 0.3766 - val_accuracy: 0.8430
    Epoch 4/15
    195/195 [==============================] - 61s 311ms/step - loss: 0.3829 - accuracy: 0.8267 - val_loss: 0.3691 - val_accuracy: 0.8462
    Epoch 5/15
    195/195 [==============================] - 61s 311ms/step - loss: 0.3719 - accuracy: 0.8346 - val_loss: 0.3690 - val_accuracy: 0.8475
    Epoch 6/15
    195/195 [==============================] - 60s 305ms/step - loss: 0.3734 - accuracy: 0.8314 - val_loss: 0.3619 - val_accuracy: 0.8514
    Epoch 7/15
    195/195 [==============================] - 61s 312ms/step - loss: 0.3694 - accuracy: 0.8312 - val_loss: 0.3623 - val_accuracy: 0.8488
    Epoch 8/15
    195/195 [==============================] - 62s 320ms/step - loss: 0.3698 - accuracy: 0.8317 - val_loss: 0.3581 - val_accuracy: 0.8539
    Epoch 9/15
    195/195 [==============================] - 62s 318ms/step - loss: 0.3674 - accuracy: 0.8338 - val_loss: 0.3581 - val_accuracy: 0.8565
    Epoch 10/15
    195/195 [==============================] - 61s 315ms/step - loss: 0.3573 - accuracy: 0.8408 - val_loss: 0.3621 - val_accuracy: 0.8468
    Epoch 11/15
    195/195 [==============================] - 63s 320ms/step - loss: 0.3572 - accuracy: 0.8367 - val_loss: 0.3540 - val_accuracy: 0.8546
    Epoch 12/15
    195/195 [==============================] - 62s 317ms/step - loss: 0.3617 - accuracy: 0.8363 - val_loss: 0.3534 - val_accuracy: 0.8552
    Epoch 13/15
    195/195 [==============================] - 61s 313ms/step - loss: 0.3588 - accuracy: 0.8360 - val_loss: 0.3536 - val_accuracy: 0.8546
    Epoch 14/15
    195/195 [==============================] - 63s 325ms/step - loss: 0.3545 - accuracy: 0.8407 - val_loss: 0.3519 - val_accuracy: 0.8507
    Epoch 15/15
    195/195 [==============================] - 60s 306ms/step - loss: 0.3519 - accuracy: 0.8421 - val_loss: 0.3510 - val_accuracy: 0.8533
    49/49 [==============================] - 12s 243ms/step - loss: 0.3510 - accuracy: 0.8533
    
    ✅ Validation Accuracy: 0.8533
    1/1 [==============================] - 2s 2s/step
    1/1 [==============================] - 0s 260ms/step
    1/1 [==============================] - 0s 272ms/step
    1/1 [==============================] - 0s 253ms/step
    1/1 [==============================] - 0s 260ms/step
    1/1 [==============================] - 0s 270ms/step
    1/1 [==============================] - 0s 240ms/step
    1/1 [==============================] - 0s 270ms/step
    1/1 [==============================] - 0s 265ms/step
    1/1 [==============================] - 0s 280ms/step
    1/1 [==============================] - 0s 266ms/step
    1/1 [==============================] - 0s 265ms/step
    1/1 [==============================] - 0s 249ms/step
    1/1 [==============================] - 0s 259ms/step
    1/1 [==============================] - 0s 253ms/step
    1/1 [==============================] - 0s 266ms/step
    1/1 [==============================] - 0s 270ms/step
    1/1 [==============================] - 0s 264ms/step
    1/1 [==============================] - 0s 294ms/step
    1/1 [==============================] - 0s 240ms/step
    1/1 [==============================] - 0s 310ms/step
    1/1 [==============================] - 0s 291ms/step
    1/1 [==============================] - 0s 277ms/step
    1/1 [==============================] - 0s 263ms/step
    1/1 [==============================] - 0s 269ms/step
    1/1 [==============================] - 0s 269ms/step
    1/1 [==============================] - 0s 267ms/step
    1/1 [==============================] - 0s 266ms/step
    1/1 [==============================] - 0s 259ms/step
    1/1 [==============================] - 0s 268ms/step
    1/1 [==============================] - 0s 267ms/step
    1/1 [==============================] - 0s 255ms/step
    1/1 [==============================] - 0s 260ms/step
    1/1 [==============================] - 0s 267ms/step
    1/1 [==============================] - 0s 258ms/step
    1/1 [==============================] - 0s 274ms/step
    1/1 [==============================] - 0s 260ms/step
    1/1 [==============================] - 0s 272ms/step
    1/1 [==============================] - 0s 271ms/step
    1/1 [==============================] - 0s 283ms/step
    1/1 [==============================] - 0s 293ms/step
    1/1 [==============================] - 0s 313ms/step
    1/1 [==============================] - 0s 255ms/step
    1/1 [==============================] - 0s 271ms/step
    1/1 [==============================] - 0s 269ms/step
    1/1 [==============================] - 0s 274ms/step
    1/1 [==============================] - 0s 265ms/step
    1/1 [==============================] - 0s 263ms/step
    1/1 [==============================] - 2s 2s/step
    
    📈 Sensitivity (Recall for Malignant): 0.9028
    📉 Specificity (True Negative Rate for Benign): 0.8031
    
    🔎 Classification Report:
                   precision    recall  f1-score   support
    
          benign       0.89      0.80      0.84       772
       malignant       0.82      0.90      0.86       782
    
        accuracy                           0.85      1554
       macro avg       0.86      0.85      0.85      1554
    weighted avg       0.86      0.85      0.85      1554
    
    


![png](output_3_1.png)



```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detected: {gpus[0].name}")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass
else:
    print("⚠️ No GPU detected! Training will be slow.")

# ✅ Dataset path and constants
dataset_dir = "organized_gastric"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
SAVE_EPOCHS = {10, 15, 19, 20, 25, 28}

# ✅ Load training & validation datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# ✅ Class names
class_names = train_ds.class_names
print(f"✅ Class names: {class_names}")

# ✅ Prefetch datasets
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ✅ Model builder
def build_model(img_shape=(IMG_SIZE, IMG_SIZE, 3)):
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False,
        input_shape=img_shape,
        weights="imagenet",
        pooling='avg'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=img_shape)
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    return model

# ✅ Compile model
model = build_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ✅ Callback to save model at selected epochs
class SaveAtEpochs(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) in SAVE_EPOCHS:
            model_path = f"model_epoch_{epoch + 1}.keras"
            self.model.save(model_path)
            print(f"📦 Saved model at epoch {epoch + 1} to '{model_path}'")

save_callback = SaveAtEpochs()

# ✅ Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[save_callback]
)

# ✅ Save final model
model.save("final_model.keras")
print("✅ Final model saved as 'final_model.keras'")

# ✅ Plot loss vs epochs
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")  # Optional: save figure
plt.show()

# ✅ Evaluate on validation set
loss, acc = model.evaluate(val_ds)
print(f"\n✅ Validation Accuracy: {acc:.4f}")

# ✅ Predictions
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)

# ✅ Threshold predictions
y_pred_labels = [1 if p > 0.5 else 0 for p in y_pred]

# ✅ Confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred_labels)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"\n📈 Sensitivity (Recall for Malignant): {sensitivity:.4f}")
print(f"📉 Specificity (True Negative Rate for Benign): {specificity:.4f}")
print("\n🔎 Classification Report:\n", classification_report(y_true, y_pred_labels, target_names=class_names))

# ✅ Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # Optional: save
plt.show()

```

    ⚠️ No GPU detected! Training will be slow.
    Found 7774 files belonging to 2 classes.
    Using 6220 files for training.
    Found 7774 files belonging to 2 classes.
    Using 1554 files for validation.
    ✅ Class names: ['benign', 'malignant']
    Model: "model_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_4 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                     
     efficientnetv2-b0 (Functio  (None, 1280)              5919312   
     nal)                                                            
                                                                     
     dropout_1 (Dropout)         (None, 1280)              0         
                                                                     
     dense_1 (Dense)             (None, 1)                 1281      
                                                                     
    =================================================================
    Total params: 5920593 (22.59 MB)
    Trainable params: 1281 (5.00 KB)
    Non-trainable params: 5919312 (22.58 MB)
    _________________________________________________________________
    Epoch 1/30
    195/195 [==============================] - 68s 316ms/step - loss: 0.5179 - accuracy: 0.7487 - val_loss: 0.4200 - val_accuracy: 0.8269
    Epoch 2/30
    195/195 [==============================] - 60s 307ms/step - loss: 0.4192 - accuracy: 0.8133 - val_loss: 0.3873 - val_accuracy: 0.8417
    Epoch 3/30
    195/195 [==============================] - 60s 306ms/step - loss: 0.3991 - accuracy: 0.8180 - val_loss: 0.3742 - val_accuracy: 0.8462
    Epoch 4/30
    195/195 [==============================] - 60s 306ms/step - loss: 0.3886 - accuracy: 0.8228 - val_loss: 0.3673 - val_accuracy: 0.8443
    Epoch 5/30
    195/195 [==============================] - 60s 309ms/step - loss: 0.3787 - accuracy: 0.8315 - val_loss: 0.3682 - val_accuracy: 0.8449
    Epoch 6/30
    195/195 [==============================] - 60s 308ms/step - loss: 0.3730 - accuracy: 0.8362 - val_loss: 0.3602 - val_accuracy: 0.8443
    Epoch 7/30
    195/195 [==============================] - 62s 317ms/step - loss: 0.3694 - accuracy: 0.8354 - val_loss: 0.3591 - val_accuracy: 0.8481
    Epoch 8/30
    195/195 [==============================] - 60s 306ms/step - loss: 0.3649 - accuracy: 0.8333 - val_loss: 0.3556 - val_accuracy: 0.8488
    Epoch 9/30
    195/195 [==============================] - 60s 305ms/step - loss: 0.3623 - accuracy: 0.8354 - val_loss: 0.3576 - val_accuracy: 0.8520
    Epoch 10/30
    195/195 [==============================] - ETA: 0s - loss: 0.3589 - accuracy: 0.8331📦 Saved model at epoch 10 to 'model_epoch_10.keras'
    195/195 [==============================] - 61s 310ms/step - loss: 0.3589 - accuracy: 0.8331 - val_loss: 0.3642 - val_accuracy: 0.8475
    Epoch 11/30
    195/195 [==============================] - 59s 303ms/step - loss: 0.3563 - accuracy: 0.8429 - val_loss: 0.3540 - val_accuracy: 0.8520
    Epoch 12/30
    195/195 [==============================] - 60s 305ms/step - loss: 0.3586 - accuracy: 0.8389 - val_loss: 0.3515 - val_accuracy: 0.8507
    Epoch 13/30
    195/195 [==============================] - 59s 305ms/step - loss: 0.3519 - accuracy: 0.8415 - val_loss: 0.3504 - val_accuracy: 0.8507
    Epoch 14/30
    195/195 [==============================] - 62s 316ms/step - loss: 0.3557 - accuracy: 0.8386 - val_loss: 0.3511 - val_accuracy: 0.8520
    Epoch 15/30
    195/195 [==============================] - ETA: 0s - loss: 0.3524 - accuracy: 0.8407📦 Saved model at epoch 15 to 'model_epoch_15.keras'
    195/195 [==============================] - 61s 314ms/step - loss: 0.3524 - accuracy: 0.8407 - val_loss: 0.3490 - val_accuracy: 0.8520
    Epoch 16/30
    195/195 [==============================] - 60s 307ms/step - loss: 0.3525 - accuracy: 0.8386 - val_loss: 0.3488 - val_accuracy: 0.8514
    Epoch 17/30
    195/195 [==============================] - 60s 307ms/step - loss: 0.3496 - accuracy: 0.8458 - val_loss: 0.3493 - val_accuracy: 0.8565
    Epoch 18/30
    195/195 [==============================] - 60s 307ms/step - loss: 0.3520 - accuracy: 0.8418 - val_loss: 0.3560 - val_accuracy: 0.8494
    Epoch 19/30
    195/195 [==============================] - ETA: 0s - loss: 0.3492 - accuracy: 0.8437📦 Saved model at epoch 19 to 'model_epoch_19.keras'
    195/195 [==============================] - 61s 312ms/step - loss: 0.3492 - accuracy: 0.8437 - val_loss: 0.3502 - val_accuracy: 0.8539
    Epoch 20/30
    195/195 [==============================] - ETA: 0s - loss: 0.3521 - accuracy: 0.8423📦 Saved model at epoch 20 to 'model_epoch_20.keras'
    195/195 [==============================] - 61s 313ms/step - loss: 0.3521 - accuracy: 0.8423 - val_loss: 0.3541 - val_accuracy: 0.8501
    Epoch 21/30
    195/195 [==============================] - 60s 307ms/step - loss: 0.3468 - accuracy: 0.8423 - val_loss: 0.3490 - val_accuracy: 0.8552
    Epoch 22/30
    195/195 [==============================] - 60s 306ms/step - loss: 0.3446 - accuracy: 0.8458 - val_loss: 0.3528 - val_accuracy: 0.8475
    Epoch 23/30
    195/195 [==============================] - 60s 306ms/step - loss: 0.3489 - accuracy: 0.8455 - val_loss: 0.3481 - val_accuracy: 0.8526
    Epoch 24/30
    195/195 [==============================] - 60s 308ms/step - loss: 0.3497 - accuracy: 0.8457 - val_loss: 0.3477 - val_accuracy: 0.8520
    Epoch 25/30
    195/195 [==============================] - ETA: 0s - loss: 0.3429 - accuracy: 0.8503📦 Saved model at epoch 25 to 'model_epoch_25.keras'
    195/195 [==============================] - 61s 312ms/step - loss: 0.3429 - accuracy: 0.8503 - val_loss: 0.3474 - val_accuracy: 0.8546
    Epoch 26/30
    195/195 [==============================] - 60s 307ms/step - loss: 0.3410 - accuracy: 0.8477 - val_loss: 0.3481 - val_accuracy: 0.8546
    Epoch 27/30
    195/195 [==============================] - 60s 308ms/step - loss: 0.3471 - accuracy: 0.8450 - val_loss: 0.3485 - val_accuracy: 0.8533
    Epoch 28/30
    195/195 [==============================] - ETA: 0s - loss: 0.3471 - accuracy: 0.8434📦 Saved model at epoch 28 to 'model_epoch_28.keras'
    195/195 [==============================] - 61s 311ms/step - loss: 0.3471 - accuracy: 0.8434 - val_loss: 0.3523 - val_accuracy: 0.8514
    Epoch 29/30
    195/195 [==============================] - 61s 310ms/step - loss: 0.3437 - accuracy: 0.8494 - val_loss: 0.3540 - val_accuracy: 0.8468
    Epoch 30/30
    195/195 [==============================] - 60s 306ms/step - loss: 0.3470 - accuracy: 0.8420 - val_loss: 0.3472 - val_accuracy: 0.8520
    ✅ Final model saved as 'final_model.keras'
    


![png](output_4_1.png)


    49/49 [==============================] - 12s 243ms/step - loss: 0.3472 - accuracy: 0.8520
    
    ✅ Validation Accuracy: 0.8520
    1/1 [==============================] - 2s 2s/step
    1/1 [==============================] - 0s 251ms/step
    1/1 [==============================] - 0s 258ms/step
    1/1 [==============================] - 0s 265ms/step
    1/1 [==============================] - 0s 266ms/step
    1/1 [==============================] - 0s 253ms/step
    1/1 [==============================] - 0s 248ms/step
    1/1 [==============================] - 0s 275ms/step
    1/1 [==============================] - 0s 271ms/step
    1/1 [==============================] - 0s 268ms/step
    1/1 [==============================] - 0s 245ms/step
    1/1 [==============================] - 0s 270ms/step
    1/1 [==============================] - 0s 266ms/step
    1/1 [==============================] - 0s 261ms/step
    1/1 [==============================] - 0s 257ms/step
    1/1 [==============================] - 0s 269ms/step
    1/1 [==============================] - 0s 264ms/step
    1/1 [==============================] - 0s 275ms/step
    1/1 [==============================] - 0s 264ms/step
    1/1 [==============================] - 0s 272ms/step
    1/1 [==============================] - 0s 268ms/step
    1/1 [==============================] - 0s 263ms/step
    1/1 [==============================] - 0s 268ms/step
    1/1 [==============================] - 0s 267ms/step
    1/1 [==============================] - 0s 267ms/step
    1/1 [==============================] - 0s 245ms/step
    1/1 [==============================] - 0s 267ms/step
    1/1 [==============================] - 0s 265ms/step
    1/1 [==============================] - 0s 269ms/step
    1/1 [==============================] - 0s 263ms/step
    1/1 [==============================] - 0s 255ms/step
    1/1 [==============================] - 0s 320ms/step
    1/1 [==============================] - 0s 306ms/step
    1/1 [==============================] - 0s 278ms/step
    1/1 [==============================] - 0s 308ms/step
    1/1 [==============================] - 0s 257ms/step
    1/1 [==============================] - 0s 283ms/step
    1/1 [==============================] - 0s 260ms/step
    1/1 [==============================] - 0s 261ms/step
    1/1 [==============================] - 0s 274ms/step
    1/1 [==============================] - 0s 265ms/step
    1/1 [==============================] - 0s 270ms/step
    1/1 [==============================] - 0s 267ms/step
    1/1 [==============================] - 0s 267ms/step
    1/1 [==============================] - 0s 276ms/step
    1/1 [==============================] - 0s 272ms/step
    1/1 [==============================] - 0s 278ms/step
    1/1 [==============================] - 0s 273ms/step
    1/1 [==============================] - 2s 2s/step
    
    📈 Sensitivity (Recall for Malignant): 0.8964
    📉 Specificity (True Negative Rate for Benign): 0.8070
    
    🔎 Classification Report:
                   precision    recall  f1-score   support
    
          benign       0.88      0.81      0.84       772
       malignant       0.82      0.90      0.86       782
    
        accuracy                           0.85      1554
       macro avg       0.85      0.85      0.85      1554
    weighted avg       0.85      0.85      0.85      1554
    
    


![png](output_4_3.png)



```python

```
