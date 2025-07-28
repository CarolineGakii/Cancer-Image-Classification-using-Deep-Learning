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

# ✅ Model builder using DenseNet121
def build_model(img_shape=(IMG_SIZE, IMG_SIZE, 3)):
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        input_shape=img_shape,
        weights="imagenet",
        pooling='avg'
    )
    base_model.trainable = False  # Freeze base model

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
model.save("final_modeldense.keras")
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
    WARNING:tensorflow:From C:\Users\Big Data Team\anaconda4\envs\python11\Lib\site-packages\keras\src\backend.py:1398: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.
    
    WARNING:tensorflow:From C:\Users\Big Data Team\anaconda4\envs\python11\Lib\site-packages\keras\src\layers\normalization\batch_normalization.py:979: The name tf.nn.fused_batch_norm is deprecated. Please use tf.compat.v1.nn.fused_batch_norm instead.
    
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5
    29084464/29084464 [==============================] - 4s 0us/step
    WARNING:tensorflow:From C:\Users\Big Data Team\anaconda4\envs\python11\Lib\site-packages\keras\src\optimizers\__init__.py:309: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_2 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                     
     densenet121 (Functional)    (None, 1024)              7037504   
                                                                     
     dropout (Dropout)           (None, 1024)              0         
                                                                     
     dense (Dense)               (None, 1)                 1025      
                                                                     
    =================================================================
    Total params: 7038529 (26.85 MB)
    Trainable params: 1025 (4.00 KB)
    Non-trainable params: 7037504 (26.85 MB)
    _________________________________________________________________
    Epoch 1/30
    WARNING:tensorflow:From C:\Users\Big Data Team\anaconda4\envs\python11\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.
    
    WARNING:tensorflow:From C:\Users\Big Data Team\anaconda4\envs\python11\Lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.
    
    195/195 [==============================] - 201s 1000ms/step - loss: 0.9810 - accuracy: 0.6587 - val_loss: 0.5419 - val_accuracy: 0.7291
    Epoch 2/30
    195/195 [==============================] - 193s 989ms/step - loss: 0.6213 - accuracy: 0.7180 - val_loss: 0.4957 - val_accuracy: 0.7677
    Epoch 3/30
    195/195 [==============================] - 192s 985ms/step - loss: 0.5328 - accuracy: 0.7479 - val_loss: 0.4649 - val_accuracy: 0.7909
    Epoch 4/30
    195/195 [==============================] - 193s 990ms/step - loss: 0.5092 - accuracy: 0.7587 - val_loss: 0.4799 - val_accuracy: 0.7677
    Epoch 5/30
    195/195 [==============================] - 193s 990ms/step - loss: 0.4824 - accuracy: 0.7670 - val_loss: 0.4471 - val_accuracy: 0.7967
    Epoch 6/30
    195/195 [==============================] - 190s 975ms/step - loss: 0.4778 - accuracy: 0.7706 - val_loss: 0.4478 - val_accuracy: 0.7947
    Epoch 7/30
    195/195 [==============================] - 189s 971ms/step - loss: 0.4839 - accuracy: 0.7666 - val_loss: 0.4607 - val_accuracy: 0.7838
    Epoch 8/30
    195/195 [==============================] - 191s 980ms/step - loss: 0.4723 - accuracy: 0.7752 - val_loss: 0.4750 - val_accuracy: 0.7780
    Epoch 9/30
    195/195 [==============================] - 191s 978ms/step - loss: 0.4619 - accuracy: 0.7825 - val_loss: 0.4691 - val_accuracy: 0.7754
    Epoch 10/30
    195/195 [==============================] - ETA: 0s - loss: 0.4702 - accuracy: 0.7752📦 Saved model at epoch 10 to 'model_epoch_10.keras'
    195/195 [==============================] - 192s 984ms/step - loss: 0.4702 - accuracy: 0.7752 - val_loss: 0.4501 - val_accuracy: 0.7941
    Epoch 11/30
    195/195 [==============================] - 190s 972ms/step - loss: 0.4711 - accuracy: 0.7760 - val_loss: 0.4481 - val_accuracy: 0.7986
    Epoch 12/30
    195/195 [==============================] - 192s 986ms/step - loss: 0.4721 - accuracy: 0.7850 - val_loss: 0.4535 - val_accuracy: 0.7876
    Epoch 13/30
    195/195 [==============================] - 192s 983ms/step - loss: 0.4749 - accuracy: 0.7735 - val_loss: 0.4506 - val_accuracy: 0.7909
    Epoch 14/30
    195/195 [==============================] - 198s 1s/step - loss: 0.4702 - accuracy: 0.7748 - val_loss: 0.4443 - val_accuracy: 0.7992
    Epoch 15/30
    195/195 [==============================] - ETA: 0s - loss: 0.4739 - accuracy: 0.7765📦 Saved model at epoch 15 to 'model_epoch_15.keras'
    195/195 [==============================] - 197s 1s/step - loss: 0.4739 - accuracy: 0.7765 - val_loss: 0.4503 - val_accuracy: 0.7941
    Epoch 16/30
    195/195 [==============================] - 202s 1s/step - loss: 0.4560 - accuracy: 0.7871 - val_loss: 0.4445 - val_accuracy: 0.7967
    Epoch 17/30
    195/195 [==============================] - 202s 1s/step - loss: 0.4694 - accuracy: 0.7809 - val_loss: 0.4433 - val_accuracy: 0.7954
    Epoch 18/30
    195/195 [==============================] - 205s 1s/step - loss: 0.4679 - accuracy: 0.7828 - val_loss: 0.4377 - val_accuracy: 0.7986
    Epoch 19/30
    195/195 [==============================] - ETA: 0s - loss: 0.4636 - accuracy: 0.7770📦 Saved model at epoch 19 to 'model_epoch_19.keras'
    195/195 [==============================] - 196s 1s/step - loss: 0.4636 - accuracy: 0.7770 - val_loss: 0.4455 - val_accuracy: 0.7960
    Epoch 20/30
    195/195 [==============================] - ETA: 0s - loss: 0.4730 - accuracy: 0.7757📦 Saved model at epoch 20 to 'model_epoch_20.keras'
    195/195 [==============================] - 191s 980ms/step - loss: 0.4730 - accuracy: 0.7757 - val_loss: 0.5065 - val_accuracy: 0.7420
    Epoch 21/30
    195/195 [==============================] - 189s 968ms/step - loss: 0.4609 - accuracy: 0.7809 - val_loss: 0.4387 - val_accuracy: 0.8024
    Epoch 22/30
    195/195 [==============================] - 190s 974ms/step - loss: 0.4675 - accuracy: 0.7852 - val_loss: 0.4436 - val_accuracy: 0.8037
    Epoch 23/30
    195/195 [==============================] - 190s 973ms/step - loss: 0.4680 - accuracy: 0.7773 - val_loss: 0.5075 - val_accuracy: 0.7400
    Epoch 24/30
    195/195 [==============================] - 189s 969ms/step - loss: 0.4609 - accuracy: 0.7830 - val_loss: 0.4518 - val_accuracy: 0.7947
    Epoch 25/30
    195/195 [==============================] - ETA: 0s - loss: 0.4739 - accuracy: 0.7754📦 Saved model at epoch 25 to 'model_epoch_25.keras'
    195/195 [==============================] - 192s 984ms/step - loss: 0.4739 - accuracy: 0.7754 - val_loss: 0.4614 - val_accuracy: 0.7806
    Epoch 26/30
    195/195 [==============================] - 190s 972ms/step - loss: 0.4700 - accuracy: 0.7791 - val_loss: 0.4478 - val_accuracy: 0.7941
    Epoch 27/30
    195/195 [==============================] - 190s 977ms/step - loss: 0.4661 - accuracy: 0.7756 - val_loss: 0.4428 - val_accuracy: 0.7979
    Epoch 28/30
    195/195 [==============================] - ETA: 0s - loss: 0.4677 - accuracy: 0.7788📦 Saved model at epoch 28 to 'model_epoch_28.keras'
    195/195 [==============================] - 191s 981ms/step - loss: 0.4677 - accuracy: 0.7788 - val_loss: 0.4450 - val_accuracy: 0.7947
    Epoch 29/30
    195/195 [==============================] - 189s 970ms/step - loss: 0.4533 - accuracy: 0.7852 - val_loss: 0.4518 - val_accuracy: 0.7915
    Epoch 30/30
    195/195 [==============================] - 188s 966ms/step - loss: 0.4642 - accuracy: 0.7830 - val_loss: 0.5361 - val_accuracy: 0.7130
    ✅ Final model saved as 'final_model.keras'
    


![png](output_0_1.png)


    49/49 [==============================] - 39s 784ms/step - loss: 0.5361 - accuracy: 0.7130
    
    ✅ Validation Accuracy: 0.7130
    1/1 [==============================] - 3s 3s/step
    1/1 [==============================] - 1s 808ms/step
    1/1 [==============================] - 1s 762ms/step
    1/1 [==============================] - 1s 789ms/step
    1/1 [==============================] - 1s 878ms/step
    1/1 [==============================] - 1s 795ms/step
    1/1 [==============================] - 1s 790ms/step
    1/1 [==============================] - 1s 756ms/step
    1/1 [==============================] - 1s 780ms/step
    1/1 [==============================] - 1s 785ms/step
    1/1 [==============================] - 1s 766ms/step
    1/1 [==============================] - 1s 779ms/step
    1/1 [==============================] - 1s 772ms/step
    1/1 [==============================] - 1s 787ms/step
    1/1 [==============================] - 1s 775ms/step
    1/1 [==============================] - 1s 777ms/step
    1/1 [==============================] - 1s 786ms/step
    1/1 [==============================] - 1s 750ms/step
    1/1 [==============================] - 1s 936ms/step
    1/1 [==============================] - 1s 803ms/step
    1/1 [==============================] - 1s 769ms/step
    1/1 [==============================] - 1s 769ms/step
    1/1 [==============================] - 1s 776ms/step
    1/1 [==============================] - 1s 777ms/step
    1/1 [==============================] - 1s 775ms/step
    1/1 [==============================] - 1s 834ms/step
    1/1 [==============================] - 1s 820ms/step
    1/1 [==============================] - 1s 763ms/step
    1/1 [==============================] - 1s 894ms/step
    1/1 [==============================] - 1s 811ms/step
    1/1 [==============================] - 1s 872ms/step
    1/1 [==============================] - 1s 832ms/step
    1/1 [==============================] - 1s 779ms/step
    1/1 [==============================] - 1s 775ms/step
    1/1 [==============================] - 1s 774ms/step
    1/1 [==============================] - 1s 829ms/step
    1/1 [==============================] - 1s 830ms/step
    1/1 [==============================] - 1s 759ms/step
    1/1 [==============================] - 1s 786ms/step
    1/1 [==============================] - 1s 851ms/step
    1/1 [==============================] - 1s 782ms/step
    1/1 [==============================] - 1s 784ms/step
    1/1 [==============================] - 1s 774ms/step
    1/1 [==============================] - 1s 832ms/step
    1/1 [==============================] - 1s 777ms/step
    1/1 [==============================] - 1s 779ms/step
    1/1 [==============================] - 1s 784ms/step
    1/1 [==============================] - 1s 781ms/step
    1/1 [==============================] - 2s 2s/step
    
    📈 Sensitivity (Recall for Malignant): 0.5281
    📉 Specificity (True Negative Rate for Benign): 0.9003
    
    🔎 Classification Report:
                   precision    recall  f1-score   support
    
          benign       0.65      0.90      0.76       772
       malignant       0.84      0.53      0.65       782
    
        accuracy                           0.71      1554
       macro avg       0.75      0.71      0.70      1554
    weighted avg       0.75      0.71      0.70      1554
    
    


![png](output_0_3.png)



```python

```
