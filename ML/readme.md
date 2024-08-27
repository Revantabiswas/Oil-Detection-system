# Oil Spill Detection Using ResNet50

This project aims to build a model for detecting oil spills in grayscale images using a pre-trained ResNet50 model. The model is fine-tuned to perform binary classification, identifying whether an oil spill is present in a given image.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Prediction on New Images](#prediction-on-new-images)
- [Results Visualization](#results-visualization)
- [Conclusion](#conclusion)

## Introduction

Oil spills can cause significant environmental damage, and detecting them promptly is crucial for mitigating their effects. This project utilizes a deep learning approach to automate the detection process using image data. We leverage the ResNet50 model, pre-trained on the ImageNet dataset, to recognize patterns indicative of oil spills.

## Installation

To run this project, you'll need Python along with the following packages:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV (for image processing)

You can install the required packages using the following command:

```bash
pip install tensorflow keras numpy matplotlib opencv-python
```

## Data Preparation

Preprocessed training and testing datasets are stored as NumPy arrays. They can be loaded using the following code:

```python
import numpy as np

X_train = np.load('preprocessed_data/X_train.npy')
y_train = np.load('preprocessed_data/y_train.npy')
X_test = np.load('preprocessed_data/X_test.npy')
y_test = np.load('preprocessed_data/y_test.npy')

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
```

Make sure that the file paths are correctly specified to avoid `SyntaxWarning: invalid escape sequence`.

## Model Architecture

We use a pre-trained ResNet50 model for feature extraction, combined with custom layers for classification. Since the input images are grayscale, they are converted to a 3-channel format before being fed into ResNet50. Here's a brief overview of the model architecture:

1. **Input Layer:** Accepts 256x256 grayscale images.
2. **Concatenation:** Converts grayscale images to 3 channels.
3. **ResNet50 Backbone:** Pre-trained ResNet50 model without the top classification layer.
4. **Global Average Pooling:** Reduces the spatial dimensions.
5. **Fully Connected Layer:** 1024 units with ReLU activation.
6. **Dropout Layer:** Reduces overfitting by randomly dropping 50% of the units.
7. **Output Layer:** 1 unit with sigmoid activation for binary classification.

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Concatenate

IMG_SIZE = 256
input_shape = (IMG_SIZE, IMG_SIZE, 1)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
inputs = Input(shape=input_shape)
x = Concatenate()([inputs, inputs, inputs])
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=predictions)
```

## Training the Model

The model is compiled using the Adam optimizer and binary cross-entropy loss. Data augmentation techniques are applied to increase the robustness of the model.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

epochs = 10
batch_size = 16

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    epochs=epochs
)
```

## Evaluating the Model

After training, the model is evaluated on the test dataset to measure its accuracy and loss.

```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc}")
```

## Prediction on New Images

To make predictions on new grayscale images, use the following function:

```python
import cv2
import matplotlib.pyplot as plt

def predict_new_image(model, image_path):
    new_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_image_resized = cv2.resize(new_image, (IMG_SIZE, IMG_SIZE))
    new_image_resized = new_image_resized.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    prediction = model.predict(new_image_resized)
    is_oil_spill = prediction <= 0.5  # Modify this line based on prediction threshold
    plt.imshow(new_image, cmap='gray')
    plt.title(f"Oil Spill Detected: {'Yes' if is_oil_spill else 'No'}")
    plt.show()

# Example usage
predict_new_image(model, 'new_image.jpg')
```

## Results Visualization

Plot training and validation accuracy and loss over epochs to visualize model performance:

```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
```

## Conclusion

The fine-tuned ResNet50 model effectively detects oil spills in grayscale images, achieving an accuracy of approximately 83.7% on the test dataset. Further improvements could be made by experimenting with more sophisticated augmentation techniques, model architectures, or additional data. 

Feel free to contribute to this project or use it as a foundation for more advanced oil spill detection models.
