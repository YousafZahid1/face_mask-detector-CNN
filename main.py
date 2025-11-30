import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define paths
with_mask = "facemask-dataset/dataset/with_mask"
without_mask = "facemask-dataset/dataset/without_mask"


# Image parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)



#When u dont have data which is alredy split into train and val folders
# use validation_split in ImageDataGenerator
"""
Explnation how to split data up for training and validation;

GPT explanation: 
How Keras uses subset='training' and subset='validation'

You call flow_from_directory twice on the same folder, but:

subset	What images it gives you
'training'	the first 80% of images
'validation'	the remaining 20%
"""
val_test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = val_test_datagen.flow_from_directory(
    "facemask-dataset/dataset",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    color_mode='grayscale'
)

val_generator = val_test_datagen.flow_from_directory(
    "facemask-dataset/dataset",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    color_mode='grayscale'
)



model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    
    Flatten(),
    
    Dense(128, activation="relu"),
    Dropout(0.5),  
    Dense(1, activation="sigmoid") 

])
model.compile(
    optimizer="adam",
    loss="binary_crossentropy", 
    metrics=["accuracy"]
)

print(model.summary())


history = model.fit(
    train_generator,
    epochs=2,
    validation_data=val_generator
)


print("Evaluation:", model.evaluate(val_generator))
# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()