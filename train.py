import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam

# Load the dataset
df = pd.read_csv('affectnet.csv')
image_paths = df.iloc[:, 1].values
labels = df.iloc[:, 2].values

# Create a dictionary to map string labels to integer labels
label_dict = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6 , 'contempt': 7}

# Convert string labels to integer labels
train_labels = np.array([label_dict[label] for label in labels])

# Preprocess the images
images = []
for path in image_paths:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (48, 48))  # Resize to 48x48
    image = np.array(image, dtype='float32') / 255.0  # Normalize pixel values
    images.append(image)
images = np.stack(images)

# Split the dataset
train_images, test_images, train_labels, test_labels = train_test_split(images, train_labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Convert the integer labels to one-hot encoding
train_labels = to_categorical(train_labels, 8)
val_labels = to_categorical(val_labels, 8)
test_labels = to_categorical(test_labels, 8)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(8, activation='softmax'))

# Define a learning rate scheduler to decrease the learning rate over time
def lr_scheduler(epoch):
    lr = 0.001
    if epoch > 5:
        lr /= 2
    if epoch > 10:
        lr /= 2
    return lr

# Define data augmentation settings
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

# Fit the data generator on the training data
train_datagen.fit(train_images)

# Compile the model with a lower learning rate and a learning rate scheduler
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define a checkpoint to save the best model
checkpoint = ModelCheckpoint('affectnet_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Define a learning rate scheduler callback
scheduler = LearningRateScheduler(lr_scheduler)

# Train the model with early stopping, checkpointing, and the learning rate scheduler
model.fit(train_datagen.flow(train_images, train_labels, batch_size=32),
        epochs=100,
        validation_data=(val_images, val_labels),
        callbacks=[checkpoint, scheduler])

# Evaluate the model on the testing set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)