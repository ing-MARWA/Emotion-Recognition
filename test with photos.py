import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model('affectnet_model.h5')

# Create a dictionary to map integer labels to string labels
label_dict = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise', 7: 'contempt'}

# Load and preprocess a new image
image = cv2.imread('test-images/test_image7.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
image = cv2.resize(image, (48, 48))  # Resize to 48x48
image = np.array(image, dtype='float32') / 255.0  # Normalize pixel values
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Make a prediction on the image
prediction = model.predict(image)

# Print the predicted probabilities for each class
for i, label in label_dict.items():
    prob = prediction[0][i]
    print(f'{label}: {prob:.2f}')

label_idx = np.argmax(prediction)
label = label_dict[label_idx]

print('Prediction:', label)
