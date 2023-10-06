import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model('affectnet_model.h5')

# Create a dictionary to map integer labels to string labels
label_dict = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise', 7: 'contempt'}

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Loop over each face and make a prediction
        for (x, y, w, h) in faces:
            # Extract the face region from the frame
            face = gray[y:y+h, x:x+w]
            # Preprocess the face image
            face = cv2.resize(face, (48, 48))
            face = np.array(face, dtype='float32') / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.repeat(face, 3, axis=-1)
            face = np.expand_dims(face, axis=0)

            # Make a prediction on the preprocessed image
            predictions = model.predict(face)

            # Get the predicted label
            label = np.argmax(predictions)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (194, 24, 91), 2)

            # Add the predicted label to the frame
            cv2.putText(frame, label_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (194, 24, 91), 2)

        # Display the resulting frame
        cv2.imshow('Emotion Recognition', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()