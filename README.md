# Emotion-Recognition

# Facial Emotion Analysis

This repository contains code for performing facial emotion analysis using deep learning techniques. It uses a pre-trained convolutional neural network model to detect facial emotions in real-time using a webcam.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- TensorFlow
- Keras
- Matplotlib

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/your-repo.git
   ```

2. Install the required dependencies:

   ```
   pip install opencv-python numpy tensorflow keras matplotlib
   ```

3. Download the pre-trained model file `best_model.h5` and place it in the cloned repository.

## Usage

1. Run the script:

   ```
   python facial_emotion_analysis.py
   ```

   This will open a window displaying the webcam feed and overlaying the detected emotions on the faces.

2. Press `q` to exit the program.

## Credits

- The pre-trained model used in this project is based on the [FER2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and was trained using deep learning techniques.
- The Haar cascade classifier for face detection is provided by OpenCV.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- This project was inspired by the work on facial emotion recognition in the field of computer vision.
- Special thanks to the contributors of the open-source libraries used in this project.
