# Emotion-Recognition

# Emotion Recognition using Deep Learning

This repository contains code for building an emotion recognition system using deep learning. The system can detect emotions like anger, disgust, fear, happy, neutral, sad, surprise, and contempt in images and live video stream.

## Prerequisites

- Python 3
- OpenCV
- NumPy
- Keras
- TensorFlow
- Pandas
- scikit-learn

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/your-username/emotion-recognition.git
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Emotion Recognition from Image

To perform emotion recognition on a single image, run the following command:

```
python predict_image.py --image_path <path-to-image>
```

Make sure to replace `<path-to-image>` with the actual path to the image you want to analyze.

### 2. Emotion Recognition from Live Video Stream

To perform real-time emotion recognition from the webcam, run the following command:

```
python predict_video.py
```

This will open a window showing the live video stream with emotions detected in real-time.

### 3. Train the Emotion Recognition Model

If you want to train your own emotion recognition model, follow these steps:

1. Prepare the dataset:
   - Create a CSV file named `affectnet.csv` with two columns: `image_path` and `label`.
   - Put all the training images in a folder and update the `image_path` column in the CSV file with the path to each image.
   - Assign labels to each image and update the `label` column in the CSV file accordingly.

2. Run the training script:

   ```
   python train_model.py
   ```

   This will train the model using the images and labels specified in the `affectnet.csv` file. The best model will be saved as `affectnet_model.h5`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The emotion recognition model is based on the [AffectNet](http://mohammadmahoor.com/affectnet/) dataset.
- The face detection is performed using the Haar cascade classifier provided by OpenCV.
