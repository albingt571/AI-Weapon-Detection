# AI Weapon Detection using YOLOv5

This project uses *YOLOv5* (You Only Look Once) for real-time weapon detection. The system is capable of identifying weapons in images or video streams and providing predictions. It also includes a text-to-speech feature that notifies the user when a weapon is detected with a high confidence level.

## Project Structure


- content/: Contains all the project files.
  - yolov5/: The YOLOv5 repository and training output.
    - runs/train/yolov5s_results/: Contains the training results such as model weights (best.pt, last.pt) and images.
- PredictionCode/: Contains scripts for making predictions.
    - main.py: The main script for making predictions.
    - text_to_speech.py: Script for converting text to speech.
    - requirements.txt: Lists the dependencies needed for running the prediction code.
- TrainingCode/: Folder containing the training code (downloaded after training on Colab).
    - TrainingCode/: Folder containing the training code (downloaded after training on Colab).
- data.yaml: File containing the class names for weapon detection.
- best.pt: model for weapon detection.


---

## Project Description

This project is designed to detect weapons in images or video streams using *YOLOv5, a state-of-the-art object detection model. The model is trained on a custom dataset of weapons loaded from **Roboflow* using their API in *Google Colab. Once the model is trained, the weights (best.pt*) are downloaded and used for inference in real-time detection. The trained model is capable of detecting weapons with a high level of accuracy. Additionally, the system includes a text-to-speech notification feature that announces the detection of a weapon when the confidence exceeds a specified threshold.

### Features:
- *Weapon Detection*: Detects weapons with YOLOv5 model.
- *Text-to-Speech*: Uses the pyttsx3 library to announce the detection of a weapon.
- *Custom Dataset*: Trained on a custom dataset of weapons.
- *Real-time Prediction*: Detects weapons in images or video streams.

---

## Setup Instructions

### 1. Clone the Repository
Start by cloning the repository to your local machine:

bash
git clone <your-repository-url>


### 2. Install Dependencies
The project requires the following Python packages:

torch: For deep learning operations.
opencv-python: For image and video processing.
pyttsx3: For text-to-speech functionality.
ultralytics: For YOLOv5 model operations.
To install these, navigate to the PredictionCode folder and run:

bash
cd PredictionCode
pip install -r requirements.txt

### 3. Training the Model (In Google Colab)
Custom Dataset: The custom dataset for training the model is hosted on Roboflow.

Training in Colab: Use the provided Colab notebook (or your own) to load the dataset and train the YOLOv5 model using the train.py script provided by YOLOv5.

Load the dataset using the Roboflow API.
Train the model using the yolov5s architecture.
The model is trained on the custom weapon dataset, and the best weights (best.pt) are saved at the end of the training.
Example Colab Code to Train:

bash
from roboflow import Roboflow
import torch
from yolov5 import train

# Load the dataset from Roboflow using your API key
rf = Roboflow(api_key="your_roboflow_api_key")
project = rf.workspace("your_workspace").project("your_project")
dataset = project.version(1).download("yolov5")

# Train YOLOv5 model on the dataset
train.run(data=dataset.location, cfg="yolov5s.yaml", epochs=50, imgsz=640)

Download Model Weights: After the training completes, download the best.pt and last.pt weights from Google Colab.

### 4. Prediction and Inference
Once the model is trained, you can run the prediction script to detect weapons in images or video streams.

Download the best.pt and last.pt files from Colab and place them in the content/yolov5/runs/train/yolov5s_results/ folder.
Navigate to the PredictionCode folder.
Run the prediction script:
bash

python main.py

This script performs the following:

Loads the trained model (best.pt).
Processes the input image or video stream.
Detects weapons and displays the results.
If a weapon is detected with a confidence greater than 0.67, a text-to-speech notification will announce the detection.
Example of a Prediction:
bash

def text_to_speech(frame):
    save_path = 'WeaponDetection Pictures'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Assuming confidence is calculated for the detected object
    if confidence > 0.67:
        pyttsx3.speak("Weapon Detected")

Files Overview
### 1. data.yaml
This file contains the class names of the detected weapons. The data.yaml file is used for training the YOLOv5 model and includes:

bash

train: path_to_train_images
val: path_to_validation_images
nc: 1  # Number of classes
names: ['weapon']  # List of class names

### 2. best.pt and last.pt
These files are the saved weights of the trained YOLOv5 model. best.pt is the model with the best performance during training, while last.pt is the most recent model checkpoint.

### 3. requirements.txt
This file lists the required Python packages to run the prediction code:

bash

torch
opencv-python
pyttsx3
ultralytics

### Model Training and Results
The model is trained using YOLOv5s, a small and efficient version of the YOLOv5 architecture. After training, the model is able to detect weapons with a confidence threshold of 0.67 or higher.

The trained model and results (images) are saved in the following directory:

bash

AIYolov5/content/yolov5/runs/train/yolov5s_results/

### Troubleshooting
Issue: pip install -r requirements.txt fails.

Solution: Ensure you are using a virtual environment and have the correct Python version (>=3.6). Try upgrading pip before running the install command:
bash

pip install --upgrade pip

Issue: Model performance is low.

Solution: Try increasing the dataset size or adjusting training parameters (e.g., learning rate).
Future Improvements
Increase Dataset Size: More diverse weapon images will help improve the model's accuracy.
Real-Time Video Processing: Implement video stream processing for real-time detection
