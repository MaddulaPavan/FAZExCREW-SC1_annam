Soil Classification Challenge:
Overview:
This project implements a deep learning solution for the Soil Classification Challenge, aiming to classify soil images into four categories: Alluvial soil, Black Soil, Clay soil, and Red soil. The solution uses a pre-trained EfficientNetB0 model with a custom classifier head, trained on a dataset of soil images. The code includes data preprocessing, model training, evaluation, and prediction generation for a test set.
Features

Data Preprocessing: Loads and prepares image data with augmentation for training and normalization for validation/testing.
Model: Utilizes EfficientNetB0 with transfer learning, fine-tuning the last 20 layers and adding a custom classifier head.
Class Imbalance Handling: Implements weighted random sampling and class-weighted loss to address class imbalance.
Training: Trains the model with AdamW optimizer, cosine annealing learning rate scheduler, and cross-entropy loss.
Evaluation: Computes F1 scores per class, generates a confusion matrix, and plots training history (loss and F1 score).
Prediction: Generates predictions for the test set and saves them in a submission-ready CSV file.

Requirements
To run this project, ensure you have the following dependencies installed:

Python 3.6+
PyTorch
torchvision
pandas
numpy
matplotlib
seaborn
Pillow (PIL)
scikit-learn
tqdm

You can install the dependencies using pip:
pip install torch torchvision pandas numpy matplotlib seaborn pillow scikit-learn tqdm

Dataset
The code expects the following dataset structure:
soil_classification-2025/
├── train/
│   └── <image_files>.jpg
├── test/
│   └── <image_files>.jpg
├── train_labels.csv
└── test_ids.csv


train_labels.csv: Contains columns image_id and soil_type for training images.
test_ids.csv: Contains column image_id for test images.
train/ and test/: Directories containing the training and test images, respectively.

Update the Config class in the code if your dataset is located elsewhere:
TRAIN_DIR = '/path/to/train'
TEST_DIR = '/path/to/test'
TRAIN_LABELS_FILE = '/path/to/train_labels.csv'
TEST_IDS_FILE = '/path/to/test_ids.csv'

Usage

Prepare the Environment:

Ensure all dependencies are installed.
Verify that the dataset is correctly structured and paths in Config are updated.


Run the Code:Execute the script to train the model, evaluate it, and generate predictions:
python soil_classification.py


Outputs:

submission.csv: Contains test set predictions with columns image_id and soil_type.
Plots: Training history (loss and F1 score) and confusion matrix for validation set.



Code Structure

Config Class: Defines hyperparameters and paths (e.g., image size, batch size, epochs).
SoilDataset Class: Custom PyTorch dataset for loading and transforming images.
SoilClassifier Class: Defines the EfficientNetB0-based model with a custom classifier head.
Data Loading: Handles loading and splitting of data into train/validation sets.
Training: Trains the model with weighted sampling and loss.
Evaluation: Computes metrics (F1 scores, confusion matrix) and visualizes results.
Prediction: Generates test set predictions and saves them to a CSV file.

Key Hyperparameters

Image Size: 224x224 pixels
Batch Size: 32
Epochs: 90
Learning Rate: 0.001
Weight Decay: 1e-4
Dropout Rate: 0.3

Notes

The code uses GPU (CUDA) if available; otherwise, it falls back to CPU.
Data augmentation includes random flips, rotations, color jitter, and affine transforms.
The model is fine-tuned with the last 20 layers unfrozen to balance learning and preserving pre-trained weights.
The minimum F1 score across classes is used as the primary evaluation metric.

Results
The script outputs:

Training and validation loss/F1 score plots.
Detailed evaluation metrics, including per-class F1 scores and a confusion matrix.
A submission file (submission.csv) with test set predictions.

Contributing
Feel free to open issues or submit pull requests for improvements, such as:

Adding support for other model architectures.
Enhancing data augmentation techniques.
Optimizing hyperparameters.

License
This project is licensed under the MIT License.
