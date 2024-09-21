# Alzheimer's Disease Detection using Pretrained CNN Models

## Project Overview
This project is focused on detecting Alzheimer's disease using convolutional neural networks (CNNs) with pretrained models like VGG19 and ResNet50. The objective is to extract features from brain images using the pretrained models and fine-tune them for accurate classification. 

### Pretrained Models Used:
- **ResNet50**: Utilized as a feature extractor and fine-tuned for Alzheimer's classification.
- **VGG19**: Applied for feature extraction and classification in a custom CNN architecture.

### Features:
- Feature extraction from brain images using ResNet50.
- Fine-tuning of VGG19 for Alzheimer's classification.
- Confusion matrix, accuracy metrics, and classification reports for evaluation.

### Files in the Repo:
1. `pretrained_vgg19_model.py` - VGG19-based pretrained model.
2. `pretrained_resnet50_model.py` - ResNet50-based pretrained model.
3. `pretrained_resnet50FE_model.py` - ResNet50 as a feature extractor.
4. `training_validation_curve.py` - Code to plot the training and validation curves.
5. `evaluation.py` - Code for evaluating model performance (accuracy, confusion matrix, classification report).

### Requirements:
- Python 3.x
- TensorFlow/Keras
- Matplotlib, Seaborn for plotting
- NumPy
- Scikit-learn for metrics

### How to Run:
1. Clone this repository to your local machine.
2. Install the required dependencies.
3. Execute the scripts according to the steps mentioned in the individual `.py` files.


### Article link
[Performance analysis of transfer learning based deep neural networks in Alzheimer classification](https://ieeexplore.ieee.org/document/10013501)
