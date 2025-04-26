# Skin-Cancer-Analysis
COMP4030 Data Science with Machine Learning Group Project


# Skin Cancer Classification using CNN and Transfer Learning

This project applies Convolutional Neural Networks (CNNs) and Transfer Learning to classify skin lesion images into various diagnostic categories. The notebook (`Final-code.ipynb`) demonstrates data preprocessing, model training, evaluation, and performance visualization using TensorFlow and OpenCV.

---

## Project Structure

- `Final-code.ipynb` – Main notebook containing all code for preprocessing, modeling, and analysis
- Input images – Assumed to be loaded and processed from a structured directory (e.g., `images/train`, `images/test`)
- Output – Evaluation metrics, confusion matrices, accuracy/loss plots

---

##  Key Features

- Image preprocessing using OpenCV and TensorFlow utilities
- Data augmentation techniques
- Transfer Learning using pre-trained models (e.g., VGG16, ResNet)
- Custom CNN training
- Model evaluation using precision, recall, F1-score, and confusion matrix
- Visualization of training progress (accuracy/loss curves)

---

## Requirements
Env:  Python 3.10.16
Install the required Python packages using:

```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow pillow
```

---



##  Sample Output

- Confusion matrix and classification report
- Training accuracy/loss over epochs
- Model comparison (custom CNN vs. transfer learning models)

---

## Notes

- GPU acceleration is recommended for faster training.
- Ensure consistent image dimensions (usually 224x224) during preprocessing.
- Label encoding must match the folder names if using directory-based generators.

