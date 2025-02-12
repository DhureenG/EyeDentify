# DeepFake Detection Using EfficientNetB0

This project implements a deep learning model using **EfficientNetB0** to detect deepfake images. The model is trained on a dataset containing real and fake images and uses transfer learning to improve accuracy. The training process involves two phases: one with a frozen base model and another with fine-tuning.

---

## **Project Overview**

- Uses **EfficientNetB0** as a feature extractor.
- Applies **data augmentation** to enhance training robustness.
- Implements **class weighting** to handle class imbalance.
- Employs **Early Stopping** and **ReduceLROnPlateau** for efficient training.
- Saves the model in both **Keras and Pickle formats**.
- Includes a function to **predict if an image is a deepfake**.

---

## **Installation**

Ensure that you have Python and the required libraries installed.

```bash
pip install tensorflow numpy scikit-learn pillow
```

---

## **Dataset Structure**

Place the dataset in the following format:

```
DeepFake Dataset/
│── Training/
│   ├── Real/
│   ├── Fake/
│── Validation/
│   ├── Real/
│   ├── Fake/
```

---

## **How to Run the Model**

### **Step 1: Train the Model**

Run the following command to start training:

```bash
python train.py
```

- This will train the model in two phases:

  1. **Initial Training** with a frozen base model.
  2. **Fine-tuning** after unfreezing the last 50 layers.

- The trained model will be saved as `deepfake_model_fixed.keras` and `deepfake_model.pkl`.

### **Step 2: Predict DeepFake Probability**

To test an image for deepfake detection, run:

```python
from detect import detect_deepfake

image_path = "./DSC_8248.jpg"
prediction = detect_deepfake(image_path)
print(f"Deepfake probability: {prediction:.4f}")
```

---

## **Model Architecture**

- **Base Model:** EfficientNetB0 (pre-trained on ImageNet)
- **Additional Layers:**
  - GlobalAveragePooling2D
  - BatchNormalization
  - Dense (256 units, ReLU, L2 regularization)
  - Dropout (0.5)
  - Dense (128 units, ReLU)
  - Dropout (0.3)
  - Dense (1 unit, Sigmoid)

---

## **Hyperparameters**

- **Optimizer (Phase 1):** Adam (learning rate: `0.0005`)
- **Optimizer (Phase 2):** Adam (learning rate: `5e-6`)
- **Loss Function:** Binary Crossentropy
- **Batch Size:** 32
- **Epochs:** 12 (for each phase)
- **Class Weights:** Computed dynamically to balance dataset

---

## **Callbacks Used**

- **Early Stopping:** Stops training if validation loss does not improve for 4 epochs.
- **ReduceLROnPlateau:** Reduces learning rate if validation loss stagnates for 2 epochs.

---

## **Results**

The model outputs a probability score indicating whether an image is a deepfake:

- **Score > 0.5:** Likely a deepfake.
- **Score ≤ 0.5:** Likely a real image.

---

## **Future Enhancements**

- Extend the dataset to improve generalization.
- Experiment with **EfficientNetB3 or B4** for improved accuracy.
- Implement a **CNN-based ensemble model** for better performance.
- Deploy the model as a **web application**.

---

## **License**

This project is open-source under the MIT License.

---

## **Contributors**

- Dhureen Gulati
- Meghaa Sathish
- Hannah John
- Cavin Shree Ramesh Kumar
- Rithu Nandana

---


