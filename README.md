# Multimodal-Breast-Cancer-Analysis

This project implements a deep learning-based multimodal classification system for breast cancer detection using three imaging modalities: **Chest X-ray**, **Histopathological**, and **Ultrasound** images. It uses **ResNet50** for feature extraction and a custom fully connected network for classification.

---

## 📁 Dataset

The dataset consists of three types of medical images:

- `Chest_XRay_MSI` – `Malignant`, `Normal`
- `Histopathological_MSI` – `benign`, `malignant`
- `Ultrasound Images_MSI` – `benign`, `malignant`

These datasets are located inside the structured folder:
```

Dataset/
├── Chest\_XRay\_MSI/
│   ├── Malignant/
│   └── Normal/
├── Histopathological\_MSI/
│   ├── benign/
│   └── malignant/
└── Ultrasound Images\_MSI/
├── benign/
└── malignant/

````

---

## 📌 Project Highlights

- **Label Standardization** for all modalities.
- **Image Preprocessing**: RGB conversion, resizing, grayscale normalization.
- **ResNet50 Feature Extraction** for each modality.
- **Feature Concatenation**: Combines X-ray, Histopathology, and Ultrasound features.
- **Balanced Training via Augmentation** for the underrepresented class.
- **Custom Classification Head** with Dropout layers.
- **Performance Evaluation** using Accuracy, AUC, Confusion Matrix, ROC curve, and Classification Report.

---

## 🚀 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AdityaPatil1705/Multimodal-Breast-Cancer-Analysis.git
   cd Multimodal-Breast-Cancer-Analysis
````

2. **Set up environment** (suggested):

   ```bash
   conda create -n cancer-env python=3.9
   conda activate cancer-env
   pip install -r requirements.txt
   ```

3. **Run the notebook**:

   * Open Jupyter Notebook in the project directory:

     ```bash
     jupyter notebook
     ```
   * Open `Multimodal Breast Cancer Analysis.ipynb` and run all cells.

---

## 📊 Results

| Metric      | Value                   |
| ----------- | ----------------------- |
| Accuracy    | \~71–74%                |
| AUC         | \~0.78+                 |
| Sensitivity | Varies by class         |
| Specificity | Varies by class         |
| Model       | ResNet50 + Dense Layers |

---

## 📈 Visualizations

* Accuracy & Loss curves over epochs.
* ROC Curve with AUC.
* Normalized Confusion Matrix.

---

## 🧪 Libraries Used

* TensorFlow / Keras
* OpenCV
* Scikit-learn
* Matplotlib / Seaborn
* NumPy / Pandas

---

## ✅ To-Do / Improvements

* Integrate Grad-CAM for visual explanation.
* Explore transformer-based backbones.
* Experiment with weighted loss functions.

---
