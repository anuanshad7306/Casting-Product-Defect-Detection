
# 🔍 Casting Product Defect Detection using Deep Learning | CNN + Streamlit

This project aims to detect casting product defects using Convolutional Neural Networks (CNN) based on image classification. It leverages deep learning techniques to distinguish between **Defective** and **Non-Defective** metal casting products and provides an interactive web interface using **Streamlit**.

---

## 📌 Project Overview

Defective casting products can cause major quality issues in manufacturing. This project helps automate the quality inspection process by using deep learning to classify casting products based on X-ray images.

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow** & **Keras** – Model building
- **OpenCV / PIL** – Image processing
- **NumPy** – Numerical operations
- **Streamlit** – Web deployment
- **ImageDataGenerator** – Image augmentation & preprocessing

---

## 📂 Dataset

- **Source**: Kaggle / Provided internally
- Contains two folders:
  - `train/` – Images for training
  - `test/` – Images for validation/testing
- Two classes:
  - `def_front` (Defective)
  - `ok_front` (Non-Defective)

---

## 🧠 Model Architecture

```text
Input Layer (224x224x3)
→ Conv2D (32 filters) + ReLU → MaxPooling
→ Conv2D (64 filters) + ReLU → MaxPooling
→ Conv2D (128 filters) + ReLU → MaxPooling
→ Flatten
→ Dense (512 units) + ReLU
→ Dense (1 unit) + Sigmoid (Binary Classification)
```

---

## 📊 Results

- ✅ **Training Accuracy:** 97.5%
- ✅ **Test Accuracy:** 62.8%
- 🔍 Implemented dropout and data augmentation to reduce overfitting.

---

## 🚀 Streamlit App

A user-friendly Streamlit web interface lets users upload casting product images and instantly classify them.

To run the app:

```bash
streamlit run app.py
```

---

## 📁 File Structure

```
casting_defect_detection/
├── app.py                  # Streamlit application
├── model_training.py       # CNN model training and saving
├── casting_defect_model.h5 # Trained model
├── README.md               # Project documentation
└── dataset/
    ├── train/
    └── test/
```

---

## ✅ Features

- Deep Learning-based classification using CNN
- Real-time image prediction via web app
- Image preprocessing and augmentation
- Model saving and loading for deployment

---

## 🧠 Future Improvements

- Improve test accuracy with more data & hyperparameter tuning
- Add Grad-CAM or heatmaps for better interpretability
- Explore transfer learning with pretrained models

---

## 👨‍💻 Author

Made with 💡 by [Your Name]  
Mentored by **LearnLogic AI**

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
