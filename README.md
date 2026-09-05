# 🧠 CIFAR-10 Image Classifier

An end-to-end Deep Learning web app powered by **TensorFlow** and **Streamlit** for multi-class image classification.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit)

---

## ✨ Features

- **🎯 High Performance:** Custom CNN architecture trained with Data Augmentation & LR Scheduling.
- **⚡ Live Inference:** Real-time predictions with per-class confidence visualizers.
- **🎨 Custom UI:** Light/Dark mode toggle with clean metrics panel.
- **📊 Model Insights:** Live architectural parameters & layer breakdown.

---

## 🛠️ Tech Stack

| Domain | Technologies |
| :--- | :--- |
| **Language** | Python 3.10 |
| **ML Framework** | TensorFlow / Keras |
| **Frontend UI** | Streamlit |
| **Processing** | NumPy, Pillow, Matplotlib |

---

## 📦 Dataset (CIFAR-10)

Trained on 60,000 $32 \times 32$ RGB images across 10 classes:
`Airplane` • `Automobile` • `Bird` • `Cat` • `Deer` • `Dog` • `Frog` • `Horse` • `Ship` • `Truck`

---

## 🚀 Quickstart

```bash
# 1. Clone & Navigate
git clone [https://github.com/YOUR_USERNAME/cifar10-cnn-image-classifier.git](https://github.com/YOUR_USERNAME/cifar10-cnn-image-classifier.git)
cd cifar10-cnn-image-classifier

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Train Model
python train_model.py

# 4. Launch App
streamlit run app.py