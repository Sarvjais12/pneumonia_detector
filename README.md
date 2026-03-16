# 🫁 Interpretable Pneumonia AI (VGG16 + Grad-CAM)

### **Explainable Deep Learning for Medical Image Analysis**
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Sarvjais12/your-pneumonia-space-name) This project bridges the gap between high-accuracy medical AI and clinical trust. It uses a fine-tuned **VGG16** convolutional neural network to detect pneumonia from chest X-rays, paired with **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visually explain the model's predictions.

---

## 🚀 Key Features

* **Visual Interpretability (Explainable AI)**: Generates Grad-CAM heatmaps to highlight the exact regions of the X-ray (e.g., lung opacities) that triggered the positive prediction, solving the "black box" problem of deep learning.
* **High-Precision Classification**: Leverages transfer learning via the VGG16 architecture to accurately distinguish between normal and pneumonia-infected lungs.
* **Interactive Clinical UI**: Deployed an intuitive web interface where users can upload an X-ray and instantly receive both the diagnosis and the visual evidence.
* **Real-time Inference**: Optimized for fast prediction delivery using Hugging Face Spaces.

---

## 🏗️ How it Works (The Architecture)

1. **Preprocessing**: Raw X-ray images are resized, normalized, and augmented to match the VGG16 input requirements.
2. **Feature Extraction**: The deep convolutional layers of the network extract hierarchical features from the medical images.
3. **Classification**: Dense layers map the extracted features to a binary prediction (Pneumonia vs. Normal).
4. **Grad-CAM Overlay**: The gradients of the target concept flowing into the final convolutional layer are computed to produce a coarse localization heatmap. This map is superimposed over the original X-ray to show exactly where the model is "looking."

---

## 🛠️ Tech Stack

* **Deep Learning Framework**: TensorFlow / Keras (or PyTorch)
* **Computer Vision**: OpenCV, NumPy, Matplotlib
* **Model Architecture**: VGG16
* **Interface & Deployment**: Gradio, Hugging Face Spaces

---

## 📋 Professional Context

In medical imaging, raw accuracy metrics are not enough; practitioners require *interpretability*. This project was developed to demonstrate how **Explainable AI (XAI)** can be implemented in healthcare diagnostic tools to build clinical trust and provide actionable visual insights.

---

### **Installation & Local Usage**
```bash
git clone [https://github.com/sarvjais12/Interpretable-Pneumonia-AI.git](https://github.com/sarvjais12/Interpretable-Pneumonia-AI.git)
cd Interpretable-Pneumonia-AI
pip install -r requirements.txt
python app.py
