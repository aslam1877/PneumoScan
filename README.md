# 🫁 PneumoScan — AI-Powered Pneumonia Detection

A deep learning web application that detects **Pneumonia** from chest X-ray images using a fine-tuned **ResNet18** model, deployed with **Streamlit**.

---

## 🚀 Features

- Upload chest X-ray images (JPG, PNG, BMP)
- Classifies as **Normal** or **Pneumonia**
- Displays **confidence scores** and **probability breakdown**
- Low-confidence warning for unreliable predictions
- Real-time inference with GPU support

---

## 🧠 Model Details

| Component | Details |
|-----------|---------|
| Architecture | ResNet18 (pretrained on ImageNet) |
| Transfer Learning | layer3 + layer4 + FC unfrozen |
| FC Head | Dropout(0.3) → Linear(512, 2) |
| Loss Function | Weighted CrossEntropyLoss (handles 3:1 class imbalance) |
| Optimizer | Adam (lr=0.0001) |
| Scheduler | StepLR (step=5, gamma=0.5) |
| Epochs | 10 |

### Data Augmentation

- Random resized crop (224×224)
- Horizontal flip
- Random rotation (±15°)
- Color jitter (brightness & contrast ±0.3)
- Random affine translation (±10%)

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **96.2%** |
| Normal Confidence | 94–100% |
| Pneumonia Confidence | 99–100% |

> ⚠️ In medical AI, minimizing false negatives is critical.
> This model uses weighted loss to compensate for the 3:1 class imbalance (3,875 Pneumonia vs 1,341 Normal images).

---

## 📁 Project Structure

```
PneumoScan-ai-main/
├── app.py                 # Streamlit web app
├── train_model.py         # Training script (primary)
├── download_data.py       # Dataset downloader (Kaggle)
├── test_inference.py      # Inference testing script
├── diagnose.py            # Model diagnostic script
├── requirements.txt       # Python dependencies
├── Dockerfile             # Production container
├── .dockerignore
├── .gitattributes         # Git LFS tracking
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── model.py           # ResNet18 model architecture
│   ├── train.py           # Training script (module version)
│   ├── data_loader.py     # Data loading & augmentation
│   └── evaluate.py        # Evaluation script
│
├── model/
│   └── pneumonia_model.pth  # Trained model weights (Git LFS)
│
└── data/                  # Dataset (not in repo, see below)
    ├── train/
    │   ├── NORMAL/        # 1,341 images
    │   └── PNEUMONIA/     # 3,875 images
    └── test/
        ├── NORMAL/        # 234 images
        └── PNEUMONIA/     # 390 images
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/aslamaslam/PneumoScan-ai.git
cd PneumoScan-ai
```

> **Note:** This repo uses Git LFS for model weights. Install [Git LFS](https://git-lfs.github.com/) first, then `git lfs pull` if weights weren't downloaded automatically.

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset (optional — only for training)

```bash
python download_data.py
```

This downloads the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle (~2.3 GB). Requires a Kaggle account — the script handles retries automatically.

### 5. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🐳 Docker Deployment

```bash
docker build -t pneumoscan .
docker run -p 8501:8501 pneumoscan
```

Open **http://localhost:8501**.

---

## 🔄 Retraining the Model

```bash
python download_data.py     # Get the dataset
python train_model.py       # Train (saves best model to model/)
```

---

## 📦 Dataset

- **Source:** [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size:** ~2.3 GB
- **Classes:** Normal, Pneumonia
- **Split:** 5,216 train / 624 test images

> Dataset is not included in the repository due to size. Use `download_data.py` to fetch it.

---

## 🛠️ Tech Stack

- **PyTorch** — Deep learning framework
- **torchvision** — Pretrained models & transforms
- **Streamlit** — Web app framework
- **Pillow** — Image processing
- **kagglehub** — Dataset download
- **Docker** — Containerized deployment

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.

---

## 👨‍💻 Author

Aslam Shaikh
