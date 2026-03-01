# 👋 Hand Landmark Data Collection & Training System (Physical AI)

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Mediapipe](https://img.shields.io/badge/Mediapipe-v0.10-007ACC?logo=google&logoColor=white)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-v4.8-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **professional, high-performance toolkit** for capturing 3D hand landmarks, building gesture datasets, and training real-time classification models. This project is designed for **Physical AI** applications such as gesture-controlled robotics, XR interfaces, and human-computer interaction (HCI).

---

## ✨ Features

- **🚀 Real-time 3D Tracking**: Powered by Mediapipe Hands for high-accuracy landmark detection.
- **🖼️ Modern GUI**: Sleek `CustomTkinter` interface for effortless data collection.
- **🏷️ Dynamic Labeling**: Switch gesture categories (e.g., Fist, Wave, Pinch) on the fly.
- **🏗️ Full ML Pipeline**: Includes tools to collect, save, and train Random Forest models on your data.
- **📊 3D Visualizer**: Live rendering of hand connections and landmarks during capture.
- **🛠️ Professional Structure**: Modular, maintainable Python architecture.

---

## 📂 Project Structure

```text
Physical-AI-Hand-Landmarks/
├── src/
│   ├── collector.py      # Core landmark detection logic
│   ├── ui.py             # Sleek CustomTkinter interface
│   ├── train.py          # Machine learning training pipeline
│   └── utils.py          # Helper functions
├── data/                 # Directory for CSV gesture datasets
├── Hand_Landmarks_Data_Collection.py  # Primary Entry Point
├── README.md             # Project documentation
├── requirements.txt      # Dependency list
└── .gitignore            # Version control filters
```

---

## 🚀 Quick Start

### 1️⃣ Installation

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
git clone https://github.com/punyamodi/Physical-AI-Hand-Landmarks
cd Physical-AI-Hand-Landmarks
pip install -r requirements.txt
```

### 2️⃣ Run the Collector

Start the modern GUI and begin recording your hand gestures:

```bash
python main.py
```

1.  Enter a category name (e.g., `thumbs_up`).
2.  Press **Start Recording** and move your hand.
3.  Press **Stop Recording** and **Save Data (CSV)**.

### 3️⃣ Train Your Model

After collecting data for a few gestures, train a classifier:

```bash
python src/train.py
```

This will generate a `.pkl` model file that can be used for real-time inference.

---

## 🧠 Physical AI Context

Physical AI bridges digital models with physical interaction. By capturing high-fidelity hand landmarks, we enable robots to understand human gestures, power immersive VR experiences, and build safer touchless interfaces in industrial environments.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.

Developed by [Punyamodi](https://github.com/punyamodi)

