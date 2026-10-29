# AirCanvas AI 🎨✨

An intelligent, real-time computer vision application that turns your webcam into an interactive air canvas. Draw, write, and sketch in mid-air using natural hand gestures—no special hardware required!

![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.14-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20API-orange.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

---

## 🌟 Key Features

* **👌 Pinch-to-Write (Natural Handwriting)**: Touch your index finger and thumb together to write; release to lift the pen. Feels just like holding a real pencil.
* **📈 Quadratic Bézier Spline Smoothing**: Eliminates polygonal jaggedness and connects stroke points into fluid, calligraphic curves.
* **⚡ Speed-Adaptive StreamLine Stabilizer**: Heavy tremor-filtering at low speeds for intricate writing, with zero lag during quick sweeps.
* **🔒 Schmitt-Trigger Pinch Hysteresis**: 3-frame debounce buffer prevents micro-gaps and stroke breaks when writing tight letter loops like "e" or "o".
* **✌️ V-Sign Hover Mode**: Lift your pen and freely glide your hand across the screen without leaving strokes.
* **🖐️ Open Palm Local Eraser**: An interactive eraser circle follows your palm to wipe away only the specific strokes you hover over.
* **✊ Fist Hold to Clear**: Make a closed fist and hold for 1 second to wipe the entire canvas clean.
* **🛡️ Anti-Interference Shield**: Anatomical geometry filtering + high confidence threshold (`0.70`) prevents the hand skeleton from latching onto background curtains, fabric folds, or wall patterns.
* **🚀 Zero-Setup Model Download**: Automatically fetches the official Google MediaPipe task model on first run.

---

## 🖐️ Gesture Control Guide

| Gesture | Mode | Function |
| :---: | :---: | :--- |
| **👌 Pinch (Index + Thumb)** | **WRITING** | Inks vibrant neon strokes on the canvas using Bézier curve interpolation. |
| **✌️ V-Sign (Index + Middle Up)** | **HOVER / MOVE** | Lifts the pen so you can reposition your hand without drawing. |
| **🖐️ Open Palm** | **LOCAL ERASER** | Wipes away only the hovered drawings under your palm. |
| **✊ Closed Fist (Hold 1s)** | **CLEAR CANVAS** | Shows a circular countdown and clears the entire screen. |
| **Keyboard `C`** | **CLEAR CANVAS** | Instantly clears the canvas. |
| **Keyboard `Q`** | **EXIT** | Exits the application and releases the webcam cleanly. |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Aabhipsa12/AirCanvas-AI.git
cd AirCanvas-AI
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python air_canvas.py
```
*(On first run, the pre-trained MediaPipe `hand_landmarker.task` model will download automatically).*

---

## 📂 Project Structure

```text
AirCanvas AI/
├── air_canvas.py            # Main AirCanvas application
├── webcam_test.py           # Milestone 1: Webcam & DirectShow verification
├── hand_tracking_test.py    # Milestone 2: 21-landmark tracking test
├── test_env.py              # Environment & package verification script
├── requirements.txt         # Core dependencies (OpenCV, MediaPipe, NumPy)
├── .gitignore               # Git ignore rules
├── CONTRIBUTING.md          # Guidelines for open-source contributors
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 

We would love community contributions for:
* 🎨 **Color Palette**: An on-screen UI menu to switch between multiple drawing colors.
* 📏 **Brush Size Selector**: Gesture or slider to adjust stroke thickness.
* 📐 **Geometric Shape Detection**: Recognizing rough circles, rectangles, and straight lines and snapping them into perfect vectors.
* 💾 **Canvas Export**: Pressing a key (e.g., `S`) to save your artwork as a PNG image.
* 🤖 **AI Handwriting Recognition (OCR)**: Using Gemini / OCR models to read and transcribe handwritten air text into digital text.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on opening Pull Requests.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.
