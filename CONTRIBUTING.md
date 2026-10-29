# Contributing to AirCanvas AI 🎨

Thank you for your interest in contributing to **AirCanvas AI**! We welcome contributions from developers of all experience levels.

---

## 🛠️ How Can You Help?

Here are some suggested features we'd love community support on:
- [ ] **Multi-Color Palette**: An on-screen toolbar to select different colors (Blue, Red, Green, Yellow, White).
- [ ] **Brush Size Adjuster**: Ability to change brush stroke thickness.
- [ ] **Save Artwork**: Keyboard shortcut or gesture to export canvas drawings as a timestamped `.png` file.
- [ ] **Geometric Shape Auto-Correction**: Fit freehand drawings to circles, rectangles, or triangles.
- [ ] **AI Handwriting OCR**: Transcribing words written in the air into digitized text using Gemini Vision or Tesseract.

---

## 🚀 Development Workflow

1. **Fork the Repository**:
   Click the **Fork** button at the top-right of the GitHub repository.

2. **Clone your fork**:
   ```bash
   git clone https://github.com/<your-username>/AirCanvas-AI.git
   cd AirCanvas-AI
   ```

3. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```

4. **Set Up the Environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

5. **Make Your Changes**:
   * Write clean, beginner-friendly, well-commented code.
   * Verify that the app launches and runs without errors.

6. **Commit & Push**:
   ```bash
   git add .
   git commit -m "Add: Amazing new feature"
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request**:
   Go to your fork on GitHub and submit a **Pull Request** describing your changes and test results.

---

## 📋 Code Style & Guidelines

- Follow PEP 8 guidelines where possible.
- Keep comments simple, clear, and beginner-accessible.
- Avoid adding heavy or unnecessary third-party dependencies unless strictly required.
