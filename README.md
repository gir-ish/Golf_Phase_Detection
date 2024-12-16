# Golf Swing Phase Detection

## Video Tutorial

<div align="center">
  <img src="https://github.com/gir-ish/Golf_Phase_Detection/blob/main/ui.gif" alt="Video Description" width="800" height="600">
</div>
## Overview
This project is a **Golf Swing Phase Detection** application built using **Streamlit**, **OpenCV**, and **MediaPipe**. The application allows users to upload a golf swing video, processes it to detect different phases of the swing, and visually represents these phases along with their corresponding frames. It is designed to provide an intuitive and interactive interface for analyzing golf swings.

---

## Features 🌟
- **Video Upload** 🎥: Upload golf swing videos in popular formats like MP4, AVI, MOV, and MKV.
- **Phase Detection** 🏌️:
  - **Setup Phase**
  - **Mid Backswing Phase**
  - **Top Backswing Phase**
  - **Mid Downswing Phase**
  - **Ball Impact Phase**
  - **Follow Through Phase**
- **Real-Time Progress Display** ⏳: Tracks and displays progress during video processing.
- **Visualization** 📸: Shows the first detected frame for each phase in a clean, modern layout.
- **Online Demo** 🚀: Try it online on Hugging Face Spaces 👉 [Golf Swing Analyzer](https://huggingface.co/spaces/your_username/golf-swing-analyzer)

---

## Directory Structure 📁
```plaintext
|- usr
  |- hip.py
  |- Phase_Co-ordinates.py
  |- vertical_hip.py
  |- spine.py
  |- head.py
  |- UI.py
  |- shoulder.py
|- README.md
```
- **usr/**: Modules for analyzing body parts and calculating swing phases.
- **README.md**: Project documentation.

---

## Installation 🛠️
Follow these steps to run the application locally:

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run usr/UI.py
   ```

4. **Open** your browser at `http://localhost:8501` to use the app.

---

## Output 📊
- A **progress bar** and status updates during video processing.
- Detected phases displayed in a beautiful grid with:
  - Phase Name
  - Corresponding Frame Image
  - Placeholders for missing phases.

---

## Technologies Used 💻
- **Python**: Backend processing and logic.
- **Streamlit**: Interactive web UI.
- **OpenCV**: Video frame processing.
- **MediaPipe**: Pose estimation for body landmark detection.
- **NumPy**: Vector and mathematical computations.

---

## Future Enhancements 🚀
- Add support for additional video formats.
- Enhance phase detection accuracy using machine learning.
- Provide advanced swing analytics (e.g., swing speed, angles).
- Save and compare results for historical analysis.

---

## Contributing 🤝
Contributions are welcome! Here's how you can help:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request for review.

---

## License 📜
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact 📧
For questions or suggestions, reach out to:

**Your Name**  
- **Email**: your.email@example.com  
- **GitHub**: [your_github_username](https://github.com/your_github_username)
