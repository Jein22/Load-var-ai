# Load-var-ai
# ⚽ VAR-AI: Smart Football Refereeing Assistant

**VAR-AI** is an AI-powered assistant designed to support football referees by automatically analyzing video footage to detect offside positions, fouls, and other critical events using computer vision techniques.

---

## 🎯 Project Objective

To enhance fairness, accuracy, and speed in football refereeing decisions by:
- Detecting players and the ball in real-time.
- Drawing offside lines automatically.
- Providing visual and data-driven decision support.
- Reducing referee decision-making load.

---

## 🛠 Technologies Used

- **Python**  
- **OpenCV** – for video processing  
- **YOLOv5 (PyTorch)** – for object detection  
- **Jupyter Notebook** – for experiments and analysis  
- **NumPy / Matplotlib** – for visualization  

---

## 📁 Project Structure

```bash
football-var-ai/
│
├── data/                # Match footage and samples
│   └── sample_match.mp4
│
├── models/              # Pretrained or trained AI models
│   └── yolo_var_model.pt
│
├── src/                 # Source code
│   ├── detect_players.py
│   ├── draw_offside_line.py
│   └── var_interface.py
│
├── notebooks/           # Jupyter Notebooks
│   └── var_analysis.ipynb
│
├── requirements.txt     # Python libraries used
├── README.md            # Project documentation
└── LICENSE              # MIT License (optional)
