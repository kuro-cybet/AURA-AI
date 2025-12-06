# AURA-AI – AI Universal Routine Advisor

AURA stands for **AI Universal Routine Advisor**.

In simple words:

> AURA is an AI that looks at your face, listens to your voice, learns your habits, and tells you the best routine for the day.

Right now, this repo is focused on **Module 1 – Facial Emotion & Tiredness Detection** for a team of 4.

---

## 🎯 What AURA Does (Overall Vision)

AURA combines three types of signals:

1. **Facial behaviour**  
   - Emotions: *Happy, Sad, Angry, Neutral*  
   - Tiredness: based on eyes (blinks, droopiness) and yawning  

2. **Voice behaviour** (future module)  
   - Stress level  
   - Calm vs tense tone  

3. **Daily activity patterns** (future module)  
   - Sleep time  
   - Screen time  
   - Study/work duration  
   - Break patterns  

All of this is fused to estimate your **current state** and generate a **personalised daily routine**.

Example:

> “You seem tired with mild stress. Today, follow this routine:
> - 8:00–8:30 – Light breakfast + hydration  
> - 9:00–11:00 – High-focus task (Pomodoro 25–5)  
> - 12:30 – Short walk (10 min)  
> - 15:00 – Breathing exercise (3 minutes)  
> - Reduce phone use by 20%  
> - Sleep early, recommended 22:45”

---

## 🧱 Current Focus – Module 1: Face & Tiredness

We are 4 members working together **only on Module 1** right now:

- **Sudharsan** – Dataset & Preprocessing  
- **Suvedhan** – Model Architecture & Training  
- **Siva Dharani** – Evaluation & Tiredness Rules  
- **Dhanushya** – Real-Time Integration (Webcam)  

### Module 1 Objectives

- Detect **Happy / Sad / Angry / Neutral** from face images
- Detect **Tiredness** using:
  - Eye aspect ratio (EAR)
  - Blink patterns
  - Optional yawning
- Output a stable, real-time **state**:
  - `Happy/Fresh`, `Sad/Low Mood`, `Angry/Stressed`, `Neutral`, `Tired`

---

## 🏛️ Project Structure (planned)

```bash
AURA-AI/
│
├── data/                 # Preprocessed datasets (LOCAL, usually gitignored)
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/               # Saved model weights (LOCAL, usually gitignored)
│   └── emotion_best.h5
│
├── src/
│   ├── preprocessing/    # Dataset & preprocessing code
│   ├── training/         # Model definitions & training scripts
│   ├── evaluation/       # Metrics, confusion matrices, reports
│   └── realtime/         # Webcam + FaceMesh + EAR + final_state
│
├── utils/
│   ├── ear.py            # Eye Aspect Ratio helpers
│   ├── smoothing.py      # Sliding window smoothing
│   └── __init__.py
│
├── notebooks/            # Jupyter notebooks for experiments
│
├── progress/             # Daily reports (e.g. 2025-12-05.md)
│
├── README.md
├── .gitignore
└── CONTRIBUTING.md

How to Run:(This changes in future its just for you reference now!!!!)

# 1. Clone the repo
git clone https://github.com/kuro-cybet/AURA-AI.git
cd AURA-AI

# 2. Create venv (recommended) and install requirements
pip install -r requirements.txt

# 3. (Later) Run training
python src/training/train_emotion_model.py

# 4. (Later) Run real-time detector
python src/realtime/realtime_emotion_tired.py
