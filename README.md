AURA-AI
AURA stands for  AI Universal Routine Advisor.  
In simple words:  AURA is an AI that looks at your face, listens to your voice, learns your habits, and tells you the best routine for the day.
🔍 What does AURA do?
Here are the functions :
1️⃣ Reads your Facial Expressions
Using webcam → detects emotions like:
Happy
Sad
Angry
Neutral
Stressed
Tired

How?
A CNN model trained on FER2013 / RAF-DB dataset.

2️⃣ Listens to your Voice Tone
Using microphone → detects:
Stress level
Calm or tense tone
Emotional cues in voice

How?
Audio ML model using MFCC features + a classifier.

3️⃣ Tracks Your Daily Behaviour
Using simple app inputs or logs:
Sleep time
Screen time
Study/work duration
Break patterns

How?
A time-series model (LSTM or simple statistical rules).

4️⃣ Combines All Signals (Multimodal Fusion)
This is the intelligent part.
AURA takes:
Your emotion (from face)
Your stress (from voice)
Your activity level (from logs)
And combines them to understand your current mental + physical state.

5️⃣ Generates a Personalized Daily Routine

Based on your state, it recommends:
If stressed:
Break times, breathing exercises, low workload ordering.

If tired:
Sleep schedule adjustment, lighter tasks, hydration reminders.

If energetic:
High-priority tasks first, no distractions mode.

If sad or demotivated:
Motivational tasks, music suggestions, shorter work cycles.

If overusing phone:
Screen-time reduction plan.

If irregular sleep:
Optimized sleep–wake schedule.


⚙️ Example Output (Easy to Explain)
Imagine you open the app in the morning.
Camera says you look tired
Voice says mild stress
Logs show you slept 5 hours
Screen time was high last night

AURA will generate:
“You seem tired with mild stress. Today, follow this routine:”
8:00–8:30 AM — Light breakfast + hydration
9:00–11:00 AM — High focus task (Pomodoro 25-5 cycles)
12:30 PM — Short walk (10 min)
3:00 PM — Relaxation: breathing for 3 mins
Reduce phone use by 20% today
Sleep early — recommended time: 10:45 PM
