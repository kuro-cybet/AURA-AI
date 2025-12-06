import tensorflow as tf
import os
import sys

# Add current directory to path so we can import our own modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from emotion_model import build_emotion_model
from data_loader import load_emotion_datasets

# --- CONFIGURATION ---
# Go up 2 levels from 'AURA_MOD1/model' to reach the root, then into 'dataset'
BASE_DIR = os.path.dirname(os.path.dirname(current_dir))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

BATCH_SIZE = 32
IMG_SIZE = (48, 48)
EPOCHS = 10  # Start with 10 for the baseline

def train_engine():
    print(f"\n🚀 STARTING AURA TRAINING ENGINE...")
    print(f"   Looking for data in: {DATA_DIR}")
    
    # 1. Load Data
    train_ds, val_ds, classes = load_emotion_datasets(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    
    if not train_ds:
        print("❌ CRITICAL: No data found. Make sure 'dataset' folder exists!")
        return

    # If classes are empty (dummy mode), default to 7 so model builds anyway
    num_classes = len(classes) if classes else 7
    print(f"   Target Classes: {num_classes} {classes}")

    # 2. Build Model
    print("\n🏗️  Building MobileNetV2 Architecture...")
    model = build_emotion_model(input_shape=IMG_SIZE + (3,), num_classes=num_classes)

    # 3. Setup Checkpoints (Save the best version automatically)
    checkpoint_path = os.path.join(current_dir, "emotion_best.h5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, 
        save_best_only=True, 
        monitor='val_accuracy', 
        mode='max',
        verbose=1
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )

    # 4. Train
    print("\n⚡ BEGINNING TRAINING... (Press Ctrl+C to stop early)")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[checkpoint, early_stop]
        )
        print(f"\n✅ SUCCESS: Training Finished! Model saved to: {checkpoint_path}")
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        print("   (Note: If this failed on 'fake.jpg', it's because the files are 0 bytes.")
        print("    This script will work perfectly once Member 1 provides real images.)")

if __name__ == "__main__":
    train_engine()