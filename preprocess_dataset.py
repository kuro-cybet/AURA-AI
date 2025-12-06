import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# --------- CONFIG ---------
RAW_DATA_DIR = "data_raw"        # where your original images are
OUTPUT_DIR = "data_processed"    # where processed images will be saved
IMG_SIZE = 96                    # final face size: 96x96 pixels
EMOTIONS = ["angry", "happy", "sad", "neutral"]
# --------------------------


def create_output_dirs():
    """
    Create output folders:
    data_processed/train/<emotion>/
    data_processed/val/<emotion>/
    data_processed/test/<emotion>/
    """
    for split in ["train", "val", "test"]:
        for emotion in EMOTIONS:
            path = os.path.join(OUTPUT_DIR, split, emotion)
            os.makedirs(path, exist_ok=True)
    print("[INFO] Output folders created (if not already).")


def load_images_and_labels():
    """
    Load images from data_raw/<emotion>/,
    resize, normalize, and build X (images) and y (labels).
    """
    images = []
    labels = []

    for label, emotion in enumerate(EMOTIONS):
        folder = os.path.join(RAW_DATA_DIR, emotion)
        if not os.path.exists(folder):
            print(f"[WARN] Folder not found: {folder}")
            continue

        print(f"[INFO] Loading images for class: {emotion}")

        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)

            # Only allow typical image extensions
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARN] Could not read: {img_path}")
                    continue

                # Convert BGR -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize to IMG_SIZE x IMG_SIZE
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                # Normalize to [0, 1]
                img = img.astype("float32") / 255.0

                images.append(img)
                labels.append(label)

            except Exception as e:
                print(f"[ERROR] Failed on {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    print(f"[INFO] Loaded {len(images)} images in total.")
    return images, labels


def split_data(X, y, test_size=0.15, val_size=0.15):
    """
    Split into train, val, test with stratification (keep class balance).
    """
    if len(X) == 0:
        print("[ERROR] No images to split.")
        return None, None, None, None, None, None

    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Then split train/val from the train_val set
    val_ratio = val_size / (1 - test_size)  # adjust ratio
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        stratify=y_train_val,
        random_state=42,
    )

    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_split(X, y, split_name):
    """
    Save processed images to:
    data_processed/<split_name>/<emotion>/
    """
    if X is None or y is None:
        print(f"[WARN] No data to save for split: {split_name}")
        return

    for img, label in zip(X, y):
        emotion = EMOTIONS[label]
        out_dir = os.path.join(OUTPUT_DIR, split_name, emotion)
        os.makedirs(out_dir, exist_ok=True)

        # Create a random filename
        filename = f"{emotion}_{np.random.randint(1_000_000)}.jpg"
        out_path = os.path.join(out_dir, filename)

        # Convert back to BGR [0,255] for saving
        img_bgr = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, img_bgr)


def main():
    print("[STEP] Creating output folders...")
    create_output_dirs()

    print("[STEP] Loading and preprocessing images from data_raw/ ...")
    X, y = load_images_and_labels()
    if len(X) == 0:
        print("[ERROR] No images found in data_raw/. Put some images first.")
        return

    print("[STEP] Splitting into train / val / test ...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("[STEP] Saving TRAIN set images ...")
    save_split(X_train, y_train, "train")

    print("[STEP] Saving VAL set images ...")
    save_split(X_val, y_val, "val")

    print("[STEP] Saving TEST set images ...")
    save_split(X_test, y_test, "test")

    print("[DONE] Preprocessing complete.")
    print(f"[PATH] Processed data saved under: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
