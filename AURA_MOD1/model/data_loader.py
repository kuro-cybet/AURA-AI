import tensorflow as tf
import os

def load_emotion_datasets(data_dir, img_size=(48, 48), batch_size=32):
    """
    Loads images from folders and creates efficient TensorFlow datasets.
    
    Expected Structure:
    data_dir/
      ├── train/
      │    ├── angry/
      │    ├── happy/
      │    └── ...
      └── test/
           ├── angry/
           ├── happy/
           └── ...
    """
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    print(f"\n🔍 Looking for data in: {data_dir}")

    # 1. Load Training Data
    if not os.path.exists(train_dir):
        print(f"⚠️  WARNING: Train folder not found at {train_dir}")
        return None, None, None

    print("   [1/2] Loading Training Set...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical' # Converts 'happy' to [0, 0, 0, 1, 0, 0, 0]
    )

    # 2. Load Validation/Test Data
    if os.path.exists(test_dir):
        print("   [2/2] Loading Test Set...")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            seed=123,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='categorical'
        )
    else:
        print("⚠️  No 'test' folder found. Splitting training data instead.")
        # Automatic split if no test folder exists (80% train / 20% val)
        val_ds = train_ds.take(int(len(train_ds) * 0.2))
        train_ds = train_ds.skip(int(len(train_ds) * 0.2))

    # 3. Optimize for Performance (Pre-fetching)
    # This keeps the GPU busy while the CPU loads the next batch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    class_names = train_ds.class_names if hasattr(train_ds, 'class_names') else []
    
    return train_ds, val_ds, class_names

if __name__ == "__main__":
    # Test the loader with a dummy path
    # (It will fail gracefully if folders don't exist, which is fine for now)
    try:
        current_dir = os.getcwd()
        # We assume Sudharsan will put data in a folder named 'dataset'
        dataset_path = os.path.join(current_dir, 'dataset') 
        
        train_data, val_data, classes = load_emotion_datasets(dataset_path)
        
        if train_data:
            print(f"\n✅ SUCCESS: Data loaded. Classes found: {classes}")
        else:
            print("\nℹ️  NOTE: No real data found yet. (Waiting for Member 1)")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        