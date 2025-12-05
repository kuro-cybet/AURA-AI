import tensorflow as tf
from tensorflow.keras import layers, models

def build_emotion_model(input_shape=(48, 48, 3), num_classes=7):
    """
    Builds the CNN model architecture using Transfer Learning (MobileNetV2).
    
    Args:
        input_shape (tuple): Shape of the input image (Height, Width, Channels).
                             Note: MobileNet expects 3 channels (RGB).
        num_classes (int): Number of emotion categories (e.g., Happy, Sad, Angry...).
    
    Returns:
        model: A compiled Keras model ready for training.
    """
    
    # 1. Load the Base Model (Pre-trained on ImageNet)
    # include_top=False means we chop off the head to add our own emotion layers
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False, 
        weights='imagenet'
    )
    
    # Freeze the base model so we don't destroy pre-trained weights initially
    base_model.trainable = False

    # 2. Add Custom Layers for Emotion Detection
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(), # Condense features
        layers.Dense(128, activation='relu'), # Intermediate layer
        layers.Dropout(0.5), # Prevents overfitting by randomly turning off neurons
        layers.Dense(num_classes, activation='softmax') # Output layer (7 emotions)
    ])

    # 3. Compile the Model
    # Using Adam optimizer and Categorical Crossentropy as planned
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Test the skeleton
    # We assume 48x48 images (standard for FER datasets) with 3 channels (RGB)
    try:
        model = build_emotion_model(input_shape=(48, 48, 3), num_classes=7)
        print("\n✅ SUCCESS: Model created successfully!")
        print("---------------------------------------")
        model.summary()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")