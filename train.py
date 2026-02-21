import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Configuration
DATASET_DIR = "/home/prakyath/.gemini/antigravity/scratch/pneumodetect/dataset/archive/chest_xray"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 1

def build_model():
    """Builds a MobileNetV2 model for fine-tuning."""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model

def main():
    print("Setting up Data Generators...")
    
    # We use ImageDataGenerator for data augmentation on the training set
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation and test data should not be augmented
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"Class indices: {train_generator.class_indices}")

    print("Building Model...")
    model = build_model()

    # Callbacks
    checkpoint = ModelCheckpoint(
        "pneumodetect_model.keras", 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        verbose=1, 
        restore_best_weights=True
    )

    print("Starting Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping]
    )

    print("Evaluating on Test Set...")
    loss, accuracy, auc = model.evaluate(test_generator)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")
    
    # Optionally unfreeze and fine-tune here, but saving the basic model for now
    model.save("pneumodetect_model_final.keras")
    print("Model saved to pneumodetect_model_final.keras")

if __name__ == "__main__":
    main()
