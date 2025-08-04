import tensorflow as tf
import tensorflow_datasets as tfds
import multiprocessing
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configure TensorFlow for multi-threading
tf.config.threading.set_intra_op_parallelism_threads(multiprocessing.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(multiprocessing.cpu_count())

# Print CPU count for multiprocessing optimization
print(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
print(f"TensorFlow configured to use all {multiprocessing.cpu_count()} CPU cores")

# Load and print dataset info
dataset_info = tfds.builder('fashion_mnist').info
print("\nDataset Information:")
print(f"Description: {dataset_info.description}")
print(f"Total examples: {dataset_info.splits['train'].num_examples + dataset_info.splits['test'].num_examples}")
print(f"Training examples: {dataset_info.splits['train'].num_examples}")
print(f"Test examples: {dataset_info.splits['test'].num_examples}")
print(f"Features: {dataset_info.features}")
print(f"Class names: {dataset_info.features['label'].names}")

(training_images, training_labels), (test_images, test_labels) = tfds.as_numpy(tfds.load('fashion_mnist', split = ['train', 'test'], batch_size= -1, as_supervised = True))

# Normalize pixel values
training_images = training_images / 255.0
test_images = test_images / 255.0

# Reshape for CNN (add channel dimension)
training_images = training_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Create augmentation function using ImageDataGenerator
def create_augmented_generator():
    """Create an ImageDataGenerator with light augmentations"""
    return ImageDataGenerator(
        rotation_range=20,          # Reduced rotation
        width_shift_range=0.2,     # Reduced horizontal shift
        height_shift_range=0.2,    # Reduced vertical shift
        zoom_range=0.1,           # Reduced zoom
        horizontal_flip=True,      # No horizontal flip for fashion items
        fill_mode='nearest'        # Fill pixels for transformations
    )

# Create the augmented data generator
datagen = create_augmented_generator()
datagen.fit(training_images)

# Create improved CNN model with batch normalization
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10)  # No activation for logits
])

# Compile model with learning rate scheduling
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Print model summary
print("\nModel Summary:")
model.summary()

# Create callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6),
]

# Train with augmented data - increase effective dataset size
print("\nStarting training with data augmentation...")
# Multiply training steps to see more augmented variations of each image
augmentation_multiplier = 3  # Each image will be seen 3x with different augmentations
steps_per_epoch = (len(training_images) // 32) * augmentation_multiplier

print(f"Original dataset size: {len(training_images)}")
print(f"Effective dataset size with augmentation: {len(training_images) * augmentation_multiplier}")
print(f"Steps per epoch: {steps_per_epoch}")

# Create tf.data dataset for better multiprocessing
def augment_data(image, label):
    """Apply augmentation using tf.image operations"""
    image = tf.cast(image, tf.float32)
    # Random rotation
    image = tf.image.rot90(image, tf.random.uniform([], 0, 4, dtype=tf.int32))
    # Random shifts
    image = tf.image.random_crop(tf.pad(image, [[2, 2], [2, 2], [0, 0]]), [28, 28, 1])
    # Random brightness
    image = tf.image.random_brightness(image, 0.1)
    return image, label

# Create tf.data pipeline for multiprocessing
train_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
train_dataset = train_dataset.repeat(augmentation_multiplier)
train_dataset = train_dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(32)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

history = model.fit(
    train_dataset,
    epochs=15,  # More epochs with early stopping
    validation_data=(test_images, test_labels),
    callbacks=callbacks,
    verbose=1
)

# Save the trained model
model.save('fashion_mnist_model.h5')
print(f"\nModel saved as 'fashion_mnist_model.h5'")
print(f"Final validation accuracy: {max(history.history['val_accuracy']):.4f}")

# Also save class names for later use
class_names = dataset_info.features['label'].names
import json
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print("Class names saved as 'class_names.json'")