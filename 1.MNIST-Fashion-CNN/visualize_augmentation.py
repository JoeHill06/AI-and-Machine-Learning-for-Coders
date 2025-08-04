import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import json

def load_data_and_classes():
    """Load the Fashion MNIST dataset and class names"""
    # Load dataset
    (training_images, training_labels), _ = tfds.as_numpy(
        tfds.load('fashion_mnist', split=['train', 'test'], batch_size=-1, as_supervised=True)
    )
    
    # Normalize and reshape
    training_images = training_images / 255.0
    training_images = training_images.reshape(-1, 28, 28, 1)
    
    # Load class names
    try:
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        # Fallback to default class names
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    return training_images, training_labels, class_names

def augment_data(image, label):
    """Apply augmentation using tf.image operations - EXACT COPY from MNIST.py"""
    image = tf.cast(image, tf.float32)
    # Random rotation
    image = tf.image.rot90(image, tf.random.uniform([], 0, 4, dtype=tf.int32))
    # Random shifts
    image = tf.image.random_crop(tf.pad(image, [[2, 2], [2, 2], [0, 0]]), [28, 28, 1])
    # Random brightness
    image = tf.image.random_brightness(image, 0.1)
    return image, label

def visualize_augmentation(num_images=50, images_per_row=10, figsize_per_image=1.2):
    """
    Visualize original images and their augmented versions
    
    Args:
        num_images: Number of images to display (default 50)
        images_per_row: Number of images per row (default 10)
        figsize_per_image: Size factor for each image subplot
    """
    print("Loading Fashion MNIST dataset...")
    training_images, training_labels, class_names = load_data_and_classes()
    
    # Select random images
    indices = np.random.choice(len(training_images), num_images, replace=False)
    selected_images = training_images[indices]
    selected_labels = training_labels[indices]
    
    # Calculate figure dimensions
    rows = 2 * ((num_images + images_per_row - 1) // images_per_row)  # 2 rows per set (original + augmented)
    cols = min(num_images, images_per_row)
    
    # Create figure
    fig = plt.figure(figsize=(cols * figsize_per_image, rows * figsize_per_image))
    fig.suptitle(f'Fashion MNIST: Original vs Augmented Images ({num_images} samples)', 
                fontsize=16, fontweight='bold', y=0.95)
    
    print(f"Generating augmented versions of {num_images} images...")
    
    # Process images in batches for efficiency
    original_batch = tf.constant(selected_images)
    labels_batch = tf.constant(selected_labels)
    
    # Apply augmentation
    augmented_batch, _ = tf.map_fn(
        lambda x: augment_data(x[0], x[1]),
        (original_batch, labels_batch),
        fn_output_signature=(tf.TensorSpec([28, 28, 1], tf.float32), tf.TensorSpec([], tf.int64))
    )
    
    # Convert back to numpy
    augmented_images = augmented_batch.numpy()
    
    # Plot images
    for i in range(num_images):
        # Calculate position
        row_group = i // images_per_row
        col = i % images_per_row
        
        # Original image (top row of each group)
        original_pos = row_group * 2 * images_per_row + col + 1
        ax1 = plt.subplot(rows, images_per_row, original_pos)
        plt.imshow(selected_images[i].squeeze(), cmap='gray')
        plt.title(f'Original\n{class_names[selected_labels[i]]}', fontsize=8, pad=2)
        plt.axis('off')
        
        # Augmented image (bottom row of each group)
        augmented_pos = (row_group * 2 + 1) * images_per_row + col + 1
        ax2 = plt.subplot(rows, images_per_row, augmented_pos)
        plt.imshow(augmented_images[i].squeeze(), cmap='gray')
        plt.title('Augmented', fontsize=8, pad=2, color='red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.1)
    
    # Save the visualization
    filename = f'augmentation_comparison_{num_images}_images.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Visualization saved as '{filename}'")
    
    plt.show()

def visualize_augmentation_grid(image_index=None, num_augmentations=20):
    """
    Show one original image with multiple augmented versions in a grid
    
    Args:
        image_index: Specific image index to use (if None, random)
        num_augmentations: Number of augmented versions to show
    """
    print("Loading Fashion MNIST dataset...")
    training_images, training_labels, class_names = load_data_and_classes()
    
    # Select image
    if image_index is None:
        image_index = np.random.randint(0, len(training_images))
    
    original_image = training_images[image_index]
    original_label = training_labels[image_index]
    class_name = class_names[original_label]
    
    print(f"Showing image {image_index}: {class_name}")
    
    # Generate multiple augmented versions
    image_tensor = tf.constant(original_image)
    label_tensor = tf.constant(original_label)
    
    augmented_images = []
    for _ in range(num_augmentations):
        aug_img, _ = augment_data(image_tensor, label_tensor)
        augmented_images.append(aug_img.numpy())
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_augmentations + 1)))  # +1 for original
    
    # Create figure
    fig = plt.figure(figsize=(grid_size * 2, grid_size * 2))
    fig.suptitle(f'Augmentation Variations: {class_name} (Image #{image_index})', 
                fontsize=14, fontweight='bold')
    
    # Plot original image
    plt.subplot(grid_size, grid_size, 1)
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.title('Original', fontweight='bold', color='blue')
    plt.axis('off')
    
    # Plot augmented images
    for i, aug_img in enumerate(augmented_images):
        plt.subplot(grid_size, grid_size, i + 2)
        plt.imshow(aug_img.squeeze(), cmap='gray')
        plt.title(f'Aug {i+1}', fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    filename = f'augmentation_grid_{class_name.lower().replace("/", "_")}_image_{image_index}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Grid visualization saved as '{filename}'")
    
    plt.show()

def compare_augmentation_techniques():
    """Compare different augmentation techniques side by side"""
    print("Loading Fashion MNIST dataset...")
    training_images, training_labels, class_names = load_data_and_classes()
    
    # Select a few sample images
    sample_indices = np.random.choice(len(training_images), 5, replace=False)
    
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle('Augmentation Techniques Comparison', fontsize=16, fontweight='bold')
    
    techniques = [
        ('Original', lambda x, l: (x, l)),
        ('Rotation', lambda x, l: (tf.image.rot90(x, tf.random.uniform([], 0, 4, dtype=tf.int32)), l)),
        ('Shift', lambda x, l: (tf.image.random_crop(tf.pad(x, [[2, 2], [2, 2], [0, 0]]), [28, 28, 1]), l)),
        ('Brightness', lambda x, l: (tf.image.random_brightness(x, 0.2), l)),
        ('Combined', augment_data)
    ]
    
    for i, idx in enumerate(sample_indices):
        original_img = training_images[idx]
        original_label = training_labels[idx]
        class_name = class_names[original_label]
        
        for j, (technique_name, technique_func) in enumerate(techniques):
            img_tensor = tf.cast(tf.constant(original_img), tf.float32)
            label_tensor = tf.constant(original_label)
            
            if technique_name == 'Original':
                processed_img = original_img
            else:
                processed_img, _ = technique_func(img_tensor, label_tensor)
                processed_img = processed_img.numpy()
            
            subplot_idx = i * len(techniques) + j + 1
            plt.subplot(len(sample_indices), len(techniques), subplot_idx)
            plt.imshow(processed_img.squeeze(), cmap='gray')
            
            if i == 0:  # First row - show technique names
                plt.title(technique_name, fontweight='bold')
            if j == 0:  # First column - show class names
                plt.ylabel(class_name, rotation=90, fontweight='bold')
            
            plt.axis('off')
    
    plt.tight_layout()
    
    # Save the comparison
    filename = 'augmentation_techniques_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Techniques comparison saved as '{filename}'")
    
    plt.show()

def main():
    """Main function with different visualization options"""
    print("Fashion MNIST Augmentation Visualizer")
    print("=====================================")
    print()
    print("Choose visualization type:")
    print("1. 50 original vs augmented images side by side")
    print("2. One image with 20 augmentation variations")
    print("3. Compare different augmentation techniques")
    print("4. Custom number of images")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        print("\nGenerating 50 original vs augmented images...")
        visualize_augmentation(50)
    
    elif choice == '2':
        try:
            img_idx = input("Enter image index (press Enter for random): ").strip()
            img_idx = int(img_idx) if img_idx else None
        except ValueError:
            img_idx = None
        
        print(f"\nGenerating augmentation grid...")
        visualize_augmentation_grid(img_idx, 20)
    
    elif choice == '3':
        print("\nComparing augmentation techniques...")
        compare_augmentation_techniques()
    
    elif choice == '4':
        try:
            num = int(input("Enter number of images to visualize: "))
            print(f"\nGenerating {num} original vs augmented images...")
            visualize_augmentation(num)
        except ValueError:
            print("Invalid number, using default of 50")
            visualize_augmentation(50)
    
    else:
        print("Invalid choice, showing default visualization...")
        visualize_augmentation(50)

if __name__ == "__main__":
    main()