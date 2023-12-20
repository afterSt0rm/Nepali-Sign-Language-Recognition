import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


def show_images(target_dir, class_name):
    """
    Shows 3 random images from `class_name` folder in `target_dir` folder.

    Parameters:
    - target_dir: Path to the parent directory that contains class_name folders.
    - class_name: Name of the folder where images of a particular class are present.
    """
    target_path = target_dir + class_name
    file_names = os.listdir(target_path)
    target_images = random.sample(file_names, 3)

    # Plot images
    plt.figure(figsize=(16, 10))
    for i, img in enumerate(target_images):
        img_path = target_path + "\\" + img
        plt.subplot(1, 3, i + 1)
        plt.imshow(mpimg.imread(img_path))
        plt.title(class_name)
        plt.axis("off")


def plot_curves(history):
    """
    Plots training, validation accuracy and loss curves.

    Parameters:
    - history: TensorFlow model History object
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()