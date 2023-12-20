import os
import shutil
from sklearn.model_selection import train_test_split


def create_dataset_splits(source_dir, target_dir, train_size=0.7, validation_size=0.15, test_size=0.15):
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
    - source_dir: Path to the source directory containing class folders.
    - target_dir: Path to the target directory where splits will be stored.
    - train_size: Size of training data split.
    - validation_size: Size of validation data split.
    - test_size: Size of test data split.
    """

    # Check if percentages sum up to 1
    if train_size + validation_size + test_size != 1:
        raise ValueError("The sum of train_size, validation_size, and test_size must be 1.")

    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)

        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue

        # Collect all file names in the current class directory
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        # Split data
        train_files, test_files = train_test_split(files, test_size=test_size + validation_size, random_state=42)
        validation_files, test_files = train_test_split(test_files, test_size=test_size / (test_size + validation_size),
                                                        random_state=42)

        # Function to copy files to the target directory
        def copy_files(files, dataset_type):
            dest_dir = os.path.join(target_dir, dataset_type, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for file in files:
                shutil.copy(os.path.join(class_dir, file), dest_dir)

        # Copy files according to the split
        copy_files(train_files, 'train')
        copy_files(validation_files, 'validation')
        copy_files(test_files, 'test')


if __name__ == "__main__":
    source_dir = "D:\\Nepali Sign Language Images"
    target_dir = "C:\\Nepali-Sign-Language-Recognition\\Nepali Sign Language Images"
    create_dataset_splits(source_dir, target_dir)