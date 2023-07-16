# -*- coding: utf-8 -*-
import argparse
import numpy as np
import cv2
import os
import logging
import random
from pathlib import Path
from preprocessing_params import (
    CLASSES_LIST,
    DATASET_DIR,
    PREPROCESSED_DIR,
    SEQUENCE_LENGTH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)

# Scikitlearn
from sklearn.model_selection import train_test_split

# Keras
from tensorflow.keras.utils import to_categorical


# from dotenv import find_dotenv, load_dotenv

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)


def create_dataset(augmentation=False):
    """
    Returns:
        features: A list containing the extracted frames of the videos.
        labels: A list containing the indexes of the classes associated with the videos.
        video_files_paths: A list containing the paths of the videos in the disk.
    """

    # Declared Empty Lists to store the features, labels and video file path values.
    features = []
    labels = []
    video_files_paths = []

    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(CLASSES_LIST):
        # Display the name of the class whose data is being extracted.
        print(f"Extracting Data of Class: {class_name}")

        # Get the list of video files present in the specific class name directory.
        files_list = os.listdir(os.path.join(Path(DATASET_DIR) / class_name))

        # Iterate through all the files present in the files list.
        for file_name in files_list:
            # Get the complete video path.
            video_file_path = os.path.join(Path(DATASET_DIR) / class_name / file_name)

            # Extract the frames of the video file.
            frames = frames_extraction(video_file_path)

            # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
            # So ignore the vides having frames less than the SEQUENCE_LENGTH.
            if len(frames) == SEQUENCE_LENGTH:
                # Append the data to their repective lists.
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    # Converting the list to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)

    # Return the frames, class index, and video file path.
    return features, labels, video_files_paths


def frames_extraction(video_path):
    """
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    """
    # Declare a list to store video frames.
    frames_list = []
    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)
    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    # Iterate through the Video Frames.
    for frame_counter in range(SEQUENCE_LENGTH):
        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        # Reading the frame from the video.
        success, frame = video_reader.read()
        # Check if Video frame is not successfully read then break the loop
        if not success:
            break
        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        # Append the normalized frame into the frames list
        frames_list.append(normalized_frame)
    # Release the VideoCapture object.
    video_reader.release()
    # Return the frames list.
    return frames_list


# sci-kit learn functions


def onehotencode(labels):
    one_hot_encoded_labels = to_categorical(labels)

    return one_hot_encoded_labels


# Split the Data into Train ( 75% ) and Test Set ( 25% ).
def traintestsplit(features, labels, test_size, **kwargs):
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=test_size, **kwargs
    )

    logging.info("Splitting dataset into train and test splits")
    logging.info(f"Train: {len(labels_train)}")
    logging.info(f"Test: {len(labels_test)}")

    return features_train, features_test, labels_train, labels_test


def save_dataset(
    features_train,
    features_test,
    labels_train,
    labels_test,
    video_files_paths,
    output_dir,
):
    """
    Saves the dataset in the specified output directory.
    Args:
        features_train: Numpy array containing the training features.
        features_test: Numpy array containing the testing features.
        labels_train: Numpy array containing the training labels.
        labels_test: Numpy array containing the testing labels.
        video_files_paths: List containing the paths of the videos in the disk.
        output_dir: Path to the directory where the processed dataset will be saved.
    """
    # Convert output_dir to a Path object
    output_dir = Path(output_dir)

    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training and testing features as numpy arrays
    np.save(output_dir / "features_train.npy", features_train)
    np.save(output_dir / "features_test.npy", features_test)
    # Save training and testing labels as numpy arrays
    np.save(output_dir / "labels_train.npy", labels_train)
    np.save(output_dir / "labels_test.npy", labels_test)
    # Save video files paths as a text file
    with open(output_dir / "video_files_paths.txt", "w") as file:
        file.write("\n".join(video_files_paths))

    logging.info("Dataset saved successfully.")


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Create Dataset
    features, labels, video_files_paths = create_dataset()

    # One hot encoding
    labels = onehotencode(labels)

    # Train test split
    features_train, features_test, labels_train, labels_test = traintestsplit(
        features, labels, test_size=0.25, shuffle=True, random_state=seed_constant
    )

    # Save dataset
    save_dataset(
        features_train,
        features_test,
        labels_train,
        labels_test,
        video_files_paths,
        PREPROCESSED_DIR,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
