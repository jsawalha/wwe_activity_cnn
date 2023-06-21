"""
visualize.py
------------
Allows you to visualize the data with it's labels. 
TODO: WILL ALSO CONTAIN OTHER FUNCTIONS TO VIEW ML PREPROCESSING STEPS?
TODO: ADD ARGS PARSE FOR THIS
"""

# OS modules
from pathlib import Path
import os
import cv2
import random
import numpy as np
import datetime as dt
from collections import deque
import matplotlib.pyplot as plt

FIGSIZE = (20, 20)
CLASS_PATH = Path(Path("/home") / "jeff" / "Downloads" / "wwe")


# Create a Matplotlib figure and specify the size of the figure.
plt.figure(figsize=FIGSIZE)

# Get the names of all classes/categories in UCF50.
all_classes_names = [
    item for item in os.listdir(CLASS_PATH) if os.path.isdir(CLASS_PATH / item)
]


def displayRawImages(class_path, class_names):
    # Generate a list of 20 random values. The values will be between 0-50,
    # where 50 is the total number of class in the dataset.
    random_range = random.sample(range(len(class_names)), 3)

    # Iterating through all the generated random values.
    for counter, random_index in enumerate(random_range, 1):
        # Retrieve a Class Name using the Random Index.
        selected_class_Name = class_names[random_index]

        # Retrieve the list of all the video files present in the randomly selected Class Directory.
        video_files_names_list = os.listdir(Path(class_path / selected_class_Name))

        # Randomly select a video file from the list retrieved from the randomly selected Class Directory.
        selected_video_file_name = random.choice(video_files_names_list)

        # Initialize a VideoCapture object to read from the video File.
        video_reader = cv2.VideoCapture(
            str(Path(class_path / selected_class_Name / selected_video_file_name))
        )

        # Read the first frame of the video file.
        _, bgr_frame = video_reader.read()

        # Release the VideoCapture object.
        video_reader.release()

        # Convert the frame from BGR into RGB format.  cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB) is used to convert the captured video frame from the
        # BGR (Blue-Green-Red) color space to the RGB (Red-Green-Blue) color space. The reason for this conversion is
        # that Matplotlib expects images in RGB format to display them correctly.
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Write the class name on the video frame.
        cv2.putText(
            rgb_frame,
            selected_class_Name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Display the frame.
        plt.subplot(1, 3, counter)
        plt.imshow(rgb_frame)
        plt.axis("off")
    plt.show()


displayRawImages(CLASS_PATH, all_classes_names)
