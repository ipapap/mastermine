import sys
import os

from object_detection.predict import detect_objects
from object_position_estimation.main import estimate_object_positions

sys.path.append('estimate_object_positions')

import utils

# Main function to unify the process
def main(image_path, model_path, las_file_path):
    detected_info = detect_objects(image_path, model_path)

    estimate_object_positions(detected_info, image_path, las_file_path)

if __name__ == '__main__':
    # Define paths
    image_path = 'samples/queries/DJI_20240703154815_0007_W.JPG'  # Image for object detection
    model_path = 'object_detection/weights/best.pt'  # YOLO model path
    las_file_path = '/home/gns/Downloads/Mpompakas_3_7_24_DUTH_LAS_0.1m.las'  # LAS file path for point cloud

    main(image_path, model_path, las_file_path)
