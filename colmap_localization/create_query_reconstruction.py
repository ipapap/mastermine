import os
import pycolmap
import glob
from PIL import Image
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from object_position_estimation.utils import read_image

# Set the path to the colmap executable
colmap_path = 'colmap'

def create_query_reconstruction(images_dir, output_dir):
    """
    Create a non-georeferenced COLMAP reconstruction from query images.
    
    Args:
        images_dir (str): Path to the directory containing query images
        output_dir (str): Path to the output directory for the reconstruction
    """
    # Set database path
    database_path = f'{output_dir}/database.db'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run colmap for reconstruction if the database and reconstruction files don't exist
    if not os.path.exists(f'{output_dir}/database.db'):
        # Run the colmap pipeline
        os.system(f'{colmap_path} feature_extractor --database_path {output_dir}/database.db --image_path {images_dir} \
                --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 1 --ImageReader.single_camera_per_folder 1')
        os.system(f'{colmap_path} spatial_matcher --database_path {output_dir}/database.db ')
        os.system(f'{colmap_path} mapper --database_path {output_dir}/database.db --image_path {images_dir} --output_path {output_dir} --Mapper.multiple_models 0')

    # Load the reconstruction
    reconstruction = pycolmap.Reconstruction(f'{output_dir}/0')
    print("Reconstruction summary:")
    print(reconstruction.summary())

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create a non-georeferenced COLMAP reconstruction from query images.")
    parser.add_argument('--images_dir', type=str, required=True, help='Path to the directory containing query images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory for the reconstruction')
    
    args = parser.parse_args()
    
    create_query_reconstruction(args.images_dir, args.output_dir)

if __name__ == '__main__':
    main() 