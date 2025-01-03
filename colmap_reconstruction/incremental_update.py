import os
import subprocess
import pycolmap
import tempfile
import shutil


def run_colmap_feature_extractor(database_path, image_path):
    """Run COLMAP's feature extractor for a single image."""
    # Create a temporary directory for the single image
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_image_path = os.path.join(temp_dir, os.path.basename(image_path))
        shutil.copy(image_path, temp_image_path)

        print(f"Running feature extractor for image: {image_path}")
        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", database_path,
            "--image_path", temp_dir,  # Use the temporary directory
            "--SiftExtraction.max_num_features", "50000", # Test 10000 features
            "--ImageReader.camera_model", "OPENCV" # Remove if recontruction fails
        ], check=True)


def run_colmap_matcher(database_path):
    """Run COLMAP's matcher."""
    print("Running feature matching...")
    subprocess.run([
        "colmap", "sequential_matcher",
        "--database_path", database_path,
        "--SequentialMatching.overlap", "2",
        "--SequentialMatching.quadratic_overlap", "0"
    ], check=True)

def process_image_incrementally(database_path, reconstruction_path, image_path, output_path):
    """
    Process a single image incrementally in the reconstruction using COLMAP CLI for pose estimation.
    
    Args:
        database_path (str): Path to the COLMAP database file.
        reconstruction_path (str): Path to the existing reconstruction directory.
        image_path (str): Path to the single image to process.
        output_path (str): Path to save the updated reconstruction.
    """
    # Step 1: Extract features for the new image
    run_colmap_feature_extractor(database_path, image_path)

    # Step 2: Match features between the new image and existing images
    run_colmap_matcher(database_path)

    # Step 3: Run COLMAP mapper to estimate pose and integrate the new image
    print(f"Running COLMAP mapper to integrate image: {image_path}")
    incremental_output = os.path.join(output_path, "sfm_incremental")
    os.makedirs(incremental_output, exist_ok=True)

    subprocess.run([
        "colmap", "image_registrator",
        "--database_path", database_path,
        "--input_path", reconstruction_path,  # Use the existing reconstruction as input
        "--output_path", incremental_output,
        "--Mapper.init_min_num_inliers", "30",
        "--Mapper.ba_local_max_num_iterations", "50",
        "--Mapper.ba_global_max_num_iterations", "50"
    ], check=True)

    subprocess.run([
        "colmap", "point_triangulator",
        "--image_path", os.path.dirname(image_path),
        "--database_path", database_path,
        "--input_path", incremental_output,  # Use the existing reconstruction as input
        "--output_path", incremental_output,
        "--clear_points", "0"
    ], check=True)

    # Step 4: Reload the updated reconstruction
    print("Reloading the updated reconstruction...")
    reconstruction = pycolmap.Reconstruction(incremental_output)
    print(f"Updated reconstruction loaded with {len(reconstruction.images)} images and {len(reconstruction.points3D)} points.")


def incremental_update_one_image_at_a_time(database_path, model_path, new_images_dir, output_path):
    """
    Incrementally update a COLMAP reconstruction one image at a time.
    
    Args:
        database_path (str): Path to the COLMAP database file.
        model_path (str): Path to the existing reconstruction model.
        new_images_dir (str): Directory containing new images to register.
        output_path (str): Directory to save the updated reconstruction.
    """
    print("Loading existing reconstruction...")
    reconstruction = pycolmap.Reconstruction(model_path)
    print(f"Loaded reconstruction with {len(reconstruction.images)} images.")

    # Process each new image one at a time
    for image_name in sorted(os.listdir(new_images_dir)):
        image_path = os.path.join(new_images_dir, image_name)
        if not os.path.isfile(image_path):
            continue

        print(f"Processing image: {image_name}")
        process_image_incrementally(database_path, model_path, image_path, output_path)


# Paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
database_path = os.path.join('.', "output_queries", "database.db")
model_path = os.path.join('.', "output_queries", "sfm/0")
new_images_dir = os.path.join(base_path, "samples", "queries_1/")
output_path = os.path.join('.', "output_queries")

# Perform incremental update
incremental_update_one_image_at_a_time(database_path, model_path, new_images_dir, output_path)
