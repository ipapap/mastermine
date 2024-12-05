import os
import subprocess


def convert_bin_to_txt(aligned_output):
    """Convert binary files to text format for inspection."""
    print("Converting binary aligned model files to text format...")

    # Iterate over all subdirectories
    for subdir in os.listdir(aligned_output):
        model_dir = os.path.join(aligned_output, subdir)
        if not os.path.isdir(model_dir):
            continue

        required_files = ['cameras.bin', 'images.bin', 'points3D.bin']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        if missing_files:
            print(f"Warning: Missing model files in {model_dir}: {missing_files}")
            continue

        try:
            subprocess.run([
                "colmap", "model_converter",
                "--input_path", model_dir,
                "--output_path", model_dir,
                "--output_type", "TXT"
            ], check=True)
            print(f"Model in {model_dir} converted to TXT format.")
        except subprocess.CalledProcessError as e:
            print(f"Error during model conversion in {model_dir}: {e}")


def incremental_update(database_path, queries_path, new_image_path, output_path):
    print("Running feature extraction for new images...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", new_image_path,
        "--SiftExtraction.max_num_features", "50000"
    ], check=True)

    print("Running exhaustive matching for all images...")
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", database_path,
        "--SiftMatching.max_distance", "25.0",
        "--SiftMatching.max_num_matches", "100000"
    ], check=True)

    print("Running mapper to integrate new images...")
    sfm_output = os.path.join(output_path, "sfm_incremental")
    if not os.path.exists(sfm_output):
        os.makedirs(sfm_output)

    # Combine the original queries and new images
    combined_image_path = os.path.join(output_path, "combined_images")
    if not os.path.exists(combined_image_path):
        os.makedirs(combined_image_path)

    for image_dir in [queries_path, new_image_path]:
        for filename in os.listdir(image_dir):
            src = os.path.join(image_dir, filename)
            dst = os.path.join(combined_image_path, filename)
            if not os.path.exists(dst):
                os.symlink(src, dst)  # Use symbolic links to avoid duplicating files

    subprocess.run([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", combined_image_path,
        "--output_path", sfm_output,
        "--Mapper.init_min_num_inliers", "2",
        "--Mapper.ba_local_max_num_iterations", "50",
        "--Mapper.ba_global_max_num_iterations", "50"
            ], check=True)

    print("Incremental update completed. Updated model saved to:", sfm_output)
    convert_bin_to_txt(sfm_output)

# Paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
database_path = os.path.join('.', "output_queries", "database.db")
model_path = os.path.join('.', "output_queries", "sfm")
new_image_path = os.path.join(base_path, "samples", "new_entries")
output_path = os.path.join('.', "output_queries")

incremental_update(database_path, model_path, new_image_path, output_path)
