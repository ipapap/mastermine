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


def initial_reconstruction(database_path, queries_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("Running feature extraction for initial reconstruction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", queries_path,
        "--SiftExtraction.max_num_features", "50000",
        "--ImageReader.camera_model", "OPENCV"
    ], check=True)

    print("Running exhaustive matching for initial reconstruction...")
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", database_path,
        "--SiftMatching.max_distance", "25.0",
        "--SiftMatching.max_num_matches", "100000"
    ], check=True)

    print("Running structure-from-motion...")
    sfm_output = os.path.join(output_path, "sfm")
    if not os.path.exists(sfm_output):
        os.makedirs(sfm_output)

    subprocess.run([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", queries_path,
        "--output_path", sfm_output,
        "--Mapper.init_min_num_inliers", "30",
        "--Mapper.ba_local_max_num_iterations", "50",
        "--Mapper.ba_global_max_num_iterations", "50",
        "--Mapper.multiple_models", "0"
    ], check=True)

    print("Initial reconstruction completed and saved to:", sfm_output)
    convert_bin_to_txt(sfm_output)
# Paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
database_path = os.path.join('.', "output_queries", "database.db")
queries_path = os.path.join(base_path, "samples", "queries_1")
output_path = os.path.join('.', "output_queries")

initial_reconstruction(database_path, queries_path, output_path)
