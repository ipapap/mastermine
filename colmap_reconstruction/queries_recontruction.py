import os
import subprocess
import shutil

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def run_colmap(database_path, image_path, output_path, add_new_image=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Feature extraction with enhanced parameters
    print("Running feature extraction with enhanced parameters...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--SiftExtraction.max_num_features", "20000"
    ], check=True)

    # Perform exhaustive or sequential matching with adjusted parameters
    matcher_type = "exhaustive_matcher" if not add_new_image else "sequential_matcher"
    print(f"Running {matcher_type} with enhanced parameters...")
    subprocess.run([
        "colmap", matcher_type,
        "--database_path", database_path,
        "--SiftMatching.max_distance", "8.0",
        "--SiftMatching.max_num_matches", "10000"
    ], check=True)

    # Run SfM to generate a sparse 3D reconstruction
    print("Running structure from motion...")
    sfm_output = os.path.join(output_path, "sfm")
    if not os.path.exists(sfm_output):
        os.makedirs(sfm_output)
    
    subprocess.run([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_path,
        "--output_path", sfm_output,
        "--Mapper.init_min_num_inliers", "20",
        "--Mapper.ba_global_max_num_iterations", "100"
    ], check=True)

    sfm_subdir = os.path.join(sfm_output, "0") if os.path.exists(os.path.join(sfm_output, "0")) else sfm_output
    
    subprocess.run([
        "colmap", "model_converter",
        "--input_path", sfm_subdir,
        "--output_path", sfm_subdir,
        "--output_type", "TXT"
    ], check=True)

    # Check if SfM output is in `sfm/0` or directly in `sfm`
    required_files = ["cameras.txt", "images.txt", "points3D.txt"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(sfm_subdir, f))]
    
    if missing_files:
        print("Error: Missing SfM output files:", missing_files)
        return
   

    print("Reconstruction saved to:", output_path)

def add_new_image(database_path, image_path, new_image_path, output_path):
    for filename in os.listdir(new_image_path):
        src = os.path.join(new_image_path, filename)
        dst = os.path.join(image_path, filename)
        shutil.copy(src, dst)
    
    run_colmap(database_path, image_path, output_path, add_new_image=True)

database_path = os.path.join('output_queries/', "database.db")
initial_image_path = os.path.join(base_path, "samples/queries")
new_image_path = os.path.join(base_path, "samples/new_entries")
output_path = os.path.join(os.path.dirname(__file__), "output_queries")

# Initial reconstruction with the first 3-4 images
run_colmap(database_path, initial_image_path, output_path)

# Add images and run incremental matching to find relation
add_new_image(database_path, initial_image_path, new_image_path, output_path)
