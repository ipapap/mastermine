import os
import subprocess

def initial_reconstruction(database_path, queries_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("Running feature extraction for initial reconstruction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", queries_path,
        "--SiftExtraction.max_num_features", "50000"
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
        "--Mapper.init_min_num_inliers", "2",
        "--Mapper.ba_local_max_num_iterations", "50",
        "--Mapper.ba_global_max_num_iterations", "50"
    ], check=True)

    print("Initial reconstruction completed and saved to:", sfm_output)

# Paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
database_path = os.path.join('.', "output_queries", "database.db")
queries_path = os.path.join(base_path, "samples", "queries")
output_path = os.path.join('.', "output_queries")

initial_reconstruction(database_path, queries_path, output_path)
