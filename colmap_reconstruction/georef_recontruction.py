import os
import subprocess
from exif import Image
from pyproj import Proj, transform, CRS

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def gps_to_ecef(lat, lon, alt):
    """
    Convert GPS coordinates (latitude, longitude, altitude) to ECEF.
    """
    wgs84 = CRS("EPSG:4326")  # WGS-84 geographic coordinate system
    ecef = CRS("EPSG:4978")  # Earth-Centered, Earth-Fixed (ECEF)
    x, y, z = transform(wgs84, ecef, lon, lat, alt)
    return x, y, z

def debug_aligned_coordinates(ref_images_path, aligned_images_path):
    """
    Compare reference GPS coordinates with aligned reconstructed coordinates.
    """
    print("\nDebugging alignment transformation:")

    # Load reference GPS coordinates from ref_images.txt
    gps_coords = {}
    with open(ref_images_path, 'r') as ref_file:
        for line in ref_file:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            image_name = parts[0]
            lat, lon, alt = map(float, parts[1:])
            gps_coords[image_name] = gps_to_ecef(lat, lon, alt)

    # Load reconstructed image coordinates from images.txt
    reconstructed_coords = {}
    with open(aligned_images_path, 'r') as aligned_file:
        for line in aligned_file:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            image_name = parts[-1]
            tx, ty, tz = map(float, parts[5:8])
            reconstructed_coords[image_name] = (tx, ty, tz)

    # Compare and print differences
    for image_name, gps_ecef in gps_coords.items():
        if image_name in reconstructed_coords:
            tx, ty, tz = reconstructed_coords[image_name]
            print(f"Image: {image_name}")
            print(f"  GPS (ECEF): {gps_ecef}")
            print(f"  Reconstructed: ({tx}, {ty}, {tz})")
            diff = (gps_ecef[0] - tx, gps_ecef[1] - ty, gps_ecef[2] - tz)
            print(f"  Difference: {diff}\n")
        else:
            print(f"Image: {image_name} not found in reconstructed coordinates.")

def create_ref_images_file(image_dir, ref_images_path):
    with open(ref_images_path, 'w') as ref_file:
        ref_file.write("# IMAGE_NAME LATITUDE LONGITUDE ALTITUDE\n")
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_dir, filename)
                with open(image_path, 'rb') as img_file:
                    img = Image(img_file)
                    if img.has_exif and img.gps_latitude and img.gps_longitude and img.gps_altitude:
                        # Convert GPS coordinates to decimal
                        lat_decimal = img.gps_latitude[0] + img.gps_latitude[1] / 60 + img.gps_latitude[2] / 3600
                        lon_decimal = img.gps_longitude[0] + img.gps_longitude[1] / 60 + img.gps_longitude[2] / 3600
                        if img.gps_latitude_ref == "S":
                            lat_decimal = -lat_decimal
                        if img.gps_longitude_ref == "W":
                            lon_decimal = -lon_decimal
                        ref_file.write(f"{filename} {lat_decimal} {lon_decimal} {img.gps_altitude}\n")
                    else:
                        print(f"Warning: No GPS data found for {filename}")

def run_command(command, description):
    """Helper function to run a subprocess command with error handling."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"{description} succeeded.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"{description} failed.")
        print(e.stdout)
        print(e.stderr)
        raise

def convert_bin_to_txt(aligned_output):
    """Convert binary files to text format for inspection."""
    print("Converting binary aligned model files to text format...")
    try:
        run_command([
            "colmap", "model_converter",
            "--input_path", aligned_output,
            "--output_path", aligned_output,
            "--output_type", "TXT"
        ], "Model conversion to text")
    except Exception as e:
        print(f"Error during model conversion: {e}")

def verify_sfm_output(sfm_output):
    """Check if the necessary SfM output files exist."""
    sfm_subdir = os.path.join(sfm_output, "0")
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(sfm_subdir, f))]
    if missing_files:
        raise FileNotFoundError(f"SfM output files missing: {', '.join(missing_files)} in {sfm_subdir}")

def run_georeferenced_reconstruction(database_path, image_path, output_path, ref_images_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("Creating reference file for GPS data...")
    create_ref_images_file(image_path, ref_images_path)

    print("Running feature extraction...")
    run_command([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_num_features", "20000"
    ], "Feature extraction")

    print("Running exhaustive feature matching...")
    run_command([
        "colmap", "exhaustive_matcher",
        "--database_path", database_path
    ], "Exhaustive feature matching")

    print("Running structure from motion...")
    sfm_output = os.path.join(output_path, "sfm")
    if not os.path.exists(sfm_output):
        os.makedirs(sfm_output)

    run_command([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_path,
        "--output_path", sfm_output,
        "--Mapper.init_min_num_inliers", "20"
    ], "Structure-from-Motion")

    try:
        verify_sfm_output(sfm_output)
    except FileNotFoundError as e:
        print(e)
        return
    print("Aligning model with GPS data...")
    aligned_output = os.path.join(output_path, "aligned_model")
    if not os.path.exists(aligned_output):
        os.makedirs(aligned_output)

    try:
        run_command([
            "colmap", "model_aligner",
            "--input_path", os.path.join(sfm_output, "0"),
            "--output_path", aligned_output,
            "--ref_images_path", ref_images_path,
            "--robust_alignment", "1",
            "--robust_alignment_max_error", "0.1"
        ], "Model alignment")
        
        # Convert binary aligned model to text format
        convert_bin_to_txt(aligned_output)
        
        # Debug alignment transformation
        debug_aligned_coordinates(ref_images_path, os.path.join(aligned_output, "images.txt"))
    except FileNotFoundError:
        print("Error: Alignment failed. Ensure GPS data in ref_images.txt is correct.")
        return

    print("Printing coordinates of reconstructed images:")
    images_file = os.path.join(aligned_output, "images.txt")
    if os.path.exists(images_file):
        with open(images_file, 'r') as file:
            for line in file:
                if not line.startswith("#") and line.strip():
                    parts = line.split()
                    if len(parts) > 4:
                        print(f"Image: {parts[-1]}, Coordinates (tx, ty, tz): ({parts[5]}, {parts[6]}, {parts[7]})")
    else:
        print("Error: Aligned images file not found. Check if alignment was successful.")

# Define paths
database_path = os.path.join(base_path, "database_georef.db")
image_path = os.path.join(base_path, "samples/db")
output_path = os.path.join(os.path.dirname(__file__), "output_georef")
ref_images_path = os.path.join(output_path, "ref_images.txt")

# Run reconstruction
run_georeferenced_reconstruction(database_path, image_path, output_path, ref_images_path)
