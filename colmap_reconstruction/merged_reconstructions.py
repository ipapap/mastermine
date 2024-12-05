import os
import numpy as np

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to 3x3 rotation matrix."""
    q = np.array(q)
    q /= np.linalg.norm(q)  # Normalize quaternion
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])

def transform_points(points, rotation, translation):
    """Apply rotation and translation to 3D points."""
    return [rotation @ np.array(pt) + translation for pt in points]

def load_points3D_file(filepath):
    points3D = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            error = float(parts[7])
            track_data = []
            for i in range(8, len(parts), 2):
                image_id, point2D_idx = map(int, parts[i:i+2])
                track_data.append((image_id, point2D_idx))
            points3D[point_id] = {
                "xyz": (x, y, z),
                "color": (r, g, b),
                "error": error,
                "track": track_data,
                "line": line.strip()
            }
    return points3D

def load_images_file(filepath):
    images = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#") or not line:
                i += 1
                continue
            
            parts = line.split()
            try:
                image_id = int(parts[0])
                qvec = tuple(map(float, parts[1:5]))
                tvec = tuple(map(float, parts[5:8]))
                camera_id = int(parts[8])
                name = parts[9]

                i += 1
                points2D = []
                if i < len(lines):
                    points_line = lines[i].strip()
                    if points_line and not points_line.startswith("#"):
                        point_parts = points_line.split()
                        for j in range(0, len(point_parts), 3):
                            x = float(point_parts[j])
                            y = float(point_parts[j+1])
                            point3D_id = int(point_parts[j+2])
                            points2D.append((x, y, point3D_id))
                
                images[image_id] = {
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": name,
                    "points2D": points2D,
                    "line": line
                }
            except (ValueError, IndexError) as e:
                print(f"Skipping malformed line in images.txt: {line} - Error: {e}")
            i += 1
    return images

def load_cameras_file(filepath):
    cameras = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = tuple(map(float, parts[4:]))
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
                "line": line.strip()
            }
    return cameras

def merge_reconstructions(geo_model_path, non_geo_model_path, common_image_name, output_path):
    os.makedirs(output_path, exist_ok=True)

    geo_points3D = load_points3D_file(os.path.join(geo_model_path, "points3D.txt"))
    geo_images = load_images_file(os.path.join(geo_model_path, "images.txt"))
    geo_cameras = load_cameras_file(os.path.join(geo_model_path, "cameras.txt"))

    non_geo_points3D = load_points3D_file(os.path.join(non_geo_model_path, "points3D.txt"))
    non_geo_images = load_images_file(os.path.join(non_geo_model_path, "images.txt"))
    non_geo_cameras = load_cameras_file(os.path.join(non_geo_model_path, "cameras.txt"))

    geo_image = next(img for img in geo_images.values() if img["name"] == common_image_name)
    non_geo_image = next((img for img in non_geo_images.values() if img["name"] == common_image_name), None)
    if non_geo_image is None:
        raise ValueError(f"Image '{common_image_name}' not found in the non-georeferenced dataset.")

    geo_rotation = quaternion_to_rotation_matrix(geo_image["qvec"])
    non_geo_rotation = quaternion_to_rotation_matrix(non_geo_image["qvec"])

    rotation = geo_rotation @ non_geo_rotation.T
    translation = np.array(geo_image["tvec"]) - rotation @ np.array(non_geo_image["tvec"])

    merged_cameras = {**geo_cameras, **non_geo_cameras}
    with open(os.path.join(output_path, "cameras.txt"), 'w') as out_f:
        for camera in merged_cameras.values():
            out_f.write(camera["line"] + "\n")

    merged_images = geo_images.copy()
    max_image_id = max(merged_images.keys()) if merged_images else 0
    for img_id, img_data in non_geo_images.items():
        if img_data["name"] == common_image_name:
            continue
        new_id = max_image_id + 1
        img_data["tvec"] = tuple(rotation @ np.array(img_data["tvec"]) + translation)
        img_data["line"] = f"{new_id} {' '.join(map(str, img_data['qvec']))} {' '.join(map(str, img_data['tvec']))} {img_data['camera_id']} {img_data['name']}"
        merged_images[new_id] = img_data
        max_image_id = new_id
    with open(os.path.join(output_path, "images.txt"), 'w') as out_f:
        for image in merged_images.values():
            out_f.write(image["line"] + "\n")

    merged_points3D = geo_points3D.copy()
    max_point_id = max(merged_points3D.keys()) if merged_points3D else 0
    for point_id, pt_data in non_geo_points3D.items():
        new_id = max_point_id + 1
        pt_data["xyz"] = tuple(rotation @ np.array(pt_data["xyz"]) + translation)
        pt_data["line"] = f"{new_id} {' '.join(map(str, pt_data['xyz']))} {' '.join(map(str, pt_data['color']))} {pt_data['error']} {' '.join(f'{x[0]} {x[1]}' for x in pt_data['track'])}"
        merged_points3D[new_id] = pt_data
        max_point_id = new_id
    with open(os.path.join(output_path, "points3D.txt"), 'w') as out_f:
        for point in merged_points3D.values():
            out_f.write(point["line"] + "\n")

    print(f"Merged reconstruction saved in: {output_path}")

# Paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
geo_model_path = os.path.join('.', "output_georef/aligned_model")
non_geo_model_path = os.path.join('.', "output_queries/sfm/0")
output_path = os.path.join('.', "output_combined_model")
common_image_name = "DJI_20240703154815_0007_W.JPG"

# Run merging process
merge_reconstructions(geo_model_path, non_geo_model_path, common_image_name, output_path)
