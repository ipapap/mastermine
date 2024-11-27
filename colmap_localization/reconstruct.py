import exifread
from PIL import Image
import os
import numpy as np
import cv2
from database import COLMAPDatabase
import glob
from geopy.distance import geodesic
import faiss
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utm
from object_position_estimation.utils import read_image
from georef import georef
# from create_reconstruction import *
import pycolmap
import shutil
def get_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(20000)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    return keypoints, descriptors

# def get_matches(features1, features2):
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(features1, features2, k=2)
#     match_idx1, match_idx2 = [], []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             match_idx1.append(m.queryIdx)
#             match_idx2.append(m.trainIdx)
        
#     matches = np.array([match_idx1, match_idx2]).T
#     return matches

def get_matches(features1, features2):
    # Use Faiss for exact nearest neighbors using IndexFlatL2
    d = features1.shape[1]  # Dimension of descriptors

    # Step 1: Initialize the IndexFlatL2 for exact search
    index = faiss.IndexFlatL2(d)  # No need for training, exact exhaustive search

    # Step 2: Add features2 to the index
    index.add(features2.astype('float32'))

    # Step 3: Perform search to find k nearest neighbors for each descriptor in features1
    k = 2  # Number of nearest neighbors to find
    distances, indices = index.search(features1.astype('float32'), k)

    # Step 4: Apply Lowe's ratio test to filter matches
    ratio_threshold = 0.75
    match_idx1, match_idx2 = [], []
    for i, (dist1, dist2) in enumerate(distances):
        if dist1 < ratio_threshold * dist2:  # Lowe's ratio test
            match_idx1.append(i)
            match_idx2.append(indices[i][0])

    matches = np.array([match_idx1, match_idx2]).T
    return matches

def extract_camera_parameters(image_path):
    # Open image to read EXIF data
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)

    # Extract relevant EXIF tags
    # focal_length_tag = tags.get('EXIF FocalLength')
    # sensor_width_tag = tags.get('EXIF SensorWidth')
    gps_latitude_tag = tags.get('GPS GPSLatitude')
    gps_longitude_tag = tags.get('GPS GPSLongitude')
    gps_latitude_ref = tags.get('GPS GPSLatitudeRef')
    gps_longitude_ref = tags.get('GPS GPSLongitudeRef')
    

    # Extract image size
    with Image.open(image_path) as img:
        width, height = img.size

    im_data=read_image(image_path)
    focal_length_pixels= im_data['K'][0,0]
    principal_point = (im_data['K'][0,2], im_data['K'][1,2])


    latitude, longitude = im_data['latlon']

    # Return the camera parameters and GPS coordinates
    return {
        'model': pycolmap.CameraModelId(2),  # Assuming  camera model (simple radial)
        'width': width,
        'height': height,
        'params': np.array([focal_length_pixels, principal_point[0], principal_point[1]]),
        'gps': (latitude, longitude)
    }

def create_database(database_path, images_path):
    # Create a new database
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    # db = pycolmap.Database(database_path)

    # Get list of images
    images = glob.glob(os.path.join(images_path, '*.JPG'))

    # Create a dictionary to store camera parameters and IDs
    cameras = {}

    # Loop over images and extract camera parameters
    image_ids = []
    image_locations = []
    for i, image_path in enumerate(images):
        # Extract camera parameters
        camera_params = extract_camera_parameters(image_path)

        # Create a key based on camera parameters to check if it already exists
        camera_key = (camera_params['model'], camera_params['width'], camera_params['height'], tuple(camera_params['params']))
        
        if camera_key not in cameras:
            # Add new camera to database
            camera_id=db.add_camera(model=2, width=camera_params['width'], height=camera_params['height'], params=tuple(camera_params['params'].tolist()+[0]))
            
            cameras[camera_key] = camera_id
            print(f'Added new camera: ID {camera_id}')
        else:
            # Use existing camera ID
            camera_id = cameras[camera_key]
            print(f'Reused existing camera: ID {camera_id}')

        # Generate features
        keypoints, descriptors = get_features(image_path)

        # Add image and keypoints to database
        image_name = os.path.basename(image_path)

        image_id = db.add_image(image_name, camera_id)
        db.add_keypoints(image_id, keypoints)
        db.add_descriptors(image_id ,descriptors)
        position= utm.from_latlon(*camera_params['gps'])[:2]
        # db.add_pose_prior(image_id,[position[0],position[1]])
        image_ids.append((image_id, descriptors))
        image_locations.append((image_id, camera_params['gps']))

    # Generate two view geometries for nearby image pairs
    for i in range(len(image_locations)):
        for j in range(i + 1, len(image_locations)):
            image_id1, gps1 = image_locations[i]
            image_id2, gps2 = image_locations[j]

            if gps1 is None or gps2 is None:
                continue

            distance = geodesic(gps1, gps2).meters
            if distance <= 100:  # Match only if images are within 100 meters of each other
                _, des1 = image_ids[i]
                _, des2 = image_ids[j]
                matches = get_matches(des1, des2)

                # Add matches and two view geometry to the database
                db.add_two_view_geometry(image_id1, image_id2, matches)
                db.add_matches(image_id1, image_id2, matches)

    # Commit the data to the file
    db.commit()
    db.close()

    return db

# def reconstruct(database_path,images_path):
#     db = pycolmap.Database(database_path)
    





if __name__ == '__main__':
    # Create a new database
    colmap_path='colmap'
    base_path= 'colmap_localization/reconstruction/'
    database_path = base_path+'database.db'
    images_path = '/home/gns/dev/mastermine/samples/db/'


    input_path = base_path+'0'
    output_dir= base_path + 'georeferenced'


    if os.path.exists(base_path):
        shutil.rmtree(base_path)
        os.mkdir(base_path)

    # if os.path.exists(database_path):
    #     # exit('Database already exists')
    #     os.remove(database_path)
    #     pass

    db = create_database(database_path, images_path)
    # reconstruct(database_path, images_path)

    # os.system(f'{colmap_path} spatial_matcher --database_path {base_path}/database.db ')
    os.system(f'{colmap_path} mapper --database_path {database_path} --image_path {images_path} --output_path {base_path} --Mapper.multiple_models 0')
    
    
    os.makedirs(output_dir, exist_ok=True)
    georef(images_path,input_path,output_dir)

    print('Database created at:', database_path)
