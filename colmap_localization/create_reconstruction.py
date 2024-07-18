import os
import pycolmap
import glob
from PIL import Image
import utm
import numpy as np

# Set the path to the colmap executable
colmap_path = 'colmap'

# Set the path to the images directory
images_dir = '../samples/db'

# Set the path to the output directory for the reconstruction
output_dir = 'colmap_localization/reconstruction'

# Set database path
database_path = f'{output_dir}/database.db'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

### USING CLI ###
# Run colmap for reconstruction if the database and reconstruction files don't exist
if not os.path.exists(database_path):
    # Run the colmap pipeline
    os.system(f'{colmap_path} feature_extractor --database_path {database_path} --image_path {images_dir} \
            --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 1 --ImageReader.single_camera_per_folder 1')
    os.system(f'{colmap_path} spatial_matcher --database_path {database_path}')
    os.system(f'{colmap_path} mapper --database_path {database_path} --image_path {images_dir} --output_path {output_dir} --Mapper.multiple_models 0')

### USING PYTHON ###
# Alternatively, you can use pycolmap for feature extraction and mapping
# pycolmap.extract_features(database_path, images_dir)
# pycolmap.match_spatial(database_path)
# maps = pycolmap.incremental_mapping(database_path, images_dir, output_dir)

# Georeference the reconstruction
# Load the reconstruction
try:
    reconstruction = pycolmap.Reconstruction(f'{output_dir}/0')
except ValueError as e:
    print(e)
    exit(1)

# Get the images
img_names = []
print(reconstruction.summary())
for image_id, image in reconstruction.images.items():
    img_names.append(image.name)

# Function to read EXIF data and convert to UTM coordinates
def read_exif(path):
    im = Image.open(path)
    im_data = im.getxmp()['xmpmeta']['RDF']['Description']
    lat_lon = np.asarray([im_data['GpsLatitude'], im_data['GpsLongitude']]).astype(float)
    x, y, zone, zone_letter = utm.from_latlon(lat_lon[0], lat_lon[1])
    alt = float(im_data['RelativeAltitude'])
    return lat_lon[0], lat_lon[1], alt

# Create georef.txt file with image names and their positions
with open(f'{output_dir}/georef.txt', 'w') as f:
    for img_name in img_names:
        position = '{} {} {}'.format(*read_exif(os.path.join(images_dir, img_name)))
        f.write(f'{img_name} {position}\n')

# Georeference the reconstruction using model aligner and the georeference file
os.makedirs(os.path.join(output_dir, 'reconstruction_georef'), exist_ok=True)
os.system(f'{colmap_path} model_aligner --input_path {output_dir}/0 --output_path {output_dir}/reconstruction_georef --ref_images_path {output_dir}/georef.txt --alignment_max_error 1 --ref_is_gps 1')
