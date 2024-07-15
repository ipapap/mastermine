import os
import pycolmap
# # Set the path to the colmap executable
colmap_path = 'colmap'

# Set the path to the images directory
images_dir = '/media/gns/8E82C78582C76FEF/Users/johnp/Documents/duth@terna/sim2'

# Set the path to the output directory for the reconstruction
output_dir = '/media/gns/8E82C78582C76FEF/Users/johnp/Documents/duth@terna/colmap'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Run colmap for reconstruction if the database and reconstruction files don't exist
if not os.path.exists(f'{output_dir}/database.db') or not os.path.exists(f'{output_dir}/reconstruction'):
    # Run the colmap pipeline
    os.system(f'{colmap_path} feature_extractor --database_path {output_dir}/database.db --image_path {images_dir} \
            --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 1 --ImageReader.single_camera_per_folder 1')
    os.system(f'{colmap_path} spatial_matcher --database_path {output_dir}/database.db ')
    # os.system(f'{colmap_path} mapper --database_path {output_dir}/database.db --image_path {images_dir} --output_path {output_dir}/reconstruction ')

database_path=f'{output_dir}/database.db'
# pycolmap.extract_features(database_path, images_dir)
# pycolmap.match_spatial(database_path)
maps = pycolmap.incremental_mapping(database_path, images_dir, output_dir, )
maps[0].write(output_dir)

#georeference the reconstruction using model aligner and the georeference file
# os.system(f'{colmap_path} model_aligner --input_path {output_dir}/reconstruction --output_path {output_dir}/reconstruction_georef --ref_images_path {images_dir} --robust_alignment_max_error 1')
#read all images and get the gps coordinates and orientation

# find all images in thecolmap database
import pycolmap

# Load the database
database = pycolmap.Reconstruction(f'{output_dir}/')#/database.db')

# Get the images
reconstruction = pycolmap.Reconstruction("path/to/reconstruction/dir")
print(reconstruction.summary())
for image_id, image in reconstruction.images.items():
    print(image_id, image)
    # Get the image path
    image_path = image.name