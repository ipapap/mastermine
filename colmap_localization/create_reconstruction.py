import os
import pycolmap
import glob
from PIL import Image
import utm
import numpy as np
# # Set the path to the colmap executable
colmap_path = 'colmap'

# Set the path to the images directory
images_dir = '/media/gns/backup/duth@terna/db'#'samples/db'

# Set the path to the output directory for the reconstruction
output_dir = 'colmap_localization/reconstruction'

#set database path
database_path=f'{output_dir}/database.db'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


### USING CLI ###
# Run colmap for reconstruction if the database and reconstruction files don't exist
if not os.path.exists(f'{output_dir}/database.db') :#or not os.path.exists(f'{output_dir}/reconstruction'):
    # Run the colmap pipeline
    os.system(f'{colmap_path} feature_extractor --database_path {output_dir}/database.db --image_path {images_dir} \
            --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 1 --ImageReader.single_camera_per_folder 1')
    os.system(f'{colmap_path} spatial_matcher --database_path {output_dir}/database.db ')
    os.system(f'{colmap_path} mapper --database_path {output_dir}/database.db --image_path {images_dir} --output_path {output_dir} --Mapper.multiple_models 0')
    # if --Mapper.multiple_models 1 then the output will be a folder with multiple reconstructions and then mergins is needed :
    # #check how many reconstructions are there
    # maps = [name for name in glob.glob(os.path.join(output_dir, '*/'))]
    # if len(maps)>1:
    #     print('There are multiple reconstructions')
    #     #merge the reconstructions
    #     os.system(f'{colmap_path} model_merger --input_path1 {maps[0]} --input_path2 {maps[1]} --output_path {output_dir}/reconstruction')

### USING PYTHON ###
# pycolmap.extract_features(database_path, images_dir)
# pycolmap.match_spatial(database_path)
# maps = pycolmap.incremental_mapping(database_path, images_dir, output_dir, )
# # Merge the maps if there are multiple
# if len(maps) > 0:
#     # maps=pycolmap.merge_maps(*maps) # not correct !!!
#     rec2_from_rec1 = pycolmap.align_reconstructions_via_reprojections(maps[0],maps[0])
#     maps[0].transform(rec2_from_rec1) #not sure if this is correct

# # Write the reconstruction to disk
# # maps[0].write(output_dir)



# Georeference the reconstruction 

# find all images in thecolmap database and get the gps coordinates and orientation
# Load the database
reconstruction = pycolmap.Reconstruction(f'{output_dir}/0')#/database.db')

# Get the images
# img_paths=[]
img_names=[]
print(reconstruction.summary())
for image_id, image in reconstruction.images.items():
    print(image_id, image)
    # Get the image path
    img_names.append(image.name)
    # img_paths.append(os.path.join(images_dir,image.name))

"""
create a text file with the following format:
image_name1.jpg X1 Y1 Z1
image_name2.jpg X2 Y2 Z2
image_name3.jpg X3 Y3 Z3
"""
#write the file
def read_exif(path):
    im=Image.open(path)
    im_data=im.getxmp()['xmpmeta']['RDF']['Description']
    # rpy=np.asarray([im_data['FlightYawDegree'],im_data['FlightRollDegree'],im_data['FlightPitchDegree']]).astype(float)
    lat_lon=np.asarray([im_data['GpsLatitude'],im_data['GpsLongtitude']]).astype(float)
    x,y,zone,zone_letter=utm.from_latlon(lat_lon[0],lat_lon[1])
    alt=float(im_data['RelativeAltitude'])
    return(lat_lon[0],lat_lon[1],alt)

with open(f'{output_dir}/georef.txt', 'w') as f:
    for i in range(len(img_names)):
        #read exif data
        position='{} {} {}'.format(*read_exif(os.path.join(images_dir,img_names[i])))
        f.write(f'{img_names[i]} {position}\n')

#
# georeference the reconstruction using model aligner and the georeference file
os.makedirs(os.path.join(output_dir,'reconstruction_georef'), exist_ok=True)
#https://colmap.github.io/faq.html#geo-registration  not working
# os.system(f'{colmap_path} model_aligner --input_path {output_dir}/0 --output_path {output_dir}/reconstruction_georef --ref_images_path {output_dir}/georef.txt --robust_alignment_max_error 1 --robust_alignment 1')

os.system(f'{colmap_path} model_aligner --input_path {output_dir}/0 --output_path {output_dir}/reconstruction_georef --ref_images_path {output_dir}/georef.txt --alignment_max_error 1 --ref_is_gps 1 --merge_image_and_ref_origins 1')
# read all images and get the gps coordinates and orientation
