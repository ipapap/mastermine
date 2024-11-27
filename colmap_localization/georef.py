import os
import pycolmap
import glob
from PIL import Image
import utm
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from object_position_estimation.utils import read_image


def georef(images_dir,input_dir,output_dir,colmap_path='colmap'):
   
    # Georeference the reconstruction 

    # find all images in thecolmap database and get the gps coordinates and orientation
    # Load the database
    reconstruction = pycolmap.Reconstruction(f'{input_dir}')#/database.db')

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

    with open(f'{output_dir}/georef.txt', 'w') as f:
        for i in range(len(img_names)):
            #read exif data
            imdata=read_image(os.path.join(images_dir,img_names[i]))
            position='{} {} {}'.format(imdata['latlon'][0],imdata['latlon'][1],imdata['altitude_abs'])
            f.write(f'{img_names[i]} {position}\n')

    #
    # georeference the reconstruction using model aligner and the georeference file
    # os.makedirs(os.path.join(output_dir,'reconstruction_georef'), exist_ok=True)
    #https://colmap.github.io/faq.html#geo-registration  not working
    os.system(f'{colmap_path} model_aligner --input_path {input_dir} --output_path {output_dir} --ref_images_path {output_dir}/georef.txt --alignment_max_error 1 --ref_is_gps 1 --merge_image_and_ref_origins 1')


if __name__ == '__main__':
        
    BASE_PATH='/home/gns/Documents/terna_colmap_reconstruction/'
    colmap_path = 'colmap'

    # Set the path to the images directory
    images_dir = BASE_PATH+'imgs'#'samples/db'

    # Set the path to the output directory for the reconstruction
    output_dir = BASE_PATH+'georeferenced'

    input_dir=BASE_PATH
    #set database path
    database_path=input_dir+'database.db'