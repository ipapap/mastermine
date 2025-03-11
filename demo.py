import object_position_estimation as ope
import object_detection
from  visualization.visualize import *
# import colmap_localization.localize as loc
from PIL import Image
import glob
import os
import numpy as np
import argparse
import threading



def load_image(image_path):
    image = Image.open(os.path.join(image_path))
    return image

def localize(args):
    frames,frames_descriptors=loc.build_database(args.reconstruction_path)
    image_paths = [frame.base_path + frame.image.name for frame in frames]
    
    



def main():
    
    parser = argparse.ArgumentParser(description="Script for mastermine demo.")

    # Adding arguments
    parser.add_argument('-images_dir', default='/home/gns/Documents/terna_colmap_reconstruction/queries/DJI_202407031532_019_Waypoint1/', type=str, help="")
    parser.add_argument('-image_ext',default='.JPG', type=str, help='')
    parser.add_argument('-pointcloud',default='/home/gns/Downloads/DATA/cloud_merged.las',type=str)
    parser.add_argument('-gps',default=True,type=bool)
    parser.add_argument('-img_path_db',default='/home/gns/Documents/terna_colmap_reconstruction/DJI_202407031342_007_H20-bobakas-1/',type=str)
    parser.add_argument('-database_path_db',default='/home/gns/Documents/terna_colmap_reconstruction/sift/small/database.db',type=str)
    parser.add_argument('-reconstruction_path',default='/home/gns/Documents/terna_colmap_reconstruction/sift/small/georeferenced/',type=str)
    # parser.add_argument("age", type=int, help="Your age")
    # parser.add_argument("--city", type=str, default="Unknown", help="Your city (optional)")

    args = parser.parse_args()


    #load pointcloud and downsample (optional)
    points,colors=ope.utils.get_pcd(path=args.pointcloud)
    # points_downsampled=ope.utils.downsample(points,colors,1)
    pointcloud=ope.utils.wgs84_to_utm(points)

    # vis_thread = threading.Thread(target=run_visualization(static_points=pointcloud), daemon=True)
    # vis_thread.start()
    vis_thread = threading.Thread(target=run_visualization, args=(pointcloud,colors,), daemon=True)
    vis_thread.start()
    time.sleep(20) #wait for the pointcloud visualization to load...

    images_dir= args.images_dir
    images= sorted(glob.glob(os.path.join(images_dir,"*"+args.image_ext)))
    for i, image_path in enumerate(images):
        im=load_image(image_path)
        
        
        detections = object_detection.detect_objects(im)
        if args.gps is True:
            objs3d = [ope.estimate(im,detection['center'],pointcloud) for detection in detections]
        else:
            
            objs3d = [ope.estimate_visual(im,detection['center'],pointcloud,position) for detection in detections]

        update(objs3d)
        






        
if __name__ == '__main__':
    main()


