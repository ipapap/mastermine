import object_position_estimation as ope
import object_detection
from  visualization.visualize import *
# import colmap_localization.localize as loc
import colmap_localization.localization
# from colmap_localization.localization import Loc
from PIL import Image
import glob
import os
import numpy as np
import argparse
import threading
import pandas as pd
import utm


def load_image(image_path):
    image = Image.open(os.path.join(image_path))
    return image



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
    parser.add_argument('-produce_csv',default=False,type=bool)
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
    time.sleep(30) #wait for the pointcloud visualization to load...

    images_dir= args.images_dir


    if args.produce_csv==True:
        rows=[]

    ccolors =[
                [1.0, 0.0, 0.0],      # Red
                [0.0, 1.0, 0.0],      # Green
                [0.0, 0.0, 1.0],      # Blue
                [1.0, 1.0, 0.0],      # Yellow
                [1.0, 0.0, 1.0],      # Magenta
                [0.0, 1.0, 1.0],      # Cyan
                [1.0, 0.65, 0.0],     # Orange
                [0.5, 0.0, 0.5],      # Purple
                ]


    classes=['bulldoze', 'car', 'driller', 'dump_truck', 'excavator', 'grader', 'human', 'truck']
    classes_colors={'bulldoze':ccolors[0], 'car':ccolors[1], 'driller':ccolors[2], 'dump_truck':ccolors[3], 'excavator':ccolors[4], 'grader':ccolors[5], 'human':ccolors[6], 'truck':ccolors[7]}
    loc=colmap_localization.localization.Loc()
    images= sorted(glob.glob(os.path.join(images_dir,"*"+args.image_ext)))
    for i, image_path in enumerate(images):
        im=load_image(image_path)
        
        
        detections = object_detection.detect_objects(im)
        if args.gps is True:
            objs3d = [ope.estimate(im,detection['center'],pointcloud) for detection in detections]
            objs3d_colors = [classes_colors[detection['class']] for detection in detections]
        else:
            raise Exception("Please run localization script") 
            pose=loc.add_query(image_path)
            if pose is not None:
                objs3d = [ope.estimate_visual(im,detection['center'],pointcloud,pose) for detection in detections]
            else: objs3d=[]
        update(objs3d,objs3d_colors)

        if args.produce_csv==True:
            columns = ["image_name", "uav_position", "obs_3d","objs_label","objs_2d","bbox","time"]
            # gimbal_yrp,utm,altitude_rel,altitude_abs,latlon,image,K
            meta= ope.utils.read_image(image_path)
            _,_,n,l=utm.from_latlon(*meta['latlon'])
            objects_latlon=[list(utm.to_latlon(o[0],o[1],n,l))+[o[2]] for o in objs3d]
            classes=[c['class'] for c in detections]
            centers=[c['center'] for c in detections]
            bboxes=[c['bbox'] for c in detections]

            rows.append([image_path.split('/')[-1],[meta['latlon'][0]]+[meta['latlon'][1]]+[meta['altitude_abs']],objects_latlon,classes,centers,bboxes,meta['time']])

    if args.produce_csv==True:
        df = pd.DataFrame(rows, columns=columns)
        csv_path = "produced_data1.csv"
        df.to_csv(csv_path, index=True)


        






        
if __name__ == '__main__':
    main()


