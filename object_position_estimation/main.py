import numpy as np
import cv2
# import pymap3d
import scipy
import open3d as o3d
import utm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils  

# Define specific colors for each class
class_colors = {
    'bulldozer': [1, 0, 0],     # Red
    'car': [0, 1, 0],           # Green
    'driller': [0, 0, 1],       # Blue
    'dump_truck': [1, 1, 0],    # Yellow
    'excavator': [0, 1, 1],     # Cyan
    'grader': [1, 0, 1],        # Magenta
    'human': [0.5, 0.5, 0.5],   # Gray
    'truck': [1, 0.5, 0]        # Orange
}

# Function to perform object estimation using point cloud data
def estimate_object_positions(detected_info, image_path, las_file_path):
    # Load image metadata
    fx=3600.522132
    fy=3600.058364
    cx = 2730.258782#im.width / 2  # principal point x-coordinate
    cy = 1808.137115#im.height / 2  # principal point y-coordinate
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    im_data = utils.read_image(image_path)
    K = im_data['K']


    # convert image coordinates to ned relative to a reference point (first image)
    lla0=im_data['latlon'][0],im_data['latlon'][1],im_data['altitude_abs']#[41.139054068767976, 24.914275053560292,10]  # get a reference point
    # im_ned=pymap3d.geodetic2ned(im_data['latlon'][0],im_data['latlon'][1],im_data['altitude_abs'],lla0[0],lla0[1],lla0[2])  # get ned coordinates of the image relative to the reference point

    # create transformation matrix from ned to camera frame
    # T_im2w=utils.make_transformation_matrix(im_data['gimbal_yrp'],im_ned[0],im_ned[1],im_ned[2]) # create transformation matrix

    # create transformation matrix from enu to camera frame
    T_im2w=utils.make_transformation_matrix_ENU(im_data['gimbal_yrp'],im_data['utm'][0],im_data['utm'][1],im_data['altitude_abs']) # create transformation matrix

    points, colors = utils.get_pcd(path=las_file_path)
    points_downsampled = points # utils.downsample(points, colors, 1)
    points = utils.wgs84_to_utm(points)
    points_ned = points

    # Visualize camera
    utils.visualize_camera(T_im2w, K, points_ned)

    # Project points onto the image plane
    res = utils.project_points(points_ned, K, im_data['image'].width, im_data['image'].height, extrinsic_matrix=np.linalg.inv(T_im2w))

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for obj in detected_info:
        object_coords = np.array(obj['center'])  # Get the (x, y) center from detection
        class_name = obj['class']

        # Find the corresponding point cloud data
        tree = scipy.spatial.cKDTree(res[0][res[1]])
        inlrs = tree.query_ball_point(object_coords, 150)  # Use radius to find nearby points

        active_points = points_ned[res[1]][inlrs]

        # Create point cloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(active_points)

        # Assign a color based on the object class
        color = class_colors.get(class_name, [1, 1, 1])  # Default to white if class not found
        point_cloud.paint_uniform_color(color)

        # Add the point cloud to the visualization
        vis.add_geometry(point_cloud)

    # Add the inactive point cloud
    point_cloud_inactive = o3d.geometry.PointCloud()
    point_cloud_inactive.points = o3d.utility.Vector3dVector(points_ned)
    point_cloud_inactive.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(point_cloud_inactive)

    # Run visualization
    vis.run()
    vis.destroy_window()