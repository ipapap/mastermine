import utils
import numpy as np
import cv2
# import pymap3d
import scipy
import open3d as o3d
import utm


path='/media/gns/backup/duth@terna/db/sim2/DJI_202407031532_019_Waypoint1/DJI_20240703154308_0004_W.JPG'#'/media/gns/backup/duth@terna/db/sim2/DJI_202407021401_006_l1-mine-m1/DJI_20240702140341_0058_l1-mine-m1.JPG'
fx=3600.522132
fy=3600.058364
cx = 2730.258782#im.width / 2  # principal point x-coordinate
cy = 1808.137115#im.height / 2  # principal point y-coordinate
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# # #add distortion coefficients to K
# # distortion_coeffs = np.array([0, 0, 0, 0, 0])
# # K = np.hstack([K, distortion_coeffs.reshape(-1, 1)])
# dist_coeffs=np.array([22.872193, -84.182615, -0.001680, -0.000548, 157.333770, 22.851804, -84.078353, 157.124954])
# # dist_coeffs=np.asarray([-3.856229584902835, 11.821708260772088, -0.0006744428586589581 ,-0.0010193520762412189 ,-8.3409150914395145 ,-3.8545590583355542 ,11.824144612066608 ,-8.340164789653846])

# K,_=cv2.getOptimalNewCameraMatrix(K, dist_coeffs , (im_data['image'].width, im_data['image'].height), 1, (im_data['image'].width, im_data['image'].height))
# get intrinsic matrix from the  image metadata
im_data = utils.read_image(path)
K=im_data['K']

print(im_data)

# convert image coordinates to ned relative to a reference point (first image)
lla0=im_data['latlon'][0],im_data['latlon'][1],im_data['altitude_abs']#[41.139054068767976, 24.914275053560292,10]  # get a reference point
# im_ned=pymap3d.geodetic2ned(im_data['latlon'][0],im_data['latlon'][1],im_data['altitude_abs'],lla0[0],lla0[1],lla0[2])  # get ned coordinates of the image relative to the reference point

# create transformation matrix from ned to camera frame
# T_im2w=utils.make_transformation_matrix(im_data['gimbal_yrp'],im_ned[0],im_ned[1],im_ned[2]) # create transformation matrix

# create transformation matrix from enu to camera frame
T_im2w=utils.make_transformation_matrix_ENU(im_data['gimbal_yrp'],im_data['utm'][0],im_data['utm'][1],im_data['altitude_abs']) # create transformation matrix

# get point cloud data
points,colors=utils.get_pcd(path="/home/gns/Downloads/Mpompakas_3_7_24_DUTH_LAS_0.1m.las")

#### transform from ggrs87
points_downsampled=utils.downsample(points,colors,1)

# convert points to lat lon
points=utils.grs87_to_wgs84(points)

# convert point cloud data to ned relative to the reference point
# points_ned=utils.get_ned(lla0,points)

points_ned=points

utils.visualize_camera(T_im2w,K,points_ned)

# project the points to the image plane
res=utils.project_points(points_ned,K,im_data['image'].width,im_data['image'].height,extrinsic_matrix=np.linalg.inv(T_im2w))


## painting only object points
object_coords=np.array([3020,1304])#np.array([630,1665])
tree= scipy.spatial.cKDTree(res[0][res[1]]) # search only the valid points
inlrs=tree.query_ball_point((np.array([object_coords[0],object_coords[1]])),150) # for some reason the image is flipped

active_points=points_ned[res[1]][inlrs]

#show the active points in red color and the rest in blue
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(active_points)
point_cloud.paint_uniform_color([1, 0, 0])
point_cloud_inactive = o3d.geometry.PointCloud()
point_cloud_inactive.points = o3d.utility.Vector3dVector(points_ned)
point_cloud_inactive.colors = o3d.utility.Vector3dVector(colors)

# point_cloud_inactive.paint_uniform_color([0, 0, 1])

# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the point clouds to the visualization
vis.add_geometry(point_cloud_inactive)
vis.add_geometry(point_cloud)

# Run the visualization
vis.run()
# Close the visualization window
vis.destroy_window()

