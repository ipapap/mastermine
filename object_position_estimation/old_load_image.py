
import numpy as np
import cv2
import exif
import PIL.Image as Image
import utm
import scipy
from scipy.spatial.transform import Rotation as R
import pymap3d
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# import matplotlib.pyplot as plt
path='/media/gns/0520-A64B/DCIM/DJI_202404111642_001_Create-Area-Route1/DJI_20240411164908_0003_W.JPG'
path='/media/gns/CA78173A781724AB/Users/Gns/Downloads/DJI_202309191217_002_Zenmuse-L1-mission/DJI_20230919121746_0007_Zenmuse-L1-mission.JPG'#DJI_20230919121743_0006_Zenmuse-L1-mission.JPG'
# path='/media/gns/0520-A64B/DCIM/DJI_202404111642_001_Create-Area-Route1/DJI_20240411164904_0002_W.JPG'
im=Image.open(path)
im_data=im.getxmp()['xmpmeta']['RDF']['Description']
im_spec=im.getexif()
# rpy=np.asarray([im_data['FlightRollDegree'],im_data['FlightPitchDegree'],im_data['FlightYawDegree']]).astype(float)
# lat_lon=np.asarray([im_data['GpsLatitude'],im_data['GpsLongtitude']]).astype(float)
# x,y,zone,zone_letter=utm.from_latlon(lat_lon[0],lat_lon[1])
# alt=float(im_data['AbsoluteAltitude'])

#create camera matrix with f=fx=fy and principal point at the center of the image
# create camera matrix with f=fx=fy and principal point at the center of the image
# f = 3883.1348531143867#3666
fx=3600.522132
fy=3600.058364
cx = 2730.258782#im.width / 2  # principal point x-coordinate
cy = 1808.137115#im.height / 2  # principal point y-coordinate
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# #add distortion coefficients to K
# distortion_coeffs = np.array([0, 0, 0, 0, 0])
# K = np.hstack([K, distortion_coeffs.reshape(-1, 1)])
dist_coeffs=np.array([22.872193, -84.182615, -0.001680, -0.000548, 157.333770, 22.851804, -84.078353, 157.124954])
# dist_coeffs=np.asarray([-3.856229584902835, 11.821708260772088, -0.0006744428586589581 ,-0.0010193520762412189 ,-8.3409150914395145 ,-3.8545590583355542 ,11.824144612066608 ,-8.340164789653846])
K,_=cv2.getOptimalNewCameraMatrix(K, dist_coeffs , (im.width, im.height), 1, (im.width, im.height))

# 4049.9561862824553 4049.132291555351 2736 1824 ([22.258479249294822, -67.219982148844068, -0.0018484272002384839 ,-0.00071189032404768546 ,179.24667667540098 ,22.230914116994295, -67.073950411894927 ,178.79347362219471])
# 4072.362236, 4071.630900, 2736.000000, 1824.000000, ([-3.100237, 0.970923, -0.001860, -0.000614, 30.622533, -3.104221, 1.001061, 30.543912])
# 3600.522132, 3600.058364, 2730.258782, 1808.137115, ([22.872193, -84.182615, -0.001680, -0.000548, 157.333770, 22.851804, -84.078353, 157.124954])
def read_image(path):
    im=Image.open(path)
    im_data=im.getxmp()['xmpmeta']['RDF']['Description']
    # rpy=np.asarray([im_data['FlightYawDegree'],im_data['FlightRollDegree'],im_data['FlightPitchDegree']]).astype(float)
    lat_lon=np.asarray([im_data['GpsLatitude'],im_data['GpsLongtitude']]).astype(float)
    x,y,zone,zone_letter=utm.from_latlon(lat_lon[0],lat_lon[1])
    alt=float(im_data['RelativeAltitude'])
    gimbal_rpy=np.asarray([im_data['GimbalYawDegree'],im_data['GimbalRollDegree'],im_data['GimbalPitchDegree'],]).astype(float)
    # K=np.array([[im_data['FocalLengthX'],0,im_data['PrincipalPointX']],[0,im_data['FocalLengthY'],im_data['PrincipalPointY']],[0,0,1]]) 
    return gimbal_rpy,x,y,alt,lat_lon

def make_transformation_matrix(rpy,x,y,alt):
    r = R.from_euler('ZXY', rpy, degrees=True)
    RMat=r.as_matrix()
    t=np.array([x,y,alt])
    tMat=np.eye(4)
    tMat[:3, :3] = RMat
    tMat[:3, 3] = t
    return tMat



spec=read_image(path) # read image data
lla0=spec[-1][0],spec[-1][1],spec[3]#[41.139054068767976, 24.914275053560292,10]  # get a reference point
ned=pymap3d.geodetic2ned(spec[-1][0],spec[-1][1],spec[3],lla0[0],lla0[1],lla0[2])  # get ned coordinates of the image relative to the reference point




T=make_transformation_matrix(spec[0]+np.array([0,0,0]),ned[0],ned[1],ned[2],) # create transformation matrix


#read point cloud data
import laspy
inFile = laspy.file.File("/media/gns/CA78173A781724AB/Users/Gns/Documents/DJI/DJITerra/gryphon.lra@gmail.com/New LiDAR Point Cloud Mission/lidars/terra_las/clouddd513f95a815aba4.las", mode="r")
points = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()

#convert points from utm to lat lon and then to ned
# points_lat_lon=np.asarray([utm.to_latlon(point[0],point[1],35,'T') for point in points])
points_lat_lon=np.vstack([np.asarray(utm.to_latlon(points[:,0],points[:,1],35,'T')),points[:,2]]).T
points_ned=np.asarray(pymap3d.geodetic2ned(points_lat_lon[:,0],points_lat_lon[:,1],points_lat_lon[:,2] ,lla0[0],lla0[1],lla0[2])).T


# points_ned=np.asarray(pymap3d.geodetic2ned( lla0[0],lla0[1],lla0[2],points_lat_lon[:,0],points_lat_lon[:,1],points_lat_lon[:,2])).T

import open3d as o3d

# make the points homogeneous and convert them to the drone's frame
points_new=(T@np.hstack([points_ned,np.ones((len(points_ned),1))]).T).T #convert points to drone's frame
points_new=points_new[:,:3]/points_new[:,3].reshape(-1,1)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points_new)#(points_ned)


# # Create a visualization window
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# fx = fy = 3666*10000  # focal length
# cx = im.width*10000 / 2  # principal point
# cy = im.height*10000 / 2
# intrinsic = o3d.camera.PinholeCameraIntrinsic(im.width*10000, im.height*10000, fx, fy, cx, cy)
# # Create a camera pose
# # Create a LineSet visualization of the camera
# camera_visualization = o3d.geometry.LineSet.create_camera_visualization(intrinsic, np.linalg.inv(T))
# # Add camera LineSet to the visualization
# vis.add_geometry(camera_visualization)
# vis.add_geometry(point_cloud)
# # Run the visualization
# vis.run()
# # Close the visualization window
# vis.destroy_window()



####### camera pixels to world coordinates
import numpy as np



def project_points(point_cloud, K, image_width, image_height, extrinsic_matrix=np.eye(4)):
    num_points = point_cloud.shape[0]
    homogeneous_points = np.hstack((point_cloud, np.ones((num_points, 1))))

    # Apply the extrinsic transformation (R, t)
    camera_points = (extrinsic_matrix @ homogeneous_points.T).T

    # Project the points using the intrinsic matrix K
    C=np.hstack([K, np.zeros((3, 1))])
    projected_points = (C @ camera_points.T).T
    # projected_points = (K @ camera_points[:, :3].T).T

    # Normalize the homogeneous coordinates to get pixel coordinates
    # pixel_coords = projected_points[:, :2] / projected_points[:, 2].reshape(-1, 1)
    pixel_coords = projected_points[:, :2] / projected_points[:, 2, np.newaxis]
    # Check which points fall within image bounds and are in front of the camera
    valid_mask = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_width) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_height) &
        (camera_points[:, 2] > 0)  # Ensure points are in front of the camera
    )

    return pixel_coords, valid_mask


res=project_points(points_new,K,im.width,im.height,extrinsic_matrix=np.linalg.inv(T))

# #show the points that are in front of the camera
# active_points=points_new[res[1]]

# #show the active points in red color and the rest in blue
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(active_points)
# point_cloud.paint_uniform_color([1, 0, 0])
# point_cloud_inactive = o3d.geometry.PointCloud()
# point_cloud_inactive.points = o3d.utility.Vector3dVector(points_new[~res[1]])

# # point_cloud_inactive.paint_uniform_color([0, 0, 1])

# # Create a visualization window
# vis = o3d.visualization.Visualizer()
# vis.create_window()

# # Add the point clouds to the visualization
# vis.add_geometry(point_cloud)
# vis.add_geometry(point_cloud_inactive)

# # Run the visualization
# vis.run()

# # Close the visualization window
# vis.destroy_window()






## painting only object points
object_coords=np.array([684,2784])#np.array([630,1665])
tree= scipy.spatial.cKDTree(res[0][res[1]]) # search only the valid points
inlrs=tree.query_ball_point((np.array([im.width-object_coords[0],im.height-object_coords[1]])),50) # for some reason the image is flipped

active_points=points_new[res[1]][inlrs]

#show the active points in red color and the rest in blue
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(active_points)
point_cloud.paint_uniform_color([0, 0, 0])
point_cloud_inactive = o3d.geometry.PointCloud()
point_cloud_inactive.points = o3d.utility.Vector3dVector(points_new)

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

1-1



# # using opencv
import cv2

rvec, _ = cv2.Rodrigues(T[:3,:3])
t=T[:3,3].reshape(3,1)

pts,_ =cv2.projectPoints(active_points, rvec, t, K, np.zeros((5, 1)))
pts=pts.squeeze(1)

new_camera_matrix = cv2.getOptimalNewCameraMatrix(K, np.zeros((5, 1)), (im.width, im.height), 1, (im.width, im.height))
 