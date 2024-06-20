import numpy as np
from PIL import Image
import utm
from scipy.spatial.transform import Rotation as R
import laspy
import pymap3d
import open3d as o3d
def read_image(path):
    im=Image.open(path)
    im_data=im.getxmp()['xmpmeta']['RDF']['Description']
    # rpy=np.asarray([im_data['FlightYawDegree'],im_data['FlightRollDegree'],im_data['FlightPitchDegree']]).astype(float)
    lat_lon=np.asarray([im_data['GpsLatitude'],im_data['GpsLongtitude']]).astype(float)
    x,y,zone,zone_letter=utm.from_latlon(lat_lon[0],lat_lon[1])
    alt=float(im_data['RelativeAltitude'])
    alt_abs=float(im_data['AbsoluteAltitude'])
    gimbal_rpy=np.asarray([im_data['GimbalYawDegree'],im_data['GimbalRollDegree'],im_data['GimbalPitchDegree'],]).astype(float)
    K=np.array([[float(im_data['CalibratedFocalLength']),0,float(im_data['CalibratedOpticalCenterX'])],[0,float(im_data['CalibratedFocalLength']),float(im_data['CalibratedOpticalCenterY'])],[0,0,1]]) 
    return {'gimbal_yrp':gimbal_rpy,'utm':[x,y],'altitude_rel':alt,'altitude_abs':alt_abs,'latlon':lat_lon,'image':im,'K':K}

def make_transformation_matrix(rpy,x,y,alt):
    r = R.from_euler('ZXY', rpy, degrees=True)
    RMat=r.as_matrix()
    t=np.array([x,y,alt])
    tMat=np.eye(4)
    tMat[:3, :3] = RMat
    tMat[:3, 3] = t
    return tMat

def get_pcd(path="/media/gns/CA78173A781724AB/Users/Gns/Documents/DJI/DJITerra/gryphon.lra@gmail.com/New LiDAR Point Cloud Mission/lidars/terra_las/clouddd513f95a815aba4.las"):

    #read point cloud data
    inFile = laspy.file.File(path, mode="r")
    points = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    return points

def get_ned(lla0,points):
    #convert points from utm to lat lon and then to ned
    # points_lat_lon=np.asarray([utm.to_latlon(point[0],point[1],35,'T') for point in points])
    points_lat_lon=np.vstack([np.asarray(utm.to_latlon(points[:,0],points[:,1],35,'T')),points[:,2]]).T
    points_ned=np.asarray(pymap3d.geodetic2ned(points_lat_lon[:,0],points_lat_lon[:,1],points_lat_lon[:,2] ,lla0[0],lla0[1],lla0[2])).T
    # pymap3d.ecef2ned
    return points_ned

def visualize(objs):
 
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for obj in objs:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(obj)
        vis.add_geometry(point_cloud)

    vis.run()
    vis.destroy_window()
    

def visualize_camera(T,K,points):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    vis.add_geometry(point_cloud)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(int(K[0,2])*2, int(K[1,2])*2, K[0,0], K[1,1], K[0,2], K[1,2])
    camera_visualization = o3d.geometry.LineSet.create_camera_visualization(intrinsic, np.linalg.inv(T))
    vis.add_geometry(camera_visualization)
    vis.run()
    vis.destroy_window()


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

