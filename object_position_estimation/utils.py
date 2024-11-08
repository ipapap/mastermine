import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import pyproj
import utm
from scipy.spatial.transform import Rotation as R
import laspy
# import pymap3d
import open3d as o3d
import scipy

def read_image(path):
    im=Image.open(path)
    im_data=im.getxmp()['xmpmeta']['RDF']['Description']
    if im_data['Model'] == 'ZH20':

        # rpy=np.asarray([im_data['FlightYawDegree'],im_data['FlightRollDegree'],im_data['FlightPitchDegree']]).astype(float)
        lat_lon=np.asarray([im_data['GpsLatitude'],im_data['GpsLongitude']]).astype(float)  # typo error in the key name
        x,y,zone,zone_letter=utm.from_latlon(lat_lon[0],lat_lon[1])
        alt=float(im_data['RelativeAltitude'])
        alt_abs=float(im_data['AbsoluteAltitude'])
        gimbal_rpy=np.asarray([im_data['GimbalYawDegree'],im_data['GimbalRollDegree'],im_data['GimbalPitchDegree'],]).astype(float)
        exif_data = im._getexif()
        def get_exif_tag_key(tag_name):
            for key, value in TAGS.items():
                if value == tag_name:
                    return key
            return None
        focal_length_key = get_exif_tag_key('FocalLength')
        focal_length = float(exif_data.get(focal_length_key))*1000
          




        # Camera intrinsic matrix
        K = np.array([
            [focal_length, 0, im.width / 2],
            [0, focal_length, im.height / 2],
            [0, 0, 1]
        ])

    elif im_data['Model'] == 'EP800':

        # rpy=np.asarray([im_data['FlightYawDegree'],im_data['FlightRollDegree'],im_data['FlightPitchDegree']]).astype(float)
        lat_lon=np.asarray([im_data['GpsLatitude'],im_data['GpsLongtitude']]).astype(float)
        x,y,zone,zone_letter=utm.from_latlon(lat_lon[0],lat_lon[1])
        alt=float(im_data['RelativeAltitude'])
        alt_abs=float(im_data['AbsoluteAltitude'])
        gimbal_rpy=np.asarray([im_data['GimbalYawDegree'],im_data['GimbalRollDegree'],im_data['GimbalPitchDegree'],]).astype(float)
        K=np.array([[float(im_data['CalibratedFocalLength']),0,float(im_data['CalibratedOpticalCenterX'])],[0,float(im_data['CalibratedFocalLength']),float(im_data['CalibratedOpticalCenterY'])],[0,0,1]]) 

    else :
        print('Camera model not supported')
        return None
    return {'gimbal_yrp':gimbal_rpy,'utm':[x,y],'altitude_rel':alt,'altitude_abs':alt_abs,'latlon':lat_lon,'image':im,'K':K}

def make_transformation_matrix(rpy,x,y,alt):
    r = R.from_euler('ZXY', rpy, degrees=True)
    # r = R.from_euler('YXZ', np.flip(rpy), degrees=True)
    RMat=r.as_matrix()
    #gimbal to image plane comversion
    T_ned_to_camera = np.array([
    [0, 0, 1],  # NED X (North) to Camera Z (Forward)
    [1, 0, 0],  # NED Y (East) to Camera X (Right)
    [0, 1, 0]   # NED Z (Down) to Camera Y (Down)
])
    RMat= RMat @T_ned_to_camera
    t=np.array([x,y,alt])
    tMat=np.eye(4)
    tMat[:3, :3] = RMat
    tMat[:3, 3] = t
    return tMat


def TMat(r,t):
    tMat=np.eye(4)
    tMat[:3, :3] = r
    tMat[:3, 3] = t
    return tMat

def make_transformation_matrix_ENU(rpy,x,y,alt):
    # rpy=np.array([-rpy[2],rpy[1],rpy[0]])
    r = R.from_euler('ZXY', rpy, degrees=True)
    RMat=r.as_matrix()


    # # NED to UTM (ENU) rotation
    #                         E  N  U            
    R_ned_to_utm = np.array([[0, 1, 0],    #N
                             [1, 0, 0],    #E
                             [0, 0, -1]])  #D

    #                            X  Y  Z
    r_enu_to_camera = np.array([[0, 0, 1],  # E 
                                [1, 0, 0],  # N 
                                [0, -1, 0]   # U 

])    
    r_ned_to_camera = np.array([
    [0, 0, 1],  # NED X (North) to Camera Z (Forward)
    [1, 0, 0],  # NED Y (East) to Camera X (Right)
    [0, 1, 0]   # NED Z (Down) to Camera Y (Down)
])

    RMat=  R_ned_to_utm.T @ RMat @ r_ned_to_camera
    
    # # # # Transform to UTM (ENU)
    # RMat = R_ned_to_utm.T @ RMat #@ R_ned_to_utm.T
    t=np.array([x,y,alt])
    tMat=np.eye(4)
    tMat[:3, :3] = RMat
    tMat[:3, 3] = t
    return tMat

# def get_pcd(path):
#     #read point cloud data
#     inFile = laspy.file.File(path, mode="r")
#     points = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
#     colors = np.vstack((inFile.red, inFile.green, inFile.blue)).transpose()/ 65535.0
#     return points,colors

def get_pcd(path):
<<<<<<< HEAD
    #read point cloud data
    inFile = laspy.read(path)
    points = np.vstack((inFile.y, inFile.x, inFile.z)).transpose()
    colors = np.vstack((inFile.red, inFile.green, inFile.blue)).transpose()/ 65535.0
    return points,colors
=======
    # Reading a LAS file using laspy 2.x
    las = laspy.read(path)
    
    # Extract the point coordinates
    points = np.vstack((las.y, las.x, las.z)).transpose()
    
    # Optionally, extract colors if available
    try:
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
    except AttributeError:
        # If no color information is available, use a default color (e.g., white)
        colors = np.ones(points.shape)

    return points, colors
>>>>>>> b1d6713f1622b724fa6ccea437aadb443a192549

def get_ned(lla0,points):
    #convert points from utm to lat lon and then to ned
    # points_lat_lon=np.asarray([utm.to_latlon(point[0],point[1],35,'T') for point in points])
    points_lat_lon=np.vstack([np.asarray(utm.to_latlon(points[:,0],points[:,1],35,'T')),points[:,2]]).T
    points_ned=np.asarray(pymap3d.geodetic2ned(points_lat_lon[:,0],points_lat_lon[:,1],points_lat_lon[:,2] ,lla0[0],lla0[1],lla0[2])).T
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

# inverce projection
def project_points_inv(pixel_coords, K, extrinsic_matrix=np.eye(4)):
    num_points = pixel_coords.shape[0]
    homogeneous_points = np.hstack((pixel_coords, np.ones((num_points, 1))))

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
    # valid_mask = (
    #     (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_width) &
    #     (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_height) &
    #     (camera_points[:, 2] > 0)  # Ensure points are in front of the camera
    # )

    return pixel_coords
def downsample(points,colors=None,voxel_size=1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downpcd.points)

def grs87_to_wgs84(points):
    ggrs87 = pyproj.CRS('EPSG:2100')
    wgs84 = pyproj.CRS('EPSG:4326')
    transformer = pyproj.Transformer.from_crs(ggrs87, wgs84, always_xy=False)
    points_lat_lon = np.array(transformer.transform(points[:,0],points[:,1]) ).T
    # to utm
    x,y,zone,letter=utm.from_latlon(points_lat_lon[:,0],points_lat_lon[:,1])
    return(np.vstack([x,y,points[:,2]]).T)


def wgs84_to_utm(points):
    e,n,zn,nl = utm.from_latlon(points[:,0],points[:,1])

    return(np.vstack([e,n,points[:,2]]).T)


# def search_obj(point_clouds,img_coords,K,T_im2w,image_shape=(0,0),obj_radius=50,search_radius=100):

#     relevant_points=[]
#     for i,point_cloud in enumerate(point_clouds):
#         pixels, valid_mask = project_points(point_cloud, K, image_shape[0], image_shape[1] , extrinsic_matrix=np.linalg.inv(T_im2w))
#         tree= scipy.spatial.cKDTree(pixels[valid_mask])
#         inlrs=tree.query_ball_point((np.array([img_coords[0],img_coords[1]])),obj_radius)

#         if i<len(point_clouds)-1:
#             relevant_points=point_cloud[valid_mask]
            
        


        

#     return inlrs