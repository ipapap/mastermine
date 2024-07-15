import numpy as np
from PIL import Image
import utm
from scipy.spatial.transform import Rotation as R
import laspy
import pymap3d
import open3d as o3d
import scipy


class Map:
    def __init__(self, pcd_path):
        # self.path = path
        # self.im = Image.open(path)
        # self.im_data = self.im.getxmp()['xmpmeta']['RDF']['Description']
        # self.lat_lon = np.asarray([self.im_data['GpsLatitude'], self.im_data['GpsLongtitude']]).astype(float)
        # self.x, self.y, self.zone, self.zone_letter = utm.from_latlon(self.lat_lon[0], self.lat_lon[1])
        # self.alt = float(self.im_data['RelativeAltitude'])
        # self.alt_abs = float(self.im_data['AbsoluteAltitude'])
        # self.gimbal_rpy = np.asarray([self.im_data['GimbalYawDegree'], self.im_data['GimbalRollDegree'], self.im_data['GimbalPitchDegree'], ]).astype(float)
        # self.K = np.array([[float(self.im_data['CalibratedFocalLength']), 0, float(self.im_data['CalibratedOpticalCenterX'])], [0, float(self.im_data['CalibratedFocalLength']), float(self.im_data['CalibratedOpticalCenterY'])], [0, 0, 1]])
        # self.pcd=self.get_pcd(pcd_path)
        self.pcd_scales = [self.self.get_pcd(pcd_path)]
        self.pcd_scales.extend([self.downsample(self.pcd, s) for s in range(1,10,3)])

        
    def make_transformation_matrix(self, rpy, x, y, alt):
        r = R.from_euler('ZXY', rpy, degrees=True)
        RMat = r.as_matrix()
        T_ned_to_camera = np.array([
            [0, 0, 1],  # NED X (North) to Camera Z (Forward)
            [1, 0, 0],  # NED Y (East) to Camera X (Right)
            [0, 1, 0]  # NED Z (Down) to Camera Y (Down)
        ])
        RMat = RMat @ T_ned_to_camera
        t = np.array([x, y, alt])
        tMat = np.eye(4)
        tMat[:3, :3] = RMat
        tMat[:3, 3] = t
        return tMat

    def TMat(self, r, t):
        tMat = np.eye(4)
        tMat[:3, :3] = r
        tMat[:3, 3] = t
        return tMat


    
    def get_pcd(path="/media/gns/CA78173A781724AB/Users/Gns/Documents/DJI/DJITerra/gryphon.lra@gmail.com/New LiDAR Point Cloud Mission/lidars/terra_las/clouddd513f95a815aba4.las"):

        #read point cloud data
        inFile = laspy.file.File(path, mode="r")
        points = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
        return points
    
    def get_ned(self, lla0, points):
        points_lat_lon = np.vstack([np.asarray(utm.to_latlon(points[:, 0], points[:, 1], 35, 'T')), points[:, 2]]).T
        points_ned = np.asarray(pymap3d.geodetic2ned(points_lat_lon[:, 0], points_lat_lon[:, 1], points_lat_lon[:, 2], lla0[0], lla0[1], lla0[2])).T
        return points_ned
    
    def visualize_camera(self, T_im2w, K, points_ned):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_ned)
        o3d.visualization.draw_geometries([point_cloud])

    def project_points(self, points_ned, K, width, height, extrinsic_matrix):
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
    
    def downsample(points,voxel_size=1):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        return np.asarray(downpcd.points)
    

    def search_obj(self,obj_coords,K,T_im2w,image_shape=(0,0),obj_radius=50,search_radius=100):

        relevant_points=[]
        for i,point_cloud in enumerate(point_clouds):
            pixels, valid_mask = project_points(point_cloud, K, image_shape[0], image_shape[1] , extrinsic_matrix=np.linalg.inv(T_im2w))
            tree= scipy.spatial.cKDTree(pixels[valid_mask])
            inlrs=tree.query_ball_point((np.array([img_coords[0],img_coords[1]])),obj_radius)

            if i<len(point_clouds)-1:
                relevant_points=point_cloud[valid_mask]

    def processImg(self,path):

        im=Image.open(path)
        im_data=im.getxmp()['xmpmeta']['RDF']['Description']
        # rpy=np.asarray([im_data['FlightYawDegree'],im_data['FlightRollDegree'],im_data['FlightPitchDegree']]).astype(float)
        lat_lon=np.asarray([im_data['GpsLatitude'],im_data['GpsLongtitude']]).astype(float)
        x,y,zone,zone_letter=utm.from_latlon(lat_lon[0],lat_lon[1])
        alt=float(im_data['RelativeAltitude'])
        alt_abs=float(im_data['AbsoluteAltitude'])
        gimbal_rpy=np.asarray([im_data['GimbalYawDegree'],im_data['GimbalRollDegree'],im_data['GimbalPitchDegree'],]).astype(float)
        K=np.array([[float(im_data['CalibratedFocalLength']),0,float(im_data['CalibratedOpticalCenterX'])],[0,float(im_data['CalibratedFocalLength']),float(im_data['CalibratedOpticalCenterY'])],[0,0,1]]) 
        self.img={'gimbal_yrp':gimbal_rpy,'utm':[x,y],'altitude_rel':alt,'altitude_abs':alt_abs,'latlon':lat_lon,'image':im,'K':K}

        # convert image coordinates to ned relative to a reference point (first image)
        lla0=im_data['latlon'][0],im_data['latlon'][1],im_data['altitude_abs']#[41.139054068767976, 24.914275053560292,10]  # get a reference point
        
        im_ned=pymap3d.geodetic2ned(im_data['latlon'][0],im_data['latlon'][1],im_data['altitude_abs'],lla0[0],lla0[1],lla0[2])  # get ned coordinates of the image relative to the reference point

        # create transformation matrix from ned to camera frame
        self.T_im2w=self.make_transformation_matrix(im_data['gimbal_yrp'],im_ned[0],im_ned[1],im_ned[2]) # create transformation matrix

        # create transformation matrix from enu to camera frame
        # T_im2w=utils.make_transformation_matrix_ENU(im_data['gimbal_yrp'],im_data['utm'][0],im_data['utm'][1],im_data['altitude_abs']) # create transformation matrix

        # convert point cloud data to ned relative to the reference point
        points_ned=self.get_ned(lla0,self.points)
            T_im2w=self.make_transformation_matrix(gimbal_rpy,x,y,alt)

        