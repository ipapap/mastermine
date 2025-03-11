import object_position_estimation.utils as utils
import numpy as np
import cv2
# import pymap3d
import scipy
import open3d as o3d
import utm


# # get point cloud data
# points,colors=utils.get_pcd(path='/home/gns/Downloads/DATA/cloud_merged.las')
# points_downsampled=utils.downsample(points,colors,1)
# points=utils.wgs84_to_utm(points)

def estimate(image,obj2d,point_cloud):
    im_data = utils.read_meta(image) #reads metadata only
    K=im_data['K']
    T_im2w=utils.make_transformation_matrix_ENU(im_data['gimbal_yrp'],im_data['utm'][0],im_data['utm'][1],im_data['altitude_abs']) # create transformation matrix
    # mapping the 3d points to image plane 
    res=utils.project_points(point_cloud,K,im_data['image'].width,im_data['image'].height,extrinsic_matrix=np.linalg.inv(T_im2w))
    # find the points close to the detected object on image plane and then get their 3d position
    object_coords=obj2d
    tree= scipy.spatial.cKDTree(res[0][res[1]]) # search only the valid points
    inlrs=tree.query_ball_point((np.array([object_coords[0],object_coords[1]])),150)
    relevant3dpoints=point_cloud[res[1]][inlrs]
    obj3d=np.median(relevant3dpoints,axis=0)
    return obj3d

def estimate_visual(image,obj2d,point_cloud,position):
    im_data = utils.read_meta(image) #reads metadata only
    K=im_data['K']
    T_im2w=utils.make_transformation_matrix_ENU(im_data['gimbal_yrp'],*position) #position in UTM !!
    # mapping the 3d points to image plane 
    res=utils.project_points(point_cloud,K,im_data['image'].width,im_data['image'].height,extrinsic_matrix=np.linalg.inv(T_im2w))
    # find the points close to the detected object on image plane and then get their 3d position
    object_coords=obj2d
    tree= scipy.spatial.cKDTree(res[0][res[1]]) # search only the valid points
    inlrs=tree.query_ball_point((np.array([object_coords[0],object_coords[1]])),150)
    relevant3dpoints=point_cloud[res[1]][inlrs]
    obj3d=np.median(relevant3dpoints,axis=0)
    return obj3d
