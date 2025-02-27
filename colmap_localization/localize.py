import pycolmap
import numpy as np
import cv2
import os
from database import *
import torch
# import glob
import scipy
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import encoder
from PIL import Image, ImageOps
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import object_position_estimation.utils as utils
import utils_localization
import utm
import pyproj
import shutil
IMG_PATH_QUERY = '/home/gns/Documents/terna_colmap_reconstruction/queries/DJI_202407031532_019_Waypoint1/'
# DATABASE_PATH_QUERY = BASE_PATH_QUERY + 'database.db'
IMG_PATH_DB = '/home/gns/Documents/terna_colmap_reconstruction/DJI_202407031342_007_H20-bobakas-1/'#'/home/gns/Documents/terna_colmap_reconstruction/'
DATABASE_PATH_DB = '/home/gns/Documents/terna_colmap_reconstruction/sift/small/' + 'database.db'

reconstruction_path = '/home/gns/Documents/terna_colmap_reconstruction/sift/small/georeferenced/'#'/home/gns/Documents/terna_colmap_reconstruction/georeferenced/reconstruction_georef/'


# camera=pycolmap.Camera(
#     model=3,
#     width=4056,
#     height=3040,
#     params=[3600.522132, 4056/2, 3040/2])
# camera = {
#     'model': 3,
#     'width': 4056,
#     'height': 3040,
#     'params': [3645.8096052435262 ,2736 ,1824 ,-0.005470386364029526],#[4500,4500, 4056/2, 3040/2]#
# }

# camera = pycolmap.Camera(
#     model='SIMPLE_PINHOLE',
#     width=4056,
#     height=3040,
#     params=[3645.8096052435262, 4056/2, 3040/2],
# )

#for h20 camera
camera = pycolmap.Camera(
    model='OPENCV',
    width=4056,
    height=3040,
    params=[2931.6527767663661, 2940.4279383883672 ,2028 ,1520 ,0.068809984031020663, -0.17458781534169701, 0.0010840641924016881, 5.0892999338629291e-05],
)

# #for l1 camera !!
# camera = pycolmap.Camera(
#     model='OPENCV',
#     width=5472,
#     height=3648,
#     params=[3714.3774966817214 ,3664.7406544074606 ,2736 ,1824 ,-0.0042931479985967502 ,0.0008636320981303202, -0.0051340506677993057, -0.00026089743088417519],
# )
def feature_matching(des1,des2):
    #lowe ratio test and get the inliers
    matcher = cv2.BFMatcher()
   
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50) 
    # matcher = cv2.FlannBasedMatcher(index_params,search_params)

    matches = matcher.knnMatch(des1.astype(np.float32),des2.astype(np.float32), k=2)
    good = []
    inds1 = []
    inds2 = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            inds1.append(m.queryIdx)
            inds2.append(m.trainIdx)
    inds1 = np.array(inds1)
    inds2 = np.array(inds2)

    return good,inds1,inds2


def localize_image(image_path,frames,frames_descriptors,encoder,top_k=1,threshold=0.5):
    # des,kp=get_features(image_id,DATABASE_PATH_QUERY)

    img = Image.open(image_path).convert('RGB')
    img = ImageOps.grayscale(img)
    img = np.array(img).astype(np.float32) / 255.
    sift = pycolmap.Sift()

    # Parameters:
    # - image: HxW float array
    kp, des = sift.extract(img) # soooo  slowwww, maybe use cuda
    des = (des *512).astype(np.uint8)

    # des=torch.tensor(des)
    descriptor=encoder.model(encoder.preprocess_image(image_path).cuda()).detach().cpu()

    dist=torch.cdist(descriptor,frames_descriptors)

    neigh=torch.argsort(dist.view(-1))[:top_k] # use top x retrieved images
    poses=[]
    points2d=[]
    points3d=[]
    for n in neigh:
        frame = frames[n]
        des1=frame.descriptors_local
        kp1=frame.points2d

        good,inds1,inds2=feature_matching(des,des1)
        ## maybe use fundamental matrix to filter out the outliers .. OPTIONAL
        # _,inliers=cv2.findFundamentalMat(kp[:,:2][inds1],np.asarray(kp1)[inds2])
        # if inliers is not None:
        #     inds1= inds1[inliers.ravel()]
        #     inds2=inds2[inliers.ravel()]
        #get the inliers
        points2D = kp[inds1][:,:2]
        points3D = np.asarray(frame.points3d)[inds2]
        # Estimate the pose
        points2d.append(points2D)
        points3d.append(points3D)

        
    points2d=np.concatenate(points2d)
    points3d=np.concatenate(points3d)

    pose = pycolmap.absolute_pose_estimation(\
            points2d, points3d, camera,
            estimation_options={'ransac':{'max_error':8.0,'min_inlier_ratio':0.0025,'confidence':.999,'min_num_trials':5000,'max_num_trials':10000}},
            refinement_options={'refine_focal_length':True,'max_num_iterations':5000,'gradient_tolerance':1e3},
        )

    ## Rest for validating the result with gt 
    if pose is  None: return
    if pose['num_inliers']/len(points2d) < threshold and pose['num_inliers']>500 : return
    quat=pose['cam_from_world'].rotation
    t=pose['cam_from_world'].translation
    T=pose_to_T_inv(pose['cam_from_world'].rotation.quat,pose['cam_from_world'].translation,w_last=False)

    # convert coordinates to utm (OPTIONAL)
    # t=pose_to_T_inv(pose['cam_from_world'].rotation.quat,pose['cam_from_world'].translation,w_last=False)[:3,3]
    # # ecef to utm
    # ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")  # ECEF
    # wgs84 = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")  # WGS84 Geodetic
    # transformer = pyproj.Transformer.from_proj(ecef, wgs84)
    # lon, lat, alt = transformer.transform(*t)  
    # x,y,_,_=utm.from_latlon(lat,lon)
    return T

def translation_error(t_est, t_gt):
    return np.linalg.norm(t_est - t_gt)

def rotation_error(R_est, R_gt):
    R_error = R_est @ R_gt.T
    trace = np.trace(R_error)
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)))

def pose_to_T_inv(q,t,w_last=True):
    r=scipy.spatial.transform.Rotation.from_quat(q,scalar_first=w_last).as_matrix()
    T=np.eye(4)
    T[:3,:3]=r
    T[:3,3]=np.asarray(t)
    return(np.linalg.inv(T))

def get_descriptors(image_id,database_path=DATABASE_PATH_DB):
    # Connect to the COLMAP SQLite database
    db = COLMAPDatabase.connect(database_path)
    descriptors = db.execute("SELECT data FROM descriptors WHERE image_id = ?", (image_id,)).fetchone()[0]

    # Convert the blob back into a numpy array
    descriptors_array = blob_to_array(descriptors, np.uint8)

    # Reshape the array into the original descriptor shape (rows, 128)
    num_descriptors = len(descriptors_array) // 128
    descriptors_array = descriptors_array.reshape((num_descriptors, 128))

    # print("Number of descriptors:", num_descriptors)
    # print("Descriptors shape:", descriptors_array.shape)
    # print(descriptors_array)
    db.close()
    return descriptors_array

def get_features(image_id, database_path):
    # Connect to the COLMAP SQLite database
    db = COLMAPDatabase.connect(database_path)
    
    # Fetch descriptors for the image
    descriptors = db.execute("SELECT data FROM descriptors WHERE image_id = ?", (image_id,)).fetchone()[0]

    # Convert the descriptor blob back into a numpy array (dtype = uint8)
    descriptors_array = blob_to_array(descriptors, np.uint8) #float for r2d2
    # descriptors_array = blob_to_array(descriptors, np.float32) #float for r2d2
    # Reshape the array into the original descriptor shape (num_descriptors, 128)
    num_descriptors = len(descriptors_array) // 128
    descriptors_array = descriptors_array.reshape((num_descriptors, 128))

    # Fetch keypoints for the image
    keypoints = db.execute("SELECT data FROM keypoints WHERE image_id = ?", (image_id,)).fetchone()[0]
    
    # Convert the keypoint blob back into a numpy array (dtype = float32)
    keypoints_array = blob_to_array(keypoints, np.float32)
    
    # Determine the dimensionality of the keypoints (2D, 4D, or 6D)
    num_keypoints = len(keypoints_array) // num_descriptors
    
    if num_keypoints == 2:  # If 2D keypoints (x, y)
        keypoints_array = keypoints_array.reshape((num_descriptors, 2))
    elif num_keypoints == 4:  # If 4D keypoints (x, y, scale, orientation)
        keypoints_array = keypoints_array.reshape((num_descriptors, 4))
    elif num_keypoints == 6:  # If 6D affine keypoints (x, y, a11, a12, a21, a22)
        keypoints_array = keypoints_array.reshape((num_descriptors, 6))

    # Close the database connection
    db.close()

    # Return both descriptors and keypoints
    return descriptors_array, keypoints_array

class Frame():
    def __init__(self, image_id, image, points3d , points2d,valid_mask, descriptor=None, base_path=IMG_PATH_DB,database_path=DATABASE_PATH_DB):
        self.image_id = image_id
        self.image = image
        self.points3d = points3d
        self.points2d = points2d
        self.descriptor = descriptor
        self.base_path = base_path
        self.database_path = database_path
        self.valid_mask = valid_mask
        self.compute_descriptors()

        
    def compute_descriptors(self):

        self.descriptors_local = get_descriptors(self.image_id,self.database_path)[self.valid_mask]
        

 
def build_database(reconstruction_path):
    # Load the database
    reconstruction = pycolmap.Reconstruction(reconstruction_path)

    # Get the images and their points
    images = reconstruction.images
    points3D = reconstruction.points3D

    #loop
    frames=[]
    frames_descriptors = []
    for image_id in images:
        image = images[image_id]
        points = image.points2D
        points_valid = [point  for point in points if point.has_point3D()]
        valid_mask = [point.has_point3D() for point in points]

        points_valid_2D = [point.xy for point in points_valid]
        points_valid_3D = [points3D[point.point3D_id].xyz for point in points_valid]
        frame=Frame(image_id,image,points_valid_3D,points_valid_2D,valid_mask)
        frames.append(frame)
        # descriptor=encoder.model(encoder.preprocess_image(BASE_PATH_DB+'imgs/'+image.name).cuda()).detach().cpu()
        # frames_descriptors.append(descriptor)#(frame.descriptors_local)#(get_descriptors(image_id,database_path))


    # FIND THE IMAGE PATH IN ORDER TO LOAD THE IMAGE AND PRE-COMPUTE THE DESCRIPTORS FOR PLACE RECOGNITION
    image_paths = [frame.base_path + frame.image.name for frame in frames]

    if os.path.exists('colmap_localization/reconstruction/descriptors.pt'):
        frames_descriptors = torch.load('colmap_localization/reconstruction/descriptors.pt')
    else:
        data= [encoder.preprocess_image(image_path) for image_path in image_paths]
        frames_descriptors = [encoder.model(d.cuda()).detach().cpu() for d in data]
        frames_descriptors=torch.cat(frames_descriptors)
        torch.save(frames_descriptors,'colmap_localization/reconstruction/descriptors.pt')
    return frames,frames_descriptors

### LOAD THE GEOREFERENCED DATABASE RECOSNTRUCTION ###   
frames,frames_descriptors=build_database(reconstruction_path)

### FIND THE IMAGE PATH IN ORDER TO LOAD THE IMAGE AND PRE-COMPUTE THE DESCRIPTORS FOR PLACE RECOGNITION ###
image_paths = [frame.base_path + frame.image.name for frame in frames]

if os.path.exists('colmap_localization/reconstruction/descriptors.pt'):
    frames_descriptors = torch.load('colmap_localization/reconstruction/descriptors.pt')
else:
    data= [encoder.preprocess_image(image_path) for image_path in image_paths]
    frames_descriptors = [encoder.model(d.cuda()).detach().cpu() for d in data]
    frames_descriptors=torch.cat(frames_descriptors)
    torch.save(frames_descriptors,'colmap_localization/reconstruction/descriptors.pt')



### FIND THE QUERIES AND LOCALIZE ### 
poses_matches=[]
images=os.listdir(IMG_PATH_QUERY)[:20]
for image_id in range(0,len(images)+0):
    image_path=IMG_PATH_QUERY+images[image_id]
    pose=localize_image(image_path,frames,frames_descriptors,encoder,top_k=1,threshold=0.8)#at least n inliers

    poses_matches.append([images[image_id],pose])

img_sequence_path = 'colmap_localization/reconstruction/waypoint/images.txt' # the sequence of queries in buffer
poses_queries=utils_localization.read_img_sequence_poses_to_matrix(img_sequence_path)


# make output directory, delete if exists and create new
if os.path.exists('colmap_localization/reconstruction/aligned'):
    shutil.rmtree('colmap_localization/reconstruction/aligned')
os.makedirs('colmap_localization/reconstruction/aligned', exist_ok=True)

# save poses_matches to file for the aligner, in format: DJI_20240702140512_0094_l1-mine-m1.JPG 38.749497268 23.395930556 63.405
with open('colmap_localization/reconstruction/aligned/poses_matches.txt', 'w') as f:
    for pose_match in poses_matches:
        if pose_match[1] is not None:
            f.write(f'{pose_match[0]} {pose_match[1][0,3]} {pose_match[1][1,3]} {pose_match[1][2,3]}\n')


# run model aligner usign terminal command
os.system('colmap model_aligner --input_path colmap_localization/reconstruction/waypoint --output_path colmap_localization/reconstruction/aligned --ref_images_path colmap_localization/reconstruction/aligned/poses_matches.txt --alignment_max_error 1 --ref_is_gps 0 --merge_image_and_ref_origins 1 --alignment_type ecef')
# load the aligned reconstruction
poses_aligned=utils_localization.read_img_sequence_poses_to_matrix('colmap_localization/reconstruction/aligned/images.txt')

# check result usign exif data
for image_id in range(0,len(images)+0):
    im_data = utils.read_image(IMG_PATH_QUERY+images[image_id])
    T_gt=utils.make_transformation_matrix_ENU(im_data['gimbal_yrp'],im_data['utm'][0],im_data['utm'][1],im_data['altitude_abs']) # create transformation matrix
    #get the error
    
    # t=np.linalg.inv(poses_aligned[image_id])[:3,3]
    t= poses_aligned[image_id][:3,3]
    # t=pose_to_T_inv(pose['cam_from_world'].rotation.quat,pose['cam_from_world'].translation,w_last=False)[:3,3]
    # ecef to utm
    ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")  # ECEF
    wgs84 = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")  # WGS84 Geodetic
    transformer = pyproj.Transformer.from_proj(ecef, wgs84)
    lon, lat, alt = transformer.transform(*t)  
    x,y,_,_=utm.from_latlon(lat,lon)

    gt_x,gt_y,gt_alt = T_gt[:3,3]
    # rot_error=rotation_error(pose_to_T_inv(pose['cam_from_world'].rotation.quat,pose['cam_from_world'].translation,w_last=False)[:3,:3],T_gt[:3,:3])
    trans_error=np.linalg.norm(np.asarray([x,y,alt])-np.asarray([gt_x,gt_y,gt_alt]))

    print('Translation error:',trans_error)
    # print('Rotational error: ',rot_error) ## needs fixing, it compares ENU with colmap orientation
    print('\n')









