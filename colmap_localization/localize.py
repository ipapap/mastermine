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


import utm
import pyproj
BASE_PATH_QUERY = '/home/gns/Documents/terna_colmap_reconstruction/queries/'
DATABASE_PATH_QUERY = BASE_PATH_QUERY + 'database.db'
BASE_PATH_DB = '/home/gns/Documents/terna_colmap_reconstruction/'
DATABASE_PATH_DB = BASE_PATH_DB + 'database.db'

database_path = '/home/gns/Documents/terna_colmap_reconstruction/georeferenced/reconstruction_georef/'

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
    def __init__(self, image_id, image, points3d , points2d,valid_mask, descriptor=None, base_path=BASE_PATH_DB,database_path=DATABASE_PATH_DB):
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
        

    
# Load the database
reconstruction = pycolmap.Reconstruction(database_path)

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



image_paths = [frame.base_path + 'imgs/' + frame.image.name for frame in frames]

if os.path.exists('colmap_localization/reconstruction/descriptors.pt'):
    frames_descriptors = torch.load('colmap_localization/reconstruction/descriptors.pt')
else:
    data= [encoder.preprocess_image(image_path) for image_path in image_paths]
    frames_descriptors = [encoder.model(d.cuda()).detach().cpu() for d in data]
    frames_descriptors=torch.cat(frames_descriptors)
    torch.save(frames_descriptors,'colmap_localization/reconstruction/descriptors.pt')

# frames_descriptors_idx = [d.shape[0] for d in frames_descriptors]
# frames_descriptors_idx = np.cumsum(frames_descriptors_idx)
# frames_descriptors = np.concatenate(frames_descriptors)#torch.tensor(frames_descriptors).squeeze(1)
# tree= scipy.spatial.cKDTree(frames_descriptors) 

# def place_recgnition(db_descriptors,query_descriptors):
    


#load queries
query_frames=[]
query_frames_descriptors = []
# if os.path.exists(DATABASE_PATH_QUERY):
#     os.remove(DATABASE_PATH_QUERY)
# pycolmap.extract_features(DATABASE_PATH_QUERY, BASE_PATH_QUERY+'imgs')

# Load the database


ransac_options = pycolmap.RANSACOptions(
    max_error=4.0,  # for example the reprojection error in pixels
    min_inlier_ratio=0.01,
    confidence=0.9999,
    min_num_trials=1000,
    max_num_trials=100000,
)
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
camera = pycolmap.Camera(
    model='SIMPLE_PINHOLE',
    width=4056,
    height=3040,
    params=[3645.8096052435262, 4056/2, 3040/2],
)

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
        if m.distance < 0.8*n.distance:
            good.append([m])
            inds1.append(m.queryIdx)
            inds2.append(m.trainIdx)
    inds1 = np.array(inds1)
    inds2 = np.array(inds2)

    return good,inds1,inds2



images=os.listdir(BASE_PATH_QUERY+'imgs')

for image_id in range(1,len(images)+1):
    
    # des,kp=get_features(image_id,DATABASE_PATH_QUERY)

    img = Image.open(BASE_PATH_QUERY + "imgs/" + images[image_id - 1]).convert('RGB')
    img = ImageOps.grayscale(img)
    img = np.array(img).astype(np.float32) / 255.
    sift = pycolmap.Sift()

    # Parameters:
    # - image: HxW float array
    kp, des = sift.extract(img) # soooo  slowwww, maybe use cuda
    des = (des *512).astype(np.uint8)

    # des=torch.tensor(des)
    descriptor=encoder.model(encoder.preprocess_image(BASE_PATH_QUERY+'imgs/'+images[image_id-1]).cuda()).detach().cpu()

    dist=torch.cdist(descriptor,frames_descriptors)

    neigh=torch.argsort(dist.view(-1))[:1] # use top x retrieved images
    poses=[]
    points2d=[]
    points3d=[]
    for n in neigh:
        frame = frames[n]
        des1=frame.descriptors_local
        kp1=frame.points2d

        good,inds1,inds2=feature_matching(des,des1)
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
            estimation_options={'ransac':{'max_error':8.0,'min_inlier_ratio':0.001,'confidence':.999,'min_num_trials':2000,'max_num_trials':10000}},
            refinement_options={'refine_focal_length':True,'max_num_iterations':5000,'gradient_tolerance':1e-5},
        )
    
    ## Rest for validating the result with gt 
    if pose is  None: continue
    #get image metadata using pil
    im=Image.open(BASE_PATH_QUERY+'imgs/'+images[image_id-1])
    im_data=im.getxmp()['xmpmeta']['RDF']['Description']
    gt_t=np.array([im_data['GpsLatitude'],im_data['GpsLongitude'],im_data['AbsoluteAltitude']]).astype(float)
    gt_x,gt_y,_,_ = utm.from_latlon(gt_t[0],gt_t[1])
    gt_alt=gt_t[2]
    gt_rpy=np.array([im_data['GimbalYawDegree'],im_data['GimbalRollDegree'],im_data['GimbalPitchDegree'],]).astype(float)

    #get the error
    quat=pose['cam_from_world'].rotation
    t=pose['cam_from_world'].translation
    t=pose_to_T_inv(pose['cam_from_world'].rotation.quat,pose['cam_from_world'].translation,w_last=False)[:3,3]
    # ecef to utm
    ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")  # ECEF
    wgs84 = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")  # WGS84 Geodetic
    transformer = pyproj.Transformer.from_proj(ecef, wgs84)
    lon, lat, alt = transformer.transform(*t)  
    x,y,_,_=utm.from_latlon(lat,lon)


    # #show matches read images as brg
    # img1 = cv2.imread(BASE_PATH_QUERY+'imgs/'+images[image_id-1])
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # img2 = cv2.imread(BASE_PATH_DB+'imgs/'+frame.image.name)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # kp1= [cv2.KeyPoint(x=point[0],y=point[1],size=1) for point in kp1]
    # kp0 = [cv2.KeyPoint(x=point[0],y=point[1],size=1) for point in kp]
    # img3 = cv2.drawMatchesKnn(img1,kp0, img2, kp1, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3)
    # plt.show(block=True)



    # plt.figure(1)
    # plt.imshow(cv2.imread(BASE_PATH_QUERY+'imgs/'+images[image_id-1]))
    # plt.title('Query')
    # plt.figure(2)
    # plt.imshow(cv2.imread(BASE_PATH_DB+'imgs/'+frame.image.name))
    # plt.title('Neighbor')
    # plt.show(block=True)

    print('Translation error:',np.linalg.norm(np.asarray([x,y,alt])-np.asarray([gt_x,gt_y,gt_alt])),'  Num of Inliers:',pose['num_inliers'])







    # # #random neighbors
    # # # neigh = np.random.randint(0, len(frames_descriptors), 5)
    # neigh_path= [image_paths[n] for n in neigh]
    # # #show the neighbors

    # fig,ax=plt.subplots(1,6)
    # ax[0].imshow(cv2.imread(BASE_PATH_QUERY+'imgs/'+images[image_id-1]))
    # ax[0].set_title('Query')
    # for i in range(1,6):
    #     ax[i].imshow(cv2.imread(neigh_path[i-1]))
    #     ax[i].set_title(f'Neighbor {i}')
    # plt.show(block=True)



    
   
# import matplotlib.pyplot as plt
# plt.plot([1, 2, 3], [4, 5, 6])
# plt.show(block=True) 

1-1



# from pyproj import Proj, transform



# transformer1 = pyproj.Transformer.from_crs(
#     {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
#     {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
#     )
# transformer2 = pyproj.Transformer.from_crs(
#     {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
#     {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
#     )
# x,y,z= transformer2.transform(*(38.749482968 ,23.3961434, 236.034 )) #lat,lon,alt
# lon,lat,alt=transformer1.transform(x,y,z)

