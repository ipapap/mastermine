import utils
import numpy as np
import cv2
import pymap3d
import scipy
import open3d as o3d
import utm
def main():
    path='/media/gns/CA78173A781724AB/Users/Gns/Downloads/DJI_202309191217_002_Zenmuse-L1-mission/DJI_20230919121746_0007_Zenmuse-L1-mission.JPG'
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
    im_data = utils.read_image(path)
    K,_=cv2.getOptimalNewCameraMatrix(K, dist_coeffs , (im_data['image'].width, im_data['image'].height), 1, (im_data['image'].width, im_data['image'].height))
    K=im_data['K']
    
    print(im_data)

    # convert image coordinates to ned relative to a reference point (first image)
    lla0=im_data['latlon'][0],im_data['latlon'][1],im_data['altitude_abs']#[41.139054068767976, 24.914275053560292,10]  # get a reference point
    im_ned=pymap3d.geodetic2ned(im_data['latlon'][0],im_data['latlon'][1],im_data['altitude_abs'],lla0[0],lla0[1],lla0[2])  # get ned coordinates of the image relative to the reference point

    # create transformation matrix
    T_im2w=utils.make_transformation_matrix(im_data['gimbal_yrp'],im_ned[0],im_ned[1],im_ned[2]) # create transformation matrix
    # T_im2w=utils.make_transformation_matrix_ENU(im_data['gimbal_yrp'],im_data['utm'][0],im_data['utm'][1],im_data['altitude_abs']) # create transformation matrix
    # T_im2w=utils.make_transformation_matrix(im_data['gimbal_yrp'],im_data['utm'][0],im_data['utm'][1],im_data['altitude_abs']) # create transformation matrix
    # T_im2w=utils.make_transformation_matrix(im_data['gimbal_yrp'],im_ned[0],im_ned[1],im_ned[2]) 
    # get point cloud data
    points=utils.get_pcd()
    # visualize the point cloud data along with the camera
    
    #create a dummy point cloud that is a plane initialized by 3 lat lon points, fill that triangle with points
    # points=np.array([[41.1390179,24.9116968,10],[41.13916938876487, 24.91592933276562,10],[41.137452385469395, 24.91446483240224,100]])
    # points=np.hstack([np.array((utm.from_latlon(points[:,0],points[:,1])[:2])).T,points[:,2].T.reshape(-1,1)])



    # convert point cloud data to ned relative to the reference point
    points_ned=utils.get_ned(lla0,points)

    # points_converted=cv2.convertPointsFromHomogeneous((np.linalg.inv(T_im2w)@cv2.convertPointsToHomogeneous(points_ned).squeeze(1).T).T).squeeze(1)
    # points_converted=cv2.convertPointsFromHomogeneous((T_im2w@cv2.convertPointsToHomogeneous(points_ned).squeeze(1).T).T).squeeze(1)
    #visualize the point cloud data with camera
    # utils.visualize_camera(T_im2w,K,points_converted)
    # utils.visualize_camera(np.linalg.inv(T_im2w),K,points_converted)
    utils.visualize_camera(T_im2w,K,points_ned)

    res=utils.project_points(points_ned,K,im_data['image'].width,im_data['image'].height,extrinsic_matrix=np.linalg.inv(T_im2w))

    ## painting only object points
    object_coords=np.array([684,2784])#np.array([630,1665])
    tree= scipy.spatial.cKDTree(res[0][res[1]]) # search only the valid points
    inlrs=tree.query_ball_point((np.array([object_coords[0],object_coords[1]])),50) # for some reason the image is flipped

    active_points=points_ned[res[1]][inlrs]

    #show the active points in red color and the rest in blue
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(active_points)
    point_cloud.paint_uniform_color([0, 0, 0])
    point_cloud_inactive = o3d.geometry.PointCloud()
    point_cloud_inactive.points = o3d.utility.Vector3dVector(points_ned)

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




if __name__ == "__main__":
    main()