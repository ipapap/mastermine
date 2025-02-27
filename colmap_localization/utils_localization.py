import numpy as np
import scipy

# img_sequence_path = 'colmap_localization/reconstruction/reconstruction_georef/images.txt'
# txt: image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name / if line not begins with int, skip
def read_img_sequence_poses_to_matrix(img_sequence_path):
    with open(img_sequence_path, 'r') as f:
        lines = f.readlines()
    poses = []
    flag=True
    for line in lines:
        if line[0] == '#':
            continue
        if flag==True:
            line = line.split()
            pose = np.eye(4)
            pose[:3, :3] = scipy.spatial.transform.Rotation.from_quat([float(line[1]),float(line[2]), float(line[3]), float(line[4])],scalar_first=True).as_matrix()
            pose[:3, 3] = [float(line[5]), float(line[6]), float(line[7])]
            poses.append(pose)
            flag=False
        else:
            flag=True
    return np.array(poses)
