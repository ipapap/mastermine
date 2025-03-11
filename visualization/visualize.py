import open3d as o3d
import numpy as np
import threading
import time

def create_sphere(center, radius=5, color=[1, 0, 1]):
    """
    Creates a sphere mesh at a given center position.
    
    Args:
        center (tuple/list/np.array): XYZ position.
        radius (float): Radius of the sphere.
        color (list): RGB color values (0-1).

    Returns:
        o3d.geometry.TriangleMesh: Sphere mesh.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)  # Apply color
    sphere.translate(center)  # Move to correct location
    return sphere


# Lock for safe threading
points_lock = threading.Lock()

dynamic_objs = []
dynamic_mesh = None 
# added_objects = set()  # Track objects added to Open3D
# Function to run Open3D visualization in the background
def run_visualization(static_points,static_colors,dynamic_points=dynamic_objs):
    # vis = o3d.visualization.Visualizer()
    global dynamic_objs, dynamic_mesh
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D Point Cloud", width=800, height=600)

    static_pcd = o3d.geometry.PointCloud()
    static_pcd.points = o3d.utility.Vector3dVector(static_points)
    static_pcd.colors = o3d.utility.Vector3dVector(static_colors)
    # static_pcd.paint_uniform_color([0, 0, 1])  # Blue color for static points

    
    vis.add_geometry(static_pcd)  # Static point cloud (won't change)
    dynamic_mesh = o3d.geometry.TriangleMesh()

    vis.add_geometry(dynamic_mesh)     

    while True:
        with points_lock:
            if dynamic_mesh is not None:
                vis.update_geometry(dynamic_mesh)

        
        vis.poll_events()
        vis.update_renderer()

        
        time.sleep(0.05)  # Control update speed

# Start Open3D visualization in a separate thread
# vis_thread = threading.Thread(target=run_visualization, daemon=True)
# vis_thread.start()

def update(points,color=[1,0,1],size=3):
    global dynamic_objs
    global dynamic_mesh
    with points_lock:

        dynamic_objs = [create_sphere(center) for center in points]


    with points_lock:
        # Create a merged mesh with all spheres
        new_mesh = o3d.geometry.TriangleMesh()
        
        for center in points:
            sphere = create_sphere(center, radius=size, color=color)
            new_mesh += sphere  # Merge all spheres into one mesh
        if dynamic_mesh is None:
            dynamic_mesh=new_mesh
     
        dynamic_mesh.clear()  # Clear previous mesh
        dynamic_mesh.vertices = new_mesh.vertices
        dynamic_mesh.triangles = new_mesh.triangles
        dynamic_mesh.vertex_colors = new_mesh.vertex_colors

