# MasterMine - 3D Object Detection and Localization System

A comprehensive system for detecting objects in images and visualizing them in a 3D point cloud environment. The system combines object detection with optional visual localization to accurately position detected objects in 3D space.

## Features

- Object detection in images
- Optional visual localization using COLMAP
- 3D point cloud visualization
- GPS-based object positioning
- Support for multiple object classes (bulldozer, car, driller, dump truck, excavator, grader, human, truck)
- Support for DJI H20 and L1 camera modules

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for inference speed)
- COLMAP (only required for localization, not for running the demo)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/mastermine.git
    cd mastermine
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. (Optional) Install COLMAP for localization support:
    ```bash
    # Ubuntu/Debian
    sudo apt-get install colmap

    # For other systems:
    # https://colmap.github.io/install.html
    ```

## Project Structure

```
mastermine/
├── demo.py                     # Main demo script for 3D visualization
├── colmap_localization/       # Visual localization module (optional)
│   ├── localization.py         # Core localization functionality
│   ├── reconstruct.py          # COLMAP reconstruction utilities
│   └── georef.py               # Georeferencing utilities
├── object_detection/          # Object detection module
├── object_position_estimation/ # Object position estimation from detections
└── visualization/             # 3D visualization utilities
```

## Setup and Usage

### Running the Demo (No Reconstruction Required)

The demo script visualizes detected objects on a 3D point cloud map. No COLMAP reconstructions are needed to run the demo if you only want to visualize detections with GPS-based positioning.

```bash
python demo.py \
    --images_dir /path/to/query/images \
    --image_ext .JPG \
    --pointcloud /path/to/pointcloud.las \
    --gps True \
    --img_path_db /path/to/database/images \
    --database_path_db /path/to/database.db \
    --reconstruction_path /path/to/reconstruction
```

#### Parameters

- `--images_dir`: Directory containing query images
- `--image_ext`: Image file extension (default: .JPG)
- `--pointcloud`: Path to the point cloud file (.las format)
- `--gps`: Enable GPS-based positioning (default: True)
- `--img_path_db`: Path to database images (used for localization if enabled)
- `--database_path_db`: Path to COLMAP database (only for localization)
- `--reconstruction_path`: Path to COLMAP reconstruction (only for localization)
- `--produce_csv`: Generate CSV output (default: False)

### Visual Localization (Reconstruction Required)

To enable precise visual localization (instead of GPS-based), you must prepare COLMAP reconstructions.

1. **Database Reconstruction (Georeferenced)**:
    - Use your database images
    - Run `create_reconstruction.py` to build a georeferenced model

2. **Query Reconstruction (Non-georeferenced)**:
    - Use your query images
    - Run `create_query_reconstruction.py` to build the model used for matching

The localization pipeline includes:
- Feature extraction
- Feature matching
- Pose estimation
- Georeferencing

## Camera Support

- DJI H20 camera module
- DJI L1 camera module

Each module requires corresponding camera calibration parameters.

## Output

- Real-time 3D visualization of detected objects
- Optional CSV output with object positions and metadata
- Localization error metrics (if visual localization is enabled)

