# Mastermine
Create a new conda env
---------------------
```
$ conda create -n mastermine python=3.8 -y
$ conda activate mastermine

```


Install the requirements
---------------------

```
$ conda install -c conda-forge boost eigen=3.3.7 freeimage flann glew glog \
    gflags sqlite libpng cgal suitesparse ceres-solver cgal
$ sudo apt install libcgal-dev libcgal-qt5-dev
$ pip install -r requirements.txt
```
Download and Install colmap
---------------------
We used colmap 3.10 version.
Git clone colmap here[https://github.com/colmap/colmap/releases/tag/3.10]
```
$ cd ~/colmap-3.10
$ mkdir build
$ cd build
$ cmake ..
$ make -j$(nproc)
```
if you encounter cuda error you can build without cuda 
```
$ cmake .. -DCUDA_ENABLED=OFF
```

Annotation tool used for class detection
---------------------
Roboflow mastermine[https://app.roboflow.com/duth-41ltl/mastermine/]


## Colmap localization:

  -create an initial 3d sparse reconstruction
  
  -merge if disconnected models

  -georeference the model
  
  -localize new image in the model & update the model
  
  -pass camera pose to object estimation


## Object estimation:

  -load the base 3d reconstruction (dense)
  
  -given an image, read pose from colmap_localization or from exif 
  
  -given object bbox, find 3d position of object
  
  -visualize

## Object detection:

-given an image use yolo to find bboxes if any, and pass to object estimation

  
  
  
  
  
