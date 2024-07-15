#Mastermine

##Colmap localization:

  -create an initial 3d sparse reconstruction
  -merge if disconnected models
  -georeference the model
  -localize new image in the model & update the model
  -pass camera pose to object estimation


##Object estimation:
  -load the base 3d reconstruction (dense)
  -given an image, read pose from colmap_localization or from exif 
  -given object bbox, find 3d position of object
  -visualize

##Object detection:
-given an image use yolo to find bboxes if any, and pass to object estimation

  
  
  
  
  
