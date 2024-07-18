import pandas as pd
import numpy as np

# Known reference coordinates
reference_points = {
    'samples/db/DJI_20240702140512_0094_l1-mine-m1.jpg': (38.749497268, 23.395930556),
    'samples/db/DJI_20240702140515_0095_l1-mine-m1.jpg': (38.749482968, 23.396143400),
    'samples/db/DJI_20240702140518_0096_l1-mine-m1.jpg': (38.749468339, 23.396362045),
    'samples/db/DJI_20240702140605_0097_l1-mine-m1.jpg': (38.749458721, 23.396489714),
    'samples/db/DJI_20240702140608_0098_l1-mine-m1.jpg': (38.749452040, 23.396560821)
}

# Load georeferenced data from georef.txt
georef_data = pd.read_csv(f'colmap_localization/reconstruction/georef.txt', sep=' ', header=None, names=['image_name', 'lat', 'lon'])

# Function to calculate the difference between reference and georef data
def compare_georef(reference_points, georef_data):
    errors = []
    for image_name, ref_coords in reference_points.items():
        georef_coords = georef_data[georef_data['image_name'] == image_name]
        if not georef_coords.empty:
            diff = np.abs(np.array(ref_coords) - georef_coords[['lat', 'lon']].values[0])
            errors.append((image_name, diff))
    return errors

# Calculate differences
errors = compare_georef(reference_points, georef_data)

# Print errors
for image_name, error in errors:
    print(f'Image: {image_name}, Error: {error}')
else:
    print("Completed without errors")
