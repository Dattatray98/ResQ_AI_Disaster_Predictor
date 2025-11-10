import rasterio #type: ignore
import numpy as np

file_path = "../data/raw/output_hh.tif"

dem = rasterio.open(file_path)

elevation = dem.read(1)
print("dem shape : ", elevation.shape)
print("Elevatioin range : ", np. min(elevation), "to", np.max(elevation))