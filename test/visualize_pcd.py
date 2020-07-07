import open3d as o3d
import pickle
import numpy as np 
import pydisco_utils
p = pickle.load(open('/Users/shamitlal/Desktop/temp/monodepth2/15939817268639078_depth.p', 'rb'))
xyz_camX = p['xyz_camX']
pcd_camX = pydisco_utils.make_pcd(xyz_camX)
o3d.visualization.draw_geometries([pcd_camX])