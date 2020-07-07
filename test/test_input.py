import matplotlib.pyplot as  plt 
import numpy as np 
import torch 
import pickle
import ipdb 
st = ipdb.set_trace
dd = pickle.load(open("/Users/shamitlal/Desktop/temp/monodep_inp.p","rb"))
rgb_m = dd[('color', -1, 2)].permute(0,2,3,1).numpy()[0]
rgb_0 = dd[('color', 0, 2)].permute(0,2,3,1).numpy()[0]
rgb_p = dd[('color', 1, 2)].permute(0,2,3,1).numpy()[0]

rgb = np.concatenate((rgb_m, rgb_0, rgb_p), axis=0)
plt.imshow(rgb)
plt.show(block=True)

