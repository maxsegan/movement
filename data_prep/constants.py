import numpy as np

# H36M edges (0-indexed)
H36M_I = np.array([0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15], dtype=np.int32)
H36M_J = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], dtype=np.int32)

# Optional left/right mask for edges and colors
H36M_LR_EDGE_MASK = np.array([0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0], dtype=bool)
COLOR_L_BGR = (0, 255, 0)
COLOR_R_BGR = (255, 0, 0)
