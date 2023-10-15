import numpy as np
import os

from modulus.sym.utils.io.vtk import *

scale = None

def set_scale(len_scale, den_scale, vel_scale, angle_scale):
    global scale
    scale = {
        "len":len_scale, 
        "mass":den_scale/(len_scale**3), 
        "vel":vel_scale, 
        "time": len_scale/vel_scale,
        "angle":angle_scale,
    }
    
def eval_inf(data):
    # Set Scales
    len_scale = scale['len']
    mass_scale = scale['mass']
    time_scale = scale['time']
    angle_scale = scale['angle']
    
    print("aoa: ", data["params"][0,1]*angle_scale)
    aoa = np.deg2rad(data["params"][0,1]*angle_scale)
    R = np.array([[np.cos(aoa), 0, np.sin(aoa)],[0, 1, 0],[-np.sin(aoa), 0, np.cos(aoa)]],dtype='float32')
    
    for i in range(data["x"].shape[0]):
        point = np.array([data["x"][i], data["y"][i], data["z"][i]])
        normal = np.array([data["normal_x"][i], data["normal_y"][i], data["normal_z"][i]])
        tau = np.array([data["tau_x"][i], data["tau_y"][i], data["tau_z"][i]])
        #print(point.shape)
        
        point = R@point
        normal = R@normal
        tau = R@tau
        
        data["x"][i], data["y"][i], data["z"][i] = point
        data["normal_x"][i], data["normal_y"][i], data["normal_z"][i] = normal
        data["tau_x"][i], data["tau_y"][i], data["tau_z"][i] = tau
            
    drag = np.sum(data['area']*(len_scale**2)*(-data['normal_x']*
        data['p']*(len_scale**2/(time_scale**2))  - 
        data["tau_x"]*(len_scale**2/(time_scale**2)) ))
        
    lift = np.sum(data['area']*(len_scale**2)*(-data['normal_z']*
        data['p']*(len_scale**2/(time_scale**2)) - 
        data["tau_z"]*(len_scale**2/(time_scale**2)) ))
        
    forces = np.array([drag, 0, lift])
    
    print('EVAL: lift: ', forces[2], " drag: ", forces[0])

# Load Data
res_path = os.path.expandvars("$SCRATCH/modulus/outputs/morph-wing_surf_big_results/inferencers")

data = np.load(res_path+"/surf_0.00_0.00.npz", allow_pickle=True).get('arr_0').item(0)

# Set Scales
len_scale = 0.7
den_scale = 1.2
vel_scale = 25.0 
angle_scale = 20.0 # degrees

set_scale(len_scale, den_scale, vel_scale, angle_scale)

time_scale = scale["time"]
mass_scale = scale["mass"]

print(data["x"].shape)
print(data.keys())

eval_inf(data)

# Convert Data
out_data = {}
out_data['x'] = data['x']*len_scale
out_data['y'] = data['y']*len_scale
out_data['z'] = data['z']*len_scale
out_data['area'] = data['area']*(len_scale**2)
out_data['normal_x'] = data['normal_x']
out_data['normal_y'] = data['normal_y']
out_data['normal_z'] = data['normal_z']

out_data['p'] = data['p']*(len_scale**2/(time_scale**2))
tau = np.array([data['tau_x'][:].flatten(),data['tau_y'][:].flatten(),data['tau_z'][:].flatten()])*len_scale**2/(time_scale**2)
#out_data['nuT'] = data['nu']*(len_scale**2)/time_scale
#print(data['u'][:].flatten())
#U = np.array([data['u'][:].flatten(),data['v'][:].flatten(),data['w'][:].flatten()])*len_scale/time_scale
out_data["tau"] = tau.T
#print(out_data["U"].shape)

out_file = os.path.expandvars("${HOME}/fly-by-feel/morph_wing/surf_{:.3f}_{:.3f}.vtp")

print(data['params'][0]*angle_scale)
d1 = 0 # data['params'][0,2]*angle_scale
d2 = 0 # data['params'][0,3]*angle_scale

var_to_polyvtk(out_data, out_file.format(d1,d2))


