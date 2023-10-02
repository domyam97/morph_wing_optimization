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
    
def eval_inf():
    # Set Scales
    len_scale = scale['len']
    mass_scale = scale['mass']
    time_scale = scale['time']
    angle_scale = scale['angle']
    
    res_dir = os.path.expandvars("$SCRATCH/modulus/outputs/morph-wing_surf_big_results/inferencers")
    for file in os.listdir(res_dir):
        if file.startswith("surf") & file.endswith(".npz"):
            data = np.load(os.path.join(res_dir,file), allow_pickle=True).get('arr_0').item(0)
            
            drag = -np.sum(data['area']*(len_scale**2)*(data['normal_x']*
                data['p']*mass_scale/(len_scale*(time_scale**2)) + 
                data["tau_x"]*mass_scale/(len_scale*(time_scale**2)) ))
                
            lift = -np.sum(data['area']*(len_scale**2)*(data['normal_z']*
                data['p']*mass_scale/(len_scale*(time_scale**2)) + 
                data["tau_z"]*mass_scale/(len_scale*(time_scale**2)) ))
    
    print('EVAL: lift: ', lift, " drag: ", drag)

# Load Data
res_path = os.path.expandvars("$SCRATCH/modulus/outputs/morph-wing_surf_big_results/inferencers")

data = np.load(res_path+"/surf_0.00_0.00.npz", allow_pickle=True).get('arr_0').item(0)

# Set Scales
len_scale = 0.7
den_scale = 1.2
vel_scale = 20.0 
angle_scale = 15.0 # degrees

set_scale(len_scale, den_scale, vel_scale, angle_scale)

time_scale = scale["time"]
mass_scale = scale["mass"]

print(data["x"].shape)
print(data.keys())

# Convert Data
out_data = {}
out_data['x'] = data['x']*len_scale
out_data['y'] = data['y']*len_scale
out_data['z'] = data['z']*len_scale
out_data['area'] = data['area']*(len_scale**2)
#out_data['normal_x'] = data['normal_x']
#out_data['normal_y'] = data['normal_y']
#out_data['normal_z'] = data['normal_z']

out_data['p'] = data['p']*mass_scale/(len_scale*(time_scale**2))
tau = np.array([data['tau_x'][:].flatten(),data['tau_y'][:].flatten(),data['tau_z'][:].flatten()])*mass_scale/(len_scale*(time_scale**2))
#out_data['nuT'] = data['nu']*(len_scale**2)/time_scale
#print(data['u'][:].flatten())
#U = np.array([data['u'][:].flatten(),data['v'][:].flatten(),data['w'][:].flatten()])*len_scale/time_scale
out_data["tau"] = tau.T
#print(out_data["U"].shape)

out_file = os.path.expandvars("${HOME}/fly-by-feel/morph_wing/surf_{:.3f}_{:.3f}.vtp")

print(data['params'][0]*angle_scale)
d1 = data['params'][0,2]*angle_scale
d2 = data['params'][0,3]*angle_scale

var_to_polyvtk(out_data, out_file.format(d1,d2))

eval_inf()


