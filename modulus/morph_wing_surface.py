import os
import warnings
import sys
import gc

sys.path.append('../modulus')
from airfoilgen import make_STL

import numpy as np
import torch

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain

from modulus.sym.geometry.tessellation import Tessellation
from modulus.sym.utils.io.vtk import *
from modulus.sym.utils.io import csv_to_dict

from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.models.deeponet import DeepONetArch

from modulus.sym.domain.constraint.continuous import DeepONetConstraint
from modulus.sym.domain.validator.discrete import GridValidator
from modulus.sym.dataset.discrete import DictGridDataset

from modulus.sym.domain.monitor import PointwiseMonitor

from modulus.sym.key import Key

from sympy import Symbol, Eq, Lt

n_points = 6000

def get_eval_sample_points(d1, d2, aoa):
    wing = None
    filename = "/tmp/stls/def_{:.5f}_{:.5f}_wing.stl"
    if os.path.isfile(filename.format(d1,d2)):
        pass
    else:
        make_STL([d1,d2],filename.format(d1,d2))
    wing = Tessellation.from_stl(filename.format(d1,d2))
    wing = wing.rotate(np.deg2rad(aoa),axis='y')
    wing = wing.scale(scale["len"])
    
    y = Symbol('y')
    s = wing.sample_boundary(n_points, criteria=Lt(y, -0.002))
    return s

rng = np.random.default_rng(0)

config = None
domain = None

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

def load_data(res_path, csv_name, n_train=50, n_test=10):
    if os.path.isfile("/tmp/data/data.npz"):
        data = np.load("/tmp/data/data.npz")
    else:
        dtype = np.dtype({'names':['V','alpha','d1','d2','folder'], 'formats':[np.float32, np.float32, np.float32, np.float32, 'U25']})
        csv_data = np.loadtxt(res_path+'/'+csv_name, dtype=dtype, skiprows=1, delimiter=',')
        csv_data = rng.permutation(csv_data,axis=0)
         
        len_scale = scale["len"]
        vel_scale = scale["vel"]
        time_scale = scale["time"]
        mass_scale = scale["mass"]
        angle_scale = scale["angle"]
        
        if (n_train+n_test) > csv_data.shape[0]:
            warnings.warn(
                f"Not enough simulations to supply requested n_sims and n_test"
            )
            n_train = 50
            n_test = 10
        # Train Data
        # Trunk Inputs    
        x_train = np.ndarray((0,1))
        y_train = np.ndarray((0,1))
        z_train = np.ndarray((0,1))
        # Branch Inputs
        params_train = np.ndarray((0,4))
        #V_train = np.ndarray((0,1))
        #alpha_train = np.ndarray((0,1))
        #d1_train = np.ndarray((0,1))
        #d2_train = np.ndarray((0,1))
        
        # Outputs
        tau_x_train = np.ndarray((0,1))
        tau_y_train = np.ndarray((0,1))
        tau_z_train = np.ndarray((0,1))
        p_train = np.ndarray((0,1))
        #nuT_train = np.ndarray((0,1))
        #tau_train = np.ndarray((0,3))
    
        for i in range(n_train):
            vtk_obj = VTKFromFile(res_path+'/'+csv_data[i]["folder"]+'/wing.vtp')
            # Get Points
            points = vtk_obj.get_points()
            x_train = np.concatenate((x_train, points[:,0].reshape((points.shape[0],1))/len_scale), axis = 0)
            y_train = np.concatenate((y_train, points[:,1].reshape((points.shape[0],1))/len_scale), axis = 0)
            z_train = np.concatenate((z_train, points[:,2].reshape((points.shape[0],1))/len_scale), axis = 0)
            # Set Input Vals
            params = np.array([csv_data[i]["V"]/vel_scale,csv_data[i]["alpha"]/angle_scale,
                csv_data[i]["d1"]/angle_scale,csv_data[i]["d2"]/angle_scale])
            params_train = np.concatenate((params_train, np.full((points.shape[0],4),params)), axis = 0) 
            #V_train  = np.concatenate((V_train,np.ones((points.shape[0],1))*csv_data[i]["V"]),axis = 0)
            #alpha_train  = np.concatenate((alpha_train,np.ones((points.shape[0],1))*csv_data[i]["alpha"]),axis = 0)
            #d1_train  = np.concatenate((d1_train,np.ones((points.shape[0],1))*csv_data[i]["d1"]),axis = 0)
            #d2_train  = np.concatenate((d2_train,np.ones((points.shape[0],1))*csv_data[i]["d2"]),axis = 0)
     
            # Get Outputs
            # Pa = kg/(m*s^2)
            tau_x_train = np.concatenate((tau_x_train,vtk_obj.get_array("wallShearStress")[:,0]
                            .reshape((points.shape[0],1))*(time_scale**2)*len_scale/mass_scale), axis=0)
            tau_y_train = np.concatenate((tau_y_train,vtk_obj.get_array("wallShearStress")[:,1]
                            .reshape((points.shape[0],1))*(time_scale**2)*len_scale/mass_scale), axis=0)
            tau_z_train = np.concatenate((tau_z_train,vtk_obj.get_array("wallShearStress")[:,2]
                            .reshape((points.shape[0],1))*(time_scale**2)*len_scale/mass_scale), axis=0)
            p_train = np.concatenate((p_train,(vtk_obj.get_array("p")-101e3)*(time_scale**2)*len_scale/mass_scale), axis=0)    
            #nuT_train = np.concatenate((nuT_train,vtk_obj.get_array("nuTilda")/mass_scale*len_scale*time_scale),axis = 0)
            #tau_train = np.concatenate((tau_train,vtk_obj.get_array("wallShearStress"))
        # Test Data
        # Trunk Inputs    
        x_test = np.ndarray((0,1))
        y_test = np.ndarray((0,1))
        z_test = np.ndarray((0,1))
        # Branch Inputs
        params_test = np.ndarray((0,4))
        #V_test = np.ndarray((0,1))
        #alpha_test = np.ndarray((0,1))
        #d1_test = np.ndarray((0,1))
        #d2_test = np.ndarray((0,1))
    
        # Outputs
        tau_x_test = np.ndarray((0,1))
        tau_y_test = np.ndarray((0,1))
        tau_z_test = np.ndarray((0,1))
        p_test = np.ndarray((0,1))
        #nuT_test = np.ndarray((0,1))
        #tau_test = np.ndarray((0,3))
    
        for i in range(n_test):
            vtk_obj = VTKFromFile(res_path+'/'+csv_data[-(i+1)]["folder"]+'/wing.vtp')
            # Get Points
            points = vtk_obj.get_points()
            x_test = np.concatenate((x_test, points[:,0].reshape((points.shape[0],1))/len_scale), axis = 0)
            y_test = np.concatenate((y_test, points[:,1].reshape((points.shape[0],1))/len_scale), axis = 0)
            z_test = np.concatenate((z_test, points[:,2].reshape((points.shape[0],1))/len_scale), axis = 0)
            # Set Input Vals
            params = np.array([csv_data[-(i+1)]["V"]/vel_scale, csv_data[-(i+1)]["alpha"]/angle_scale,
                csv_data[-(i+1)]["d1"]/angle_scale, csv_data[-(i+1)]["d2"]/angle_scale])
            params_test = np.concatenate((params_test, np.full((points.shape[0],4),params)), axis = 0) 
            #V_test  = np.concatenate((V_test,np.ones((points.shape[0],1))*csv_data[i]["V"]),axis = 0)
            #alpha_test  = np.concatenate((alpha_test,np.ones((points.shape[0],1))*csv_data[i]["alpha"]),axis = 0)
            #d1_test  = np.concatenate((d1_test,np.ones((points.shape[0],1))*csv_data[i]["d1"]),axis = 0)
            #d2_test  = np.concatenate((d2_test,np.ones((points.shape[0],1))*csv_data[i]["d2"]),axis = 0)
     
            # Get Outputs
            tau_x_test = np.concatenate((tau_x_test,vtk_obj.get_array("wallShearStress")[:,0]
                           .reshape((points.shape[0],1))*(time_scale**2)*len_scale/mass_scale), axis=0)
            tau_y_test = np.concatenate((tau_y_test,vtk_obj.get_array("wallShearStress")[:,1]
                           .reshape((points.shape[0],1))*(time_scale**2)*len_scale/mass_scale), axis=0)
            tau_z_test = np.concatenate((tau_z_test,vtk_obj.get_array("wallShearStress")[:,2]
                           .reshape((points.shape[0],1))*(time_scale**2)*len_scale/mass_scale), axis=0)
            p_test = np.concatenate((p_test,(vtk_obj.get_array("p")-101e3)*(time_scale**2)*len_scale/mass_scale), axis=0)    
            #nuT_test = np.concatenate((nuT_test,vtk_obj.get_array("nuTilda")*len_scale*time_scale/mass_scale), axis=0)
            #tau_test = np.concatenate((tau_test,vtk_obj.get_array("wallShearStress"), axis=0)
        print(x_train.shape, n_train)
        
        data = {'x_test':x_test,
                'y_test':y_test,
                'z_test':z_test,
                'params_test':params_test, 
                #'V_test':V_test, 
                #'alpha_test':alpha_test, 
                #'d1_test':d1_test, 
                #'d2_test':d2_test, 
                'tau_x_test':tau_x_test,
                'tau_y_test':tau_y_test,
                'tau_z_test':tau_z_test,
                'p_test':p_test,
                #'nuT_test':nuT_test,
                #'tau_test':tau_test,
                'x_train':x_train,
                'y_train':y_train,
                'z_train':z_train,
                'params_train':params_train, 
                #'V_train':V_train, 
                #'alpha_train':alpha_train, 
                #'d1_train':d1_train, 
                #'d2_train':d2_train, 
                'tau_x_train':tau_x_train,
                'tau_y_train':tau_y_train,
                'tau_z_train':tau_z_train,
                'p_train':p_train,
                #'nuT_train':nuT_train,
                #'tau_train':tau_train, 
               }
        np.savez("/tmp/data/data.npz", **data)
    return data
    


@modulus.sym.main(config_path="conf", config_name="conf_surf.yaml")
def run(cfg: ModulusConfig):
    global config, domain, nodes
    cfg.optimizer.lr = 0.0008
    # [init-model]
    trunk_net = FullyConnectedArch(
        input_keys=[Key("x"),Key("y"),Key("z")],
        output_keys=[Key("trunk", 128)],
    )
    branch_net_tau = FullyConnectedArch(
        input_keys=[Key("params",4)],
        output_keys=[Key("branch", 128)],
    )
    branch_net_p = FullyConnectedArch(
        input_keys=[Key("params",4)],
        output_keys=[Key("branch", 128)],
    )
    #branch_net_nu = FullyConnectedArch(
    #    input_keys=[Key("params",4)],
    #    output_keys=[Key("branch", 128)],
    #)

    deeponet_p = DeepONetArch(
        output_keys=[Key("p")],
        branch_net=branch_net_p,
        trunk_net=trunk_net,
    )
    deeponet_tau = DeepONetArch(
        output_keys=[Key("tau_x"), Key("tau_y"), Key("tau_z")],
        branch_net=branch_net_tau,
        trunk_net=trunk_net,
    )
    #deeponet_nuT = DeepONetArch(
    #    output_keys=[Key("nu")],
    #    branch_net=branch_net_nu,
    #    trunk_net=trunk_net,
    #)
    deepo_nodes = [
        deeponet_p.make_node("deepo_p"), 
        deeponet_tau.make_node("deepo_tau")
    ]

    nodes = deepo_nodes
    
    # [load-data]
    len_scale = 0.7
    den_scale = 1.2
    vel_scale = 20.0 
    angle_scale = 15.0 # degrees
    
    set_scale(len_scale, den_scale, vel_scale, angle_scale)
    
    
    
    # Load from dataset csv list
    csv_name = 'morph-wing_dataset_big.csv'
    res_path = os.path.expandvars('${GROUP_HOME}/${USER}/vtk_res')
    data = load_data(res_path, csv_name, 225,25)
 
    # [constraint]
    # Make Domain
    domain = Domain()

    datacon_tau = DeepONetConstraint.from_numpy(
        nodes = nodes,
        invar = {
            "x":data["x_train"],
            "y":data["y_train"],
            "z":data["z_train"],
            "params":data["params_train"],
            },
        outvar={
            "tau_x":data["tau_x_train"],
            "tau_y":data["tau_y_train"],
            "tau_z":data["tau_z_train"],
            },
        batch_size=cfg.batch_size.train,
        )
    domain.add_constraint(datacon_tau, "data_tau")

    datacon_p = DeepONetConstraint.from_numpy(
        nodes = nodes,
        invar = {
            "x":data["x_train"],
            "y":data["y_train"],
            "z":data["z_train"],
            "params":data["params_train"],
            },
        outvar={
            "p":data["p_train"],
        },
        batch_size=cfg.batch_size.train,
        )
    domain.add_constraint(datacon_p, "data_p")
    
    
    # [constraint]
    # only use validators in train mode
    if cfg.run_mode == "train":
        # [validator]
        for k in range(3):
            n = int(data['x_test'].shape[0]/3)
            invar_valid= {
                'x': data['x_test'][k*n:(k+1)*n],
                'y': data['y_test'][k*n:(k+1)*n],
                'z': data['z_test'][k*n:(k+1)*n],
                'params': data['params_test'][k*n:(k+1)*n],
                }
            outvar_valid={
                'tau_x': data['tau_x_test'][k*n:(k+1)*n],
                'tau_y': data['tau_y_test'][k*n:(k+1)*n],
                'tau_z': data['tau_z_test'][k*n:(k+1)*n],
                'p': data['p_test'][k*n:(k+1)*n],
                #'nu': data['nuT_test'][k*n:(k+1)*n],
                }
            dataset = DictGridDataset(invar_valid, outvar_valid)
         
            validator = GridValidator(nodes=nodes,dataset=dataset, plotter=None)
            domain.add_validator(validator, "validator_{}".format(k))
        
        
        
    
    cfg.initialization_network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_surf_big")
    cfg.network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_surf_big")
    
    config = cfg
    
    
def get_config():
    return config
def get_domain():
    return domain
def get_nodes():
    return nodes
    
def solve_nn(cfg, domain, nodes):
    domain.monitors = {}
    len_scale = scale["len"]
    time_scale = scale["time"]
    mass_scale = scale["mass"]
    vel_scale = scale["vel"]
    angle_scale = scale["angle"]
    
    
    if cfg.run_mode == "eval":
        cfg.initialization_network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_surf_big_cos_annealing_lr_8e-4")
        cfg.network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_surf_big_results")
        sp = {}
        V = cfg.custom.Vinf
        alpha = cfg.custom.alpha
        d1 = cfg.custom.d1
        d2 = cfg.custom.d2
        print(d1, d2)
        # get sample points
        sp = get_eval_sample_points(d1,d2,alpha)
        params = np.array([V/vel_scale, alpha/angle_scale,
            d1/angle_scale, d2/angle_scale])
        sp["params"] = np.full((sp["x"].shape[0],4),params)
        
        # Make monitor
        ld_mon = PointwiseMonitor(
            invar = sp,
            output_names = ["p","tau_z", "tau_x"],
            metrics = {'lift': lambda var: -torch.sum(var["area"]*(var["normal_z"]*(var["p"]*mass_scale/(len_scale*(time_scale**2))) +
                var["tau_z"]*mass_scale/(len_scale*(time_scale**2)) )),
                
                'drag': lambda var: -torch.sum(var["area"]*(var["normal_x"]*(var["p"]*mass_scale/(len_scale*(time_scale**2))) +
                var["tau_x"]*mass_scale/(len_scale*(time_scale**2)) )),
            },
            nodes=nodes,
            )
        domain.add_monitor(ld_mon)
    
    slv = Solver(cfg,domain)
    slv.solve()
    slv = None
    gc.collect()
    torch.cuda.empty_cache()
    
def solve_nn_batch(cfg, domain, nodes, d1_range, d2_range, n_samples=10):
    domain.monitors = {}
    len_scale = scale["len"]
    time_scale = scale["time"]
    mass_scale = scale["mass"]
    vel_scale = scale["vel"]
    angle_scale = scale["angle"]
    
    
    if cfg.run_mode == "eval":
        cfg.initialization_network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_surf_big_cos_annealing_lr_8e-4")
        cfg.network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_surf_big_results")
        V = cfg.custom.Vinf
        alpha = cfg.custom.alpha
        d1_space = np.linspace(d1_range[0], d1_range[1], n_samples, endpoint=True)
        d2_space = np.linspace(d2_range[0], d2_range[1], n_samples, endpoint=True)
        for d1 in d1_space:
            for d2 in d2_space:
                sp = {}
                # get sample points
                sp = get_eval_sample_points(d1,d2,alpha)
                params = np.array([V/vel_scale, alpha/angle_scale,
                    d1/angle_scale, d2/angle_scale])
                sp["params"] = np.full((sp["x"].shape[0],4),params)
                l_mon = "lift_{}_{}_mon".format(d1,d2)
                d_mon = "drag_{}_{}_mon".format(d1,d2)
                # Make monitor
                ld_mon = PointwiseMonitor(
                    invar = sp,
                    output_names = ["p","tau_z", "tau_x"],
                    metrics = {l_mon: lambda var: -torch.sum(var["area"]*(var["normal_z"]*var["p"]*mass_scale/(len_scale*(time_scale**2)))), #+ 
                        #var["tau_z"]*mass_scale/(len_scale*(time_scale**2)) )),
                        
                        d_mon: lambda var: -torch.sum(var["area"]*(var["normal_x"]*var["p"]*mass_scale/(len_scale*(time_scale**2)) + 
                        var["tau_x"]*mass_scale/(len_scale*(time_scale**2)) )),
                    },
                    nodes=nodes,
                    )
                domain.add_monitor(ld_mon)
                torch.cuda.empty_cache()
                
    slv = Solver(cfg,domain)
    slv.solve()
    slv = None
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    #for key, val in os.environ.items():
    #    print(key, val)
    #print(os.environ.get("SLURM_LAUNCH_NODE_IPADDR"))
    #print(os.environ.get("HOSTNAME"))
    torch.cuda.empty_cache()
    run()
    solve_nn(config, domain, nodes)
