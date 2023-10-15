import os
import warnings
import sys
import gc

sys.path.append('../modulus_scripts')
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
from modulus.sym.models.layers import Activation

from modulus.sym.domain.constraint.continuous import DeepONetConstraint
from modulus.sym.domain.validator.discrete import DeepONet_Data_Validator
from modulus.sym.domain.validator.continuous import PointwiseValidator
from modulus.sym.dataset.discrete import DictGridDataset

from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.domain.inferencer import PointwiseInferencer

from modulus.sym.key import Key

from sympy import Symbol, Eq, Lt

n_points = 16384*8

def get_eval_sample_points(d1, d2, aoa):
    wing = None
    filename = os.path.expandvars("${SCRATCH}/stls/def_{:.3f}_{:.3f}_wing.stl")
    if os.path.isfile(filename.format(d1,d2)):
        pass
    else:
        make_STL([d1,d2],filename.format(d1,d2))
    wing = Tessellation.from_stl(filename.format(d1,d2))
    # wing = wing.rotate(-1*np.deg2rad(aoa),axis='y')
    wing = wing.scale(1/scale["len"])
    
    y = Symbol('y')
    s = wing.sample_boundary(n_points, quasirandom=True, criteria=Lt(y,-0.0031/scale["len"]))
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

def load_data(cfg, res_path, csv_name, n_train=50, n_test=10, n_params = 4):
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
        
        if n_train > csv_data.shape[0]:
            warnings.warn(
                f"Not enough simulations to supply requested n_train"
            )
            n_train = 50
            n_test = 10
        # Train Data
        # Trunk Inputs    
        x_train = np.ndarray((0,1))
        y_train = np.ndarray((0,1))
        z_train = np.ndarray((0,1))
        # Branch Inputs
        params_train = np.ndarray((0,n_params))
        
        # Outputs
        tau_x_train = np.ndarray((0,1))
        tau_y_train = np.ndarray((0,1))
        tau_z_train = np.ndarray((0,1))
        p_train = np.ndarray((0,1))
        #nuT_train = np.ndarray((0,1))
    
        for i in range(n_train):
            vtk_obj = VTKFromFile(res_path+'/'+csv_data[i]["folder"]+'/wing.vtp')
            # Get Points
            points = vtk_obj.get_points()
            
            aoa = np.radians(csv_data[i]["alpha"])
            R = np.array([[np.cos(-aoa), 0, np.sin(-aoa)],[0, 1, 0],[-np.sin(-aoa), 0, np.cos(-aoa)]],dtype='float32')
            
            for it in range(points.shape[0]):
                points[it,:] = R@points[it,:]
            
            x_train = np.concatenate((x_train, points[:,0].reshape((points.shape[0],1))/len_scale), axis = 0)
            y_train = np.concatenate((y_train, points[:,1].reshape((points.shape[0],1))/len_scale), axis = 0)
            z_train = np.concatenate((z_train, points[:,2].reshape((points.shape[0],1))/len_scale), axis = 0)
            # Set Input Vals
            if n_params == 4:
                params = np.array([csv_data[i]["V"]/vel_scale,csv_data[i]["alpha"]/angle_scale,
                csv_data[i]["d1"]/angle_scale,csv_data[i]["d2"]/angle_scale])
            else:
                 params = np.array([csv_data[i]["V"]/vel_scale,csv_data[i]["alpha"]/angle_scale])
            params_train = np.concatenate((params_train, np.full((points.shape[0],n_params),params)), axis = 0)
     
            # Get Outputs
            # Pa = kg/(m*s^2)
            tau = vtk_obj.get_array("wallShearStress")
            
            for it in range(tau.shape[0]):
                tau[it,:] = R@tau[it,:]
            
            tau_x_train = np.concatenate((tau_x_train,tau[:,0]
                            .reshape((points.shape[0],1))*(time_scale**2)/(len_scale**2)), axis=0)
            tau_y_train = np.concatenate((tau_y_train,tau[:,1]
                            .reshape((points.shape[0],1))*(time_scale**2)/(len_scale**2)), axis=0)
            tau_z_train = np.concatenate((tau_z_train,tau[:,2]
                            .reshape((points.shape[0],1))*(time_scale**2)/(len_scale**2)), axis=0)
            p_train = np.concatenate((p_train,(vtk_obj.get_array("p")-101e3)*(time_scale**2)/(len_scale**2)), axis=0)    
            #nuT_train = np.concatenate((nuT_train,vtk_obj.get_array("nuTilda")/(len_scale**2)*time_scale),axis = 0)
            #tau_train = np.concatenate((tau_train,vtk_obj.get_array("wallShearStress"))
        
        print("z", np.max(z_train),np.min(z_train))
        print("tau", np.max(tau_x_train),np.min(tau_x_train))
        
        # Test Data
        # Trunk Inputs    
        x_test = np.ndarray((0,1))
        y_test = np.ndarray((0,1))
        z_test = np.ndarray((0,1))
        # Branch Inputs
        params_test = np.ndarray((0,n_params))
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
            
            aoa = np.radians(csv_data[-(i+1)]["alpha"])
            R = np.array([[np.cos(-aoa), 0, np.sin(-aoa)],[0, 1, 0],[-np.sin(-aoa), 0, np.cos(-aoa)]],dtype='float32')
            
            for it in range(points.shape[0]):
                points[it,:] = R@points[it,:]
            
            x_test = np.concatenate((x_test, points[:,0].reshape((points.shape[0],1))/len_scale), axis = 0)
            y_test = np.concatenate((y_test, points[:,1].reshape((points.shape[0],1))/len_scale), axis = 0)
            z_test = np.concatenate((z_test, points[:,2].reshape((points.shape[0],1))/len_scale), axis = 0)
            # Set Input Vals
            if n_params == 4:
                params = np.array([csv_data[-(i+1)]["V"]/vel_scale,csv_data[-(i+1)]["alpha"]/angle_scale,
                csv_data[-(i+1)]["d1"]/angle_scale,csv_data[-(i+1)]["d2"]/angle_scale])
            else:
                 params = np.array([csv_data[-(i+1)]["V"]/vel_scale,csv_data[-(i+1)]["alpha"]/angle_scale])
            print(params[0], params[1],np.rad2deg(aoa))
            params_test = np.concatenate((params_test, np.full((points.shape[0],n_params),params)), axis = 0) 
            #V_test  = np.concatenate((V_test,np.ones((points.shape[0],1))*csv_data[i]["V"]),axis = 0)
            #alpha_test  = np.concatenate((alpha_test,np.ones((points.shape[0],1))*csv_data[-(i+1)]["alpha"]),axis = 0)
            #d1_test  = np.concatenate((d1_test,np.ones((points.shape[0],1))*csv_data[-(i+1)]["d1"]),axis = 0)
            #d2_test  = np.concatenate((d2_test,np.ones((points.shape[0],1))*csv_data[-(i+1)]["d2"]),axis = 0)
            
            tau = vtk_obj.get_array("wallShearStress")
            
            for it in range(tau.shape[0]):
                tau[it,:] = R@tau[it,:]
                
            # Get Outputs
            tau_x_test = np.concatenate((tau_x_test,tau[:,0]
                           .reshape((points.shape[0],1))*(time_scale**2)/(len_scale**2)), axis=0)
            tau_y_test = np.concatenate((tau_y_test,tau[:,1]
                           .reshape((points.shape[0],1))*(time_scale**2)/(len_scale**2)), axis=0)
            tau_z_test = np.concatenate((tau_z_test,tau[:,2]
                           .reshape((points.shape[0],1))*(time_scale**2)/(len_scale**2)), axis=0)
            p_test = np.concatenate((p_test,(vtk_obj.get_array("p")-101e3)*(time_scale**2)/(len_scale**2)), axis=0)    
            #nuT_test = np.concatenate((nuT_test,vtk_obj.get_array("nuTilda")/(len_scale**2)*time_scale), axis=0)
            #tau_test = np.concatenate((tau_test,vtk_obj.get_array("wallShearStress"), axis=0)
        print("p", np.max(p_test),np.min(p_test))
        print("tau", np.max(tau_x_test),np.min(tau_x_test))
        print(x_train.shape, n_train)
        
        data = {'x_test':x_test,
                'y_test':y_test,
                'z_test':z_test,
                'params_test':params_test,
                'tau_x_test':tau_x_test,
                'tau_y_test':tau_y_test,
                'tau_z_test':tau_z_test,
                'p_test':p_test,
                #'nuT_test':nuT_test,
                'x_train':x_train,
                'y_train':y_train,
                'z_train':z_train,
                'params_train':params_train, 
                'tau_x_train':tau_x_train,
                'tau_y_train':tau_y_train,
                'tau_z_train':tau_z_train,
                'p_train':p_train,
                #'nuT_train':nuT_train, 
               }
        if cfg.run_mode == "eval":
            np.savez("/tmp/data/data.npz", **data)
    return data
    


@modulus.sym.main(config_path="conf", config_name="conf_surf.yaml")
def run(cfg: ModulusConfig):
    global config, domain, nodes
    
    n_params = cfg.custom.n_params
    
    # [init-model]
    trunk_net = instantiate_arch(
        input_keys=[Key("x"),Key("y"),Key("z")],
        cfg=cfg.arch.trunk,
    )
    
    branch_net_tau = instantiate_arch(
        input_keys=[Key("params",n_params)],
        cfg=cfg.arch.branch_tau,
    )
    
    branch_net_p = instantiate_arch(
        input_keys=[Key("params",n_params)],
        cfg=cfg.arch.branch_p,
    )

    deeponet_p = DeepONetArch(
        output_keys=[Key("p")],
        branch_net=branch_net_p,
        trunk_net=trunk_net,
    )
    
    print(deeponet_p)
    
    deeponet_tau = DeepONetArch(
        output_keys=[Key("tau_x"), Key("tau_y"), Key("tau_z")],
        branch_net=branch_net_tau,
        trunk_net=trunk_net,
    )
    
    p_nodes = deeponet_p.make_node("deepo_p")
    tau_nodes = deeponet_tau.make_node("deepo_tau")

    nodes = [p_nodes, tau_nodes]
    
    # [load-data]
    len_scale = cfg.custom.len_scale
    den_scale = cfg.custom.den_scale
    vel_scale =  cfg.custom.v_scale
    angle_scale =  cfg.custom.ang_scale # degrees
    
    set_scale(len_scale, den_scale, vel_scale, angle_scale)
    
    
    
    # Load from dataset csv list
    csv_name = 'morph-wing_dataset_big.csv'
    res_path = os.path.expandvars('${GROUP_HOME}/${USER}/vtk_res')
    data = load_data(cfg, res_path, csv_name, cfg.custom.train_set, cfg.custom.eval_set, n_params)
    #print(type(cfg.loss.weights), cfg.loss.weights)
    print("Loaded Data")
    
    # [constraint]
    # Make Domain
    domain = Domain()

    datacon_tau = DeepONetConstraint.from_numpy(
        nodes = [tau_nodes],
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
        nodes = [p_nodes],
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
    
    print("Created Constraints")
    # [constraint]
    # only use validators in train mode
    if cfg.run_mode == "train":
        # [validator]
        n = int(data['x_test'].shape[0]/3)
        for k in range(3):
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
            print("Creating node {}".format(k))
            validator = PointwiseValidator(
                nodes=nodes,
                invar=invar_valid,
                true_outvar= outvar_valid,
                batch_size=cfg.batch_size.validation,
                plotter=None,
                )
            domain.add_validator(validator, "validator_{}".format(k))
        print("Created Validators")       
        
     
    
    cfg.initialization_network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_surf_four")
    cfg.network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_surf_four")
    
    config = cfg
    
    
def get_config():
    return config
def get_domain():
    return domain
def get_nodes():
    return nodes
    
def solve_nn(cfg, domain, nodes):
    domain.inferencers = {}
    len_scale = scale["len"]
    time_scale = scale["time"]
    mass_scale = scale["mass"]
    vel_scale = scale["vel"]
    angle_scale = scale["angle"]
    n_params = cfg.custom.n_params
    
    
    if cfg.run_mode == "eval":
        cfg.initialization_network_dir = os.path.expandvars("${HOME}/fly-by-feel/modulus_results/morph-wing_surf_no_def_full")
        cfg.network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_surf_big_results")
        sp = {}
        V = cfg.custom.Vinf
        alpha = cfg.custom.alpha
        d1 = cfg.custom.d1
        d2 = cfg.custom.d2
        print(d1, d2)
        # get sample points
        sp = get_eval_sample_points(d1,d2,alpha)
        if n_params == 4:
            params = np.array([V/vel_scale, alpha/angle_scale,
                d1/angle_scale, d2/angle_scale])
        else:
            params = np.array([V/vel_scale, alpha/angle_scale])
        sp["params"] = np.full((sp["x"].shape[0],n_params),params)
        
        # Make monitor
        surface_inf = PointwiseInferencer(
            nodes=nodes,
            invar=sp,
            output_names=["p","tau_z","tau_y","tau_x"],
            batch_size=int(n_points*2),
            )
        domain.add_inferencer(surface_inf, "surf_{:.2f}_{:.2f}".format(d1,d2))
        torch.cuda.empty_cache()
    
    slv = Solver(cfg,domain)
    slv.solve()
    slv = None
    gc.collect()
    torch.cuda.empty_cache()
    
def solve_nn_batch_inf(cfg, domain, nodes, d1_range, d2_range, n_samples=10):
    domain.inferencers = {}
    len_scale = scale["len"]
    time_scale = scale["time"]
    mass_scale = scale["mass"]
    vel_scale = scale["vel"]
    angle_scale = scale["angle"]
    n_params = cfg.custom.n_params
    
    
    if cfg.run_mode == "eval":
        cfg.initialization_network_dir = os.path.expandvars("${HOME}/fly-by-feel/modulus_results/morph-wing_surf_no_def_full")
        cfg.network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_surf_big_results")
        print(cfg)
        V = cfg.custom.Vinf
        alpha = cfg.custom.alpha
        print(V, alpha)
        d1_space = np.linspace(d1_range[0], d1_range[1], n_samples, endpoint=True)
        d2_space = np.linspace(d2_range[0], d2_range[1], n_samples, endpoint=True)
        sp = {}
        # get sample points
        sp = get_eval_sample_points(0.0,0.0, alpha)
        for d1 in d1_space:
            for d2 in d2_space:
                if n_params == 4:
                    params = np.array([V/vel_scale, alpha/angle_scale,
                    d1/angle_scale, d2/angle_scale])
                else:
                    params = np.array([V/vel_scale, alpha/angle_scale])
                sp["params"] = np.full((sp["x"].shape[0],n_params),params)
                # Make inferencer
                surface_inf = PointwiseInferencer(
                    nodes=nodes,
                    invar=sp,
                    output_names=["p","tau_z","tau_y","tau_x"],
                    batch_size=int(n_points/8),
                    )
                domain.add_inferencer(surface_inf, "surf_{:.2f}_{:.2f}".format(d1,d2))
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
