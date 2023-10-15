import os
import warnings
import sys
import gc

import numpy as np
import torch


sys.path.append('../modulus_scripts')
from airfoilgen import make_STL

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain

from modulus.sym.geometry.tessellation import Tessellation
from modulus.sym.geometry.primitives_3d import Box
from modulus.sym.utils.io.vtk import *
from modulus.sym.utils.io import csv_to_dict

from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.models.deeponet import DeepONetArch
from modulus.sym.models.layers import Activation

from modulus.sym.eq.pdes.navier_stokes import NavierStokes

from modulus.sym.domain.constraint.continuous import DeepONetConstraint
from modulus.sym.domain.validator.discrete import GridValidator
from modulus.sym.dataset.discrete import DictGridDataset

from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.domain.inferencer import PointwiseInferencer

from modulus.sym.key import Key

from sympy import Symbol, Eq, Lt
from sympy import Function, Number

from modulus.sym.node import Node
        

n_points = 16384
n_points_vol = 16384*4

rng = np.random.default_rng()

def get_eval_sample_points(d1, d2, aoa):
    wing = None
    filename = os.path.expandvars("${SCRATCH}/stls/def_{:.3f}_{:.3f}_wing.stl")
    if os.path.isfile(filename.format(d1,d2)):
        pass
    else:
        make_STL([d1,d2],filename.format(d1,d2))
    wing = Tessellation.from_stl(filename.format(d1,d2))
    wing = wing.rotate(-1*np.deg2rad(aoa),axis='y')
    box =  Box(point_1=(-0.3, -0.6, -0.2), point_2=(0.7, 0, 0.2))
    
    vol = box-wing
    vol = vol.scale(1/scale["len"])
    wing = wing.scale(1/scale["len"])
    
    y = Symbol('y')
    
    s = {"vol":None, "surf":None}
    
    s["vol"] = vol.sample_interior(n_points_vol, quasirandom=True)
    s["surf"] = wing.sample_boundary(n_points, quasirandom=True)
    print(s.keys())
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

def load_data(cfg, res_path, csv_name, n_train=50, n_test=10):
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
        
        # Outputs
        nuT_train = np.ndarray((0,1))
        p_train = np.ndarray((0,1))
        u_train = np.ndarray((0,1))
        v_train = np.ndarray((0,1))
        w_train = np.ndarray((0,1))
        #tau_train = np.ndarray((0,3))
    
        for i in range(n_train):
            vtk_obj = VTKFromFile(res_path+'/'+csv_data[i]["folder"]+'/internal.vtu')
            # Get Points
            points = vtk_obj.get_points()
            x_train = np.concatenate((x_train, points[:,0].reshape((points.shape[0],1))/len_scale), axis = 0)
            y_train = np.concatenate((y_train, points[:,1].reshape((points.shape[0],1))/len_scale), axis = 0)
            z_train = np.concatenate((z_train, points[:,2].reshape((points.shape[0],1))/len_scale), axis = 0)
            # Set Input Vals
            params = np.array([csv_data[i]["V"]/vel_scale,csv_data[i]["alpha"]/angle_scale,
                csv_data[i]["d1"]/angle_scale,csv_data[i]["d2"]/angle_scale])
            params_train = np.concatenate((params_train, np.full((points.shape[0],4),params)), axis = 0) 
     
            # Get Outputs
            # OpenFoam pressure = p/rho = [m^2/s^2]
            u_train = np.concatenate((u_train,vtk_obj.get_array("U")[:,0]
                            .reshape((points.shape[0],1))*time_scale/len_scale), axis=0)
            v_train = np.concatenate((v_train,vtk_obj.get_array("U")[:,1]
                            .reshape((points.shape[0],1))*time_scale/len_scale), axis=0)
            w_train = np.concatenate((w_train,vtk_obj.get_array("U")[:,2]
                            .reshape((points.shape[0],1))*time_scale/len_scale), axis=0)
            p_train = np.concatenate((p_train,(vtk_obj.get_array("p")-101e3)*(time_scale**2)/(len_scale**2)), axis=0)    
            nuT_train = np.concatenate((nuT_train,vtk_obj.get_array("nut")/(len_scale**2)*time_scale),axis = 0)
            #tau_train = np.concatenate((tau_train,vtk_obj.get_array("wallShearStress"))
        # Test Data
        # Trunk Inputs    
        x_test = np.ndarray((0,1))
        y_test = np.ndarray((0,1))
        z_test = np.ndarray((0,1))
        # Branch Inputs
        params_test = np.ndarray((0,4))
    
        # Outputs
        nuT_test = np.ndarray((0,1))
        p_test = np.ndarray((0,1))
        u_test = np.ndarray((0,1))
        v_test = np.ndarray((0,1))
        w_test = np.ndarray((0,1))
        #tau_test = np.ndarray((0,3))
    
        for i in range(n_test):
            vtk_obj = VTKFromFile(res_path+'/'+csv_data[-(i+1)]["folder"]+'/internal.vtu')
            # Get Points
            points = vtk_obj.get_points()
            x_test = np.concatenate((x_test, points[:,0].reshape((points.shape[0],1))/len_scale), axis = 0)
            y_test = np.concatenate((y_test, points[:,1].reshape((points.shape[0],1))/len_scale), axis = 0)
            z_test = np.concatenate((z_test, points[:,2].reshape((points.shape[0],1))/len_scale), axis = 0)
            # Set Input Vals
            params = np.array([csv_data[-(i+1)]["V"]/vel_scale, csv_data[-(i+1)]["alpha"]/angle_scale,
                csv_data[-(i+1)]["d1"]/angle_scale, csv_data[-(i+1)]["d2"]/angle_scale])
            params_test = np.concatenate((params_test, np.full((points.shape[0],4),params)), axis = 0)
     
            # Get Outputs
            u_test = np.concatenate((u_test,vtk_obj.get_array("U")[:,0]
                            .reshape((points.shape[0],1))*time_scale/len_scale), axis=0)
            v_test = np.concatenate((v_test,vtk_obj.get_array("U")[:,1]
                            .reshape((points.shape[0],1))*time_scale/len_scale), axis=0)
            w_test = np.concatenate((w_test,vtk_obj.get_array("U")[:,2]
                            .reshape((points.shape[0],1))*time_scale/len_scale), axis=0)
            p_test = np.concatenate((p_test,(vtk_obj.get_array("p")-101e3)*(time_scale**2)/(len_scale**2)), axis=0)    
            nuT_test = np.concatenate((nuT_train,vtk_obj.get_array("nut")/(len_scale**2)*time_scale),axis = 0)
            #tau_test = np.concatenate((tau_test,vtk_obj.get_array("wallShearStress"), axis=0)
        print(x_train.shape, n_train)
        
        data = {'x_test':x_test,
                'y_test':y_test,
                'z_test':z_test,
                'params_test':params_test,
                'nuT_test':nuT_test,
                'p_test':p_test,
                'u_test':x_test,
                'v_test':y_test,
                'w_test':z_test,
                #'nuT_test':nuT_test,
                #'tau_test':tau_test,
                'x_train':x_train,
                'y_train':y_train,
                'z_train':z_train,
                'params_train':params_train,
                'nuT_train':nuT_train,
                'p_train':p_train,
                'u_train':x_train,
                'v_train':y_train,
                'w_train':z_train,
                #'nuT_train':nuT_train,
                #'tau_train':tau_train, 
               }
        if cfg.run_mode == "eval":
            np.savez("/tmp/data/data.npz", **data)
    return data
    
    
@modulus.sym.main(config_path="conf", config_name="conf_multi.yaml")
def run(cfg: ModulusConfig):
    global config, domain, nodes
    cfg.optimizer.lr = 0.002
    
    # [init-model]
    trunk_net = instantiate_arch(
        input_keys=[Key("x"),Key("y"),Key("z")],
        output_keys=[Key("trunk", 128)],
        activation_fn=Activation.STAN,
        cfg=cfg.arch.trunk,
    )
    branch_net_uvw = instantiate_arch(
        input_keys=[Key("params",4)],
        output_keys=[Key("branch", 128)],
        activation_fn=Activation.STAN,
        cfg=cfg.arch.branch_uvw,
    )
    branch_net_p = instantiate_arch(
        input_keys=[Key("params",4)],
        output_keys=[Key("branch", 128)],
        activation_fn=Activation.STAN,
        cfg=cfg.arch.branch_p,
    )
    branch_net_nut = instantiate_arch(
        input_keys=[Key("params",4)],
        output_keys=[Key("branch", 128)],
        activation_fn=Activation.STAN,
        cfg=cfg.arch.branch_nuT,
    )

    deeponet_p = DeepONetArch(
        output_keys=[Key("p")],
        branch_net=branch_net_p,
        trunk_net=trunk_net,
    )
    deeponet_uvw = DeepONetArch(
        output_keys=[Key("u"), Key("v"), Key("w")],
        branch_net=branch_net_uvw,
        trunk_net=trunk_net,
    )
    deeponet_nuT = DeepONetArch(
        output_keys=[Key("nut")],
        branch_net=branch_net_nut,
        trunk_net=trunk_net,
    )
    p_nodes = [deeponet_p.make_node("deepo_p")]
    vel_nodes = [deeponet_uvw.make_node("deepo_uvw")]
    nut_nodes = [deeponet_nuT.make_node("deepo_nut")]

    # [equations]
    nu = 1.5e-5
    
    # coordinates
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    input_vars = {"x":x, "y":y, "z":z}
    
    nut = Function("nut")(*input_vars)
    
    nu = Number(nu)
    turb_nodes = Node.from_sympy(eq=nut+nu, out_name="nu")
    
    ns = NavierStokes(nu="nu", rho=1.0, dim=3, time=False)
    navier_stokes_nodes = ns.make_nodes()
    
    nodes = p_nodes + vel_nodes + nut_nodes + [turb_nodes] + navier_stokes_nodes
    
    # [load-data]
    len_scale = cfg.custom.len_scale
    den_scale = 1.0
    vel_scale =  cfg.custom.v_scale
    angle_scale =  cfg.custom.ang_scale # degrees
    
    set_scale(len_scale, den_scale, vel_scale, angle_scale)
    
    # Load from dataset csv list
    csv_name = 'morph-wing_dataset_big.csv'
    res_path = os.path.expandvars('${GROUP_HOME}/${USER}/vtk_res')
    data = load_data(cfg, res_path, csv_name, 225,25)
 
 
    # [constraint]
    # Make Domain
    domain = Domain()

    datacon_uvw = DeepONetConstraint.from_numpy(
        nodes = vel_nodes,
        invar = {
            "x":data["x_train"],
            "y":data["y_train"],
            "z":data["z_train"],
            "params":data["params_train"],
            },
        outvar={
            "u":data["u_train"],
            "v":data["v_train"],
            "w":data["w_train"],
            },
        batch_size=cfg.batch_size.train,
        )
    domain.add_constraint(datacon_uvw, "data_uvw")

    datacon_p = DeepONetConstraint.from_numpy(
        nodes = p_nodes,
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
    
    datacon_nu = DeepONetConstraint.from_numpy(
        nodes = nodes,
        invar = {
            "x":data["x_train"],
            "y":data["y_train"],
            "z":data["z_train"],
            "params":data["params_train"],
            },
        outvar={
            "nut":data["nuT_train"],
            },
        batch_size=cfg.batch_size.nut_train,
        )
    domain.add_constraint(datacon_nu, "data_nut")
    
    flowcon = DeepONetConstraint.from_numpy(
        nodes = nodes,
        invar = {
            "x":data["x_train"],
            "y":data["y_train"],
            "z":data["z_train"],
            "params":data["params_train"],
            },
        outvar = {
            "continuity":np.zeros_like(data["p_train"]),
            "momentum_x":np.zeros_like(data["p_train"]),
            "momentum_y":np.zeros_like(data["p_train"]),
            "momentum_z":np.zeros_like(data["p_train"]),
            },
        batch_size=cfg.batch_size.phys_train,
        )
    domain.add_constraint(flowcon, "flow")

    # [constraint]
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
                'u': data['u_test'][k*n:(k+1)*n],
                'v': data['v_test'][k*n:(k+1)*n],
                'w': data['w_test'][k*n:(k+1)*n],
                'p': data['p_test'][k*n:(k+1)*n],
                'nut': data['nuT_test'][k*n:(k+1)*n],
                }
            dataset = DictGridDataset(invar_valid, outvar_valid)
         
            validator = GridValidator(nodes=nodes,dataset=dataset, plotter=None)
            domain.add_validator(validator, "validator_{}".format(k))
        
    cfg.initialization_network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_multi")
    cfg.network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_multi")   
    
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
    
    
    if cfg.run_mode == "eval":
        cfg.initialization_network_dir = os.path.expandvars("${HOME}/fly-by-feel/modulus_results/morph-wing_multi")
        cfg.network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_multi_results")
        sp = {}
        V = cfg.custom.Vinf
        alpha = cfg.custom.alpha
        d1 = cfg.custom.d1
        d2 = cfg.custom.d2
        print(d1, d2, alpha)
        # get sample points
        sp = get_eval_sample_points(d1,d2,alpha)
        params = np.array([V/vel_scale, alpha/angle_scale,
            d1/angle_scale, d2/angle_scale])
        vol_sp = sp["vol"]
        vol_sp["params"] = np.full((vol_sp["x"].shape[0],4),params)
        
        # Make inferencer
        intern_inf = PointwiseInferencer(
            nodes=nodes,
            invar=vol_sp,
            output_names=["p","u","v","w","nu"],
            batch_size=int(n_points/8),
            )
        domain.add_inferencer(intern_inf, "int_{:.2f}_{:.2f}".format(d1,d2))
        
        wing_sp = sp["surf"]
        wing_sp["params"] = np.full((wing_sp["x"].shape[0],4),params)
        # Make monitor
        wing_inf = PointwiseInferencer(
            nodes=nodes,
            invar=wing_sp,
            output_names=["p","u","v","w","nu"],
            batch_size=int(n_points/8),
            )
        domain.add_inferencer(wing_inf, "surf_{:.2f}_{:.2f}".format(d1,d2))
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
    
    
    if cfg.run_mode == "eval":
        cfg.initialization_network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_multi")
        cfg.network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_surf_multi_results")
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
                params = np.array([V/vel_scale, alpha/angle_scale,
                    d1/angle_scale, d2/angle_scale])
                sp["params"] = np.full((sp["x"].shape[0],4),params)
                # Make inferencer
                intern_inf = PointwiseInferencer(
                    nodes=nodes,
                    invar=sp,
                    output_names=["p","u","v","w","nu"],
                    batch_size=int(n_points/8),
                    )
                domain.add_inferencer(intern_inf, "int_{:.2f}_{:.2f}".format(d1,d2))
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