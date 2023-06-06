import os
import warnings
import sys

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

from modulus.sym.eq.pdes.navier_stokes import NavierStokes

from modulus.sym.domain.constraint.continuous import DeepONetConstraint
from modulus.sym.domain.validator.discrete import GridValidator
from modulus.sym.dataset.discrete import DictGridDataset

from modulus.sym.key import Key

rng = np.random.default_rng(0)

def load_data(res_path, csv_name, n_train=50, n_test=10):
    print(os.getcwd())
    dtype = np.dtype({'names':['V','alpha','d1','d2','folder'], 'formats':[np.float32, np.float32, np.float32, np.float32, 'U25']})
    csv_data = np.loadtxt(res_path+'/'+csv_name, dtype=dtype, skiprows=1, delimiter=',')
    csv_data = rng.permutation(csv_data,axis=0)
     
    len_scale = 0.7
    den_scale = 1.2
    vel_scale = 20.0
    time_scale = len_scale/vel_scale
    mass_scale = den_scale*(len_scale**3)
    angle_scale = 15.0 # degrees 

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
    u_train = np.ndarray((0,1))
    v_train = np.ndarray((0,1))
    w_train = np.ndarray((0,1))
    p_train = np.ndarray((0,1))
    nuT_train = np.ndarray((0,1))
    #tau_train = np.ndarray((0,3))

    for i in range(n_train):
        vtk_obj = VTKFromFile(res_path+'/'+csv_data[i]["folder"]+'/internal.vtu')
        # Get Points
        points = vtk_obj.get_points()
        x_train = np.concatenate((x_train, points[:,0].reshape((points.shape[0],1))/len_scale), axis = 0)
        y_train = np.concatenate((y_train, points[:,1].reshape((points.shape[0],1))/len_scale), axis = 0)
        z_train = np.concatenate((z_train, points[:,2].reshape((points.shape[0],1))/len_scale), axis = 0)
        # Set Input Vals
        params = np.array([csv_data[i]["V"]/vel_scale, csv_data[i]["alpha"]/angle_scale,
             csv_data[i]["d1"]/angle_scale, csv_data[i]["d2"]/angle_scale])
        params_train = np.concatenate((params_train, np.full((points.shape[0],4),params)), axis = 0) 
        #V_train  = np.concatenate((V_train,np.ones((points.shape[0],1))*csv_data[i]["V"]),axis = 0)
        #alpha_train  = np.concatenate((alpha_train,np.ones((points.shape[0],1))*csv_data[i]["alpha"]),axis = 0)
        #d1_train  = np.concatenate((d1_train,np.ones((points.shape[0],1))*csv_data[i]["d1"]),axis = 0)
        #d2_train  = np.concatenate((d2_train,np.ones((points.shape[0],1))*csv_data[i]["d2"]),axis = 0)
 
        # Get Outputs
        u_train = np.concatenate((u_train,vtk_obj.get_array("U")[:,0].reshape((points.shape[0],1))/vel_scale), axis=0)
        v_train = np.concatenate((v_train,vtk_obj.get_array("U")[:,1].reshape((points.shape[0],1))/vel_scale), axis=0)
        w_train = np.concatenate((w_train,vtk_obj.get_array("U")[:,2].reshape((points.shape[0],1))/vel_scale), axis=0)
        p_train = np.concatenate((p_train,(vtk_obj.get_array("p")-101e3)*(time_scale**2)*len_scale/mass_scale), axis=0)    
        nuT_train = np.concatenate((nuT_train,vtk_obj.get_array("nuTilda")/mass_scale*len_scale*time_scale),axis = 0)
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
    u_test = np.ndarray((0,1))
    v_test = np.ndarray((0,1))
    w_test = np.ndarray((0,1))
    p_test = np.ndarray((0,1))
    nuT_test = np.ndarray((0,1))
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
        #V_test  = np.concatenate((V_test,np.ones((points.shape[0],1))*csv_data[i]["V"]),axis = 0)
        #alpha_test  = np.concatenate((alpha_test,np.ones((points.shape[0],1))*csv_data[i]["alpha"]),axis = 0)
        #d1_test  = np.concatenate((d1_test,np.ones((points.shape[0],1))*csv_data[i]["d1"]),axis = 0)
        #d2_test  = np.concatenate((d2_test,np.ones((points.shape[0],1))*csv_data[i]["d2"]),axis = 0)
 
        # Get Outputs
        u_test = np.concatenate((u_test,vtk_obj.get_array("U")[:,0].reshape((points.shape[0],1))/vel_scale), axis=0)
        v_test = np.concatenate((v_test,vtk_obj.get_array("U")[:,1].reshape((points.shape[0],1))/vel_scale), axis=0)
        w_test = np.concatenate((w_test,vtk_obj.get_array("U")[:,2].reshape((points.shape[0],1))/vel_scale), axis=0)
        p_test = np.concatenate((p_test,(vtk_obj.get_array("p")-101e3)*(time_scale**2)*len_scale/mass_scale), axis=0)    
        nuT_test = np.concatenate((nuT_test,vtk_obj.get_array("nuTilda")*len_scale*time_scale/mass_scale), axis=0)
        #tau_test = np.concatenate((tau_test,vtk_obj.get_array("wallShearStress"), axis=0)
    print(x_train.shape, x_test.shape)
    scale = {"len":len_scale, "mass":mass_scale, "time":time_scale, "vel":vel_scale}
    data = {'x_test':x_test,
            'y_test':y_test,
            'z_test':z_test,
            'params_test':params_test, 
            #'V_test':V_test, 
            #'alpha_test':alpha_test, 
            #'d1_test':d1_test, 
            #'d2_test':d2_test, 
            'u_test':u_test,
            'v_test':v_test,
            'w_test':w_test,
            'p_test':p_test,
            'nuT_test':nuT_test,
            #'tau_test':tau_test,
            'x_train':x_train,
            'y_train':y_train,
            'z_train':z_train,
            'params_train':params_train, 
            #'V_train':V_train, 
            #'alpha_train':alpha_train, 
            #'d1_train':d1_train, 
            #'d2_train':d2_train, 
            'u_train':u_train,
            'v_train':v_train,
            'w_train':w_train,
            'p_train':p_train,
            'nuT_train':nuT_train,
            #'tau_train':tau_train, 
           }
    return data, scale

@modulus.sym.main(config_path="conf", config_name="conf_multi.yaml")
def run(cfg: ModulusConfig) -> None:
    
    cfg.optimizer.lr = 0.001
    
    # [init-model]
    trunk_net = FullyConnectedArch(
        input_keys=[Key("x"),Key("y"),Key("z")],
        output_keys=[Key("trunk", 128)],
    )
    branch_net_uvw = FullyConnectedArch(
        input_keys=[Key("params",4)],
        output_keys=[Key("branch", 128)],
    )
    branch_net_p = FullyConnectedArch(
        input_keys=[Key("params",4)],
        output_keys=[Key("branch", 128)],
    )
    branch_net_nu = FullyConnectedArch(
        input_keys=[Key("params",4)],
        output_keys=[Key("branch", 128)],
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
        output_keys=[Key("nu")],
        branch_net=branch_net_nu,
        trunk_net=trunk_net,
    )
    deepo_nodes = [
        deeponet_p.make_node("deepo_p"), 
        deeponet_uvw.make_node("deepo_uvw"),
        deeponet_nuT.make_node("deepo_nu")
    ]

    # [equations]
    ns = NavierStokes(nu="nu", rho=1.2, dim=3, time=False)
    navier_stokes_nodes = ns.make_nodes()
    
    nodes = deepo_nodes + navier_stokes_nodes
    # [load-data]
    # Load from dataset csv list
    csv_name = 'morph-wing_dataset_big.csv'
    res_path = os.path.expandvars('${GROUP_HOME}/${USER}/vtk_res')
    data, scale = load_data(res_path, csv_name, 225,25)
 
 
    # [constraint]
    # Make Domain
    domain = Domain()

    datacon_uvw = DeepONetConstraint.from_numpy(
        nodes = nodes,
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
    
    datacon_nu = DeepONetConstraint.from_numpy(
        nodes = nodes,
        invar = {
            "x":data["x_train"],
            "y":data["y_train"],
            "z":data["z_train"],
            "params":data["params_train"],
            },
        outvar={
            "nu":data["nuT_train"],
            },
        batch_size=cfg.batch_size.train,
        )
    domain.add_constraint(datacon_nu, "data_nu")
    
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
        batch_size=cfg.batch_size.lr_phys,
        )
    domain.add_constraint(flowcon, "flow")

    # [constraint]

    # [validator]
    for k in range(10):
        n = int(data['x_test'].shape[0]/10)
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
            'nu': data['nuT_test'][k*n:(k+1)*n],
            }
        dataset = DictGridDataset(invar_valid, outvar_valid)
     
        validator = GridValidator(nodes=nodes,dataset=dataset, plotter=None)
        domain.add_validator(validator, "validator_{}".format(k))
        
    cfg.initialization_network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_multi")
    cfg.network_dir = os.path.expandvars("${SCRATCH}/modulus/outputs/morph-wing_multi")    
    # make solver
    slv = Solver(cfg,domain)
    
    slv.solve()

if __name__ == "__main__":
    #for key, val in os.environ.items():
    #    print(key, val)
    print(os.environ.get("SLURM_LAUNCH_NODE_IPADDR"))
    print(os.environ.get("HOSTNAME"))
    torch.cuda.empty_cache()
    run()
