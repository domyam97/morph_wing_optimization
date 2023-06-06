import numpy as np
import subprocess
import os
import sys 

import time

from optimizer import *


sys.path.append("../..")
from morph_wing.modulus.morph_wing_surface import *

from modulus.sym.hydra import ModulusConfig
from modulus.sym.domain import Domain
from modulus.sym.utils.io.vtk import *
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

rng = np.random.default_rng(0)

L0 = 0
counter = 0

last_x_d = None
last_x_l = None

def load_vtk_data(res_path, csv_name):
    global L0

    dtype = np.dtype({'names':['V','alpha','d1','d2','folder'], 'formats':[np.float32, np.float32, np.float32, np.float32, 'U25']})
    csv_data = np.loadtxt(res_path+'/'+csv_name, dtype=dtype, skiprows=1, delimiter=',')
    print("loading data")
    # Design points [d1, d2, Lift].T
    x = np.zeros((csv_data.shape[0],3,1)) 
    # Objective Evaluations, Drag
    y = np.zeros((csv_data.shape[0],1))

    for i in range(csv_data.shape[0]):
        vtk_wrap = VTKFromFile(res_path+'/'+csv_data[i]["folder"]+'/wing.vtp')
        points = vtk_wrap.get_points()

        polyDatNormals = vtk.vtkPolyDataNormals()
        polyDatNormals.SetComputeCellNormals(True)
        polyDatNormals.SetInputData(vtk_wrap.vtk_obj)
        polyDatNormals.Update()
    
        normals = vtk_to_numpy(polyDatNormals.GetOutput().GetCellData().GetArray('Normals'))
        cell_press = vtk_to_numpy(vtk_wrap.vtk_obj.GetCellData().GetArray('p')).reshape((normals.shape[0],1))-101e3
        cell_tau = vtk_to_numpy(vtk_wrap.vtk_obj.GetCellData().GetArray('wallShearStress'))

        cell_lift =  cell_press*(normals[:,2].reshape((normals.shape[0],1))) - cell_tau[:,2].reshape((cell_tau.shape[0],1))
        cell_drag =  cell_press*(normals[:,0].reshape((normals.shape[0],1))) - cell_tau[:,0].reshape((cell_tau.shape[0],1))


        lift_array = numpy_to_vtk(cell_lift, deep = True)
        lift_array.SetNumberOfComponents(cell_lift.shape[1])
        lift_array.SetName('Lift')
        vtk_wrap.vtk_obj.GetCellData().AddArray(lift_array)

        drag_array = numpy_to_vtk(cell_drag, deep = True)
        drag_array.SetNumberOfComponents(cell_drag.shape[1])
        drag_array.SetName('Drag')
        vtk_wrap.vtk_obj.GetCellData().AddArray(drag_array)
         

        integrator = vtk.vtkIntegrateAttributes()
        integrator.SetInputData(vtk_wrap.vtk_obj)
        integrator.Update()


        lift  = vtk_to_numpy(integrator.GetOutputDataObject(0).GetCellData().GetArray('Lift'))
        drag  = vtk_to_numpy(integrator.GetOutputDataObject(0).GetCellData().GetArray('Drag'))

        x[i] = np.array([[csv_data[i]['d1'], csv_data[i]['d2'], lift[0]]]).T
        y[i,0] = np.array(drag[0]) 

        if csv_data[i]['d1'] == 0 and csv_data[i]['d2'] == 0:
            L0 = lift[0]
    return x,y


class PolyBasis:
    def __init__(self, dims, orders):
        self.dims = dims
        self.orders = orders
        self.theta = 0
    def evaluate(self, x):
        out = 1;
        for i in range(self.dims):
            out = out*np.power(x[i,0],self.orders[i])
        return out
class RadialBasis:
    def __init__(self, basis, center, p):
        self.basis = basis
        self.center = center
        self.p = p
        self.theta = 0
    def evaluate(self,x):
        return self.basis(np.linalg.norm((x-self.center),self.p))

def gaussian_basis(r):
    return np.exp(-2*(r**2))

def thin_plate_basis(r):
    return (r**2)*np.log(r+1e-8)

def linear_basis(r):
    return r

def cubic_basis(r):
    return r**3

def regression(x, y, bases):
    B = np.zeros((x.shape[0],x.shape[0]))
    for i in range(x.shape[0]):
        for j,b in enumerate(bases):
            B[i,j] = b.evaluate(x[i])
    theta = np.linalg.pinv(B)@y

    for i in range(theta.shape[0]):
        bases[i].theta = theta[i,0]

def eval_surrogate_l(x, bases_l = None):
    if bases_l == None:
        bases_l = surrogate_l
    x_l = x[0:2].reshape((2,1))
    predict = 0
    for b in bases_l:
        predict += b.theta*b.evaluate(x_l)
    return predict

def eval_surrogate(x, bases = None, bases_l = None):
    global counter
    if bases == None:
        bases = surrogate_d
    if bases_l == None:
        bases_l = surrogate_l

    counter += 1

    x_l = x[0:2].reshape((2,1))
    if x.shape[0] == 2:
        x = np.append(x_l, [[eval_surrogate_l(x_l, bases_l)]], axis=0)
    else:
        x = x.reshape((3,1))
    
    predict = 0
    
    for b in bases:
        predict += b.theta*b.evaluate(x)
     
    return predict

def gen_basis_func(x,y):
    xy = np.concatenate((x,y.reshape((y.shape[0],1,1))), axis=1)
    rng.shuffle(xy)
    x = xy[:,0:3,0].reshape(x.shape)
    y = xy[:,3,0].reshape(y.shape)

    # Try radial basis + 2d polynomial
    centers = x[0:20].reshape((20,x.shape[1],1))
    centers_l = x[0:20,0:2,0].reshape((20,x.shape[1]-1,1))

    y_train = y[0:20].reshape((20,1))
    l_train = x[0:20,2,0].reshape((20,1))

    gauss_bases_d = []
    tp_bases_d = []
    lin_bases_d = []
    cub_bases_d = []
    for c in range(centers.shape[0]):
        gauss_bases_d.append(RadialBasis(gaussian_basis, centers[c], 2))
        tp_bases_d.append(RadialBasis(thin_plate_basis, centers[c], 2))
        lin_bases_d.append(RadialBasis(linear_basis, centers[c], 2))
        cub_bases_d.append(RadialBasis(cubic_basis, centers[c], 2))

    gauss_bases_l = []
    tp_bases_l = []
    cub_bases_l = []
    lin_bases_l = []
    poly_bases_l = [PolyBasis(2,[0,0])]
    for c in range(centers_l.shape[0]):
        gauss_bases_l.append(RadialBasis(gaussian_basis, centers_l[c], 2))
        tp_bases_l.append(RadialBasis(thin_plate_basis, centers_l[c], 2))
        lin_bases_l.append(RadialBasis(linear_basis, centers_l[c], 2))
        cub_bases_l.append(RadialBasis(cubic_basis, centers_l[c], 2))

    k = 4
    for i in range(k):
        for j in range(k):
            poly_bases_l.append(PolyBasis(2,[i+1,j+1]))


    regression(centers, y_train, gauss_bases_d)
    regression(centers, y_train, tp_bases_d)
    regression(centers, y_train, lin_bases_d)
    regression(centers, y_train, cub_bases_d)
    
    regression(centers_l, l_train, gauss_bases_l)
    regression(centers_l, l_train, tp_bases_l)
    regression(centers_l, l_train, lin_bases_l)
    regression(centers_l, l_train, cub_bases_l)
    num_pb = len(poly_bases_l)
    print(num_pb)
    regression(centers_l[0:num_pb].reshape((num_pb,2,1)), l_train[0:num_pb].reshape((num_pb,1)), poly_bases_l)

    # test basis functions
    y_test = y[20:].reshape((y.shape[0]-20,1))
    l_test = x[20:,2,0].reshape((x.shape[0]-20,x.shape[1]-2,1))

    x_test = x[20:].reshape((x.shape[0]-20,x.shape[1],1))
    x_test_l = x[20:,0:2,0].reshape((x.shape[0]-20,x.shape[1]-1,1))    

    basis_names = ['gauss', 'linear', 'cubic', 'thin plate', 'polynomial']
    bases_l = [gauss_bases_l, lin_bases_l, cub_bases_l, tp_bases_l, poly_bases_l]
    errors_l = [0, 0, 0, 0, 0]

    bases_d = [gauss_bases_d, lin_bases_d, cub_bases_d, tp_bases_d]
    errors_d = [0, 0, 0, 0]

    for i in range(l_test.shape[0]):
        for b,basis in enumerate(bases_l):
            predict = eval_surrogate_l(x_test_l[i],basis)
            errors_l[b] += np.linalg.norm((l_test[i]-predict)/l_test[i],2) 
    
    surrogate_l = bases_l[np.argmin(errors_l)] 

    for i in range(y_test.shape[0]):
        for b,basis in enumerate(bases_d):
            predict = eval_surrogate(x_test[i],basis, surrogate_l)
            errors_d[b] += np.linalg.norm((y_test[i]-predict)/y_test[i],2)
    
    print("Tot L2 Rel Error Gauss: ", errors_l[0], " ", errors_d[0])
    print("Tot L2 Rel Error Linear: ", errors_l[1], " ", errors_d[1])
    print("Tot L2 Rel Error Cubic: ", errors_l[2], " ", errors_d[2])
    print("Tot L2 Rel Error TP: ", errors_l[3], " ", errors_d[3])
    print("Tot L2 Rel Error Polynomial: ", errors_l[4])

    
    print("Lift: ", basis_names[np.argmin(errors_l)], " Drag: ", basis_names[np.argmin(errors_d)])

    return bases_d[np.argmin(errors_d)], bases_l[np.argmin(errors_l)]

def constraints(x):
    l_x = eval_surrogate_l(x)

    c1 = max([0,np.abs(l_x-L0)-0.1*L0])
    c2 = max([0,np.abs(x[0])-3])
    c3 = max([0,np.abs(x[1])-3])
    return np.array([c1, c2, c3])

def count():
    return counter

def reset_count():
    global counter
    counter = 0 

def optimize(f, c, x0):
    start_time = time.time()
    x_his = cross_entropy_pen(f, None, c, x0, 2000, count, var=0.2, m0=50)
    x_best_cross_en = x_his[-1,:]
    con_cross_en = constraints(x_best_cross_en)
    d_cross_en = eval_surrogate(x_best_cross_en) + quad_pen(con_cross_en) + \
        count_pen(con_cross_en,1000)
    tot_time_cross_en = time.time() - start_time
    
    print("Cross En - Lift: ", eval_surrogate_l(x_best_cross_en), " Drag: ", eval_surrogate(x_best_cross_en))
    print("Total Time Cross Entropy, Surroagte (s): ", tot_time_cross_en)
    
    reset_count()

    start_time = time.time()
    x_his = mesh_adaptive_pen(f, None, c, x0, 2000, count)
    x_best_mesh_ad = x_his[-1,:] 
    con_mesh_ad = constraints(x_best_mesh_ad)
    d_mesh_ad = eval_surrogate(x_best_mesh_ad) + quad_pen(con_mesh_ad) + \
        count_pen(con_mesh_ad,1000)
    
    print("Mesh Ad - Lift: ", eval_surrogate_l(x_best_mesh_ad), " Drag: ", eval_surrogate(x_best_mesh_ad))
    tot_time_mesh_ad = time.time() - start_time
    print("Total Time Mesh Adaptive, Surroagte (s): ", tot_time_mesh_ad)
    
    reset_count()

    start_time = time.time()
    x_his = hooke_jeeves(f, None, c, x0, 2000, count)
    x_best_hj = x_his[-1,:] 
    con_hj = constraints(x_best_hj)
    d_hj = eval_surrogate(x_best_hj) + quad_pen(con_hj) + \
        count_pen(con_hj,1000)
    print("HJ - Lift: ", eval_surrogate_l(x_best_hj), " Drag: ", eval_surrogate(x_best_hj))
    tot_time_hj = time.time() - start_time
    print("Total Time Hooke-Jeeves, Surroagte (s): ", tot_time_hj)
 
    bests = [x_best_cross_en, x_best_mesh_ad, x_best_hj]

    print(bests)
    print(d_cross_en, d_mesh_ad, d_hj) 
    return bests[np.argmin([d_cross_en, d_mesh_ad, d_hj])] 
    
def optimize_nn(f, c, x0):
    
    start_time = time.time()
    x_his = cross_entropy_pen(f, None, c, x0, 2000, count, var=0.5, m0=20)
    x_best_cross_en = x_his[-1,:]
    d_cross_en = eval_nn(x_best_cross_en)
    tot_time_cross_en = time.time() - start_time
    
    print("Total Time Cross Entropy, NN (s): ", tot_time_cross_en)
    print(count())
    reset_count()
    
    start_time = time.time()
    x_his = hooke_jeeves(f, None, c, x0, 2000, count, alpha=1.0, gam = 0.8)
    x_best_hooke_jeeves = x_his[-1,:]
    d_hooke_jeeves = eval_nn(x_best_hooke_jeeves)
    tot_time_hooke_jeeves = time.time() - start_time
    
    print("Total Time Hooke-Jeeves, NN (s): ", tot_time_hooke_jeeves)
    print(count())
    reset_count()

    start_time = time.time()
    x_his = mesh_adaptive_pen(f, None, c, x0, 2000, count)
    x_best_mesh_ad = x_his[-1,:] 
    d_mesh_ad = eval_nn(x_best_mesh_ad)

    tot_time_mesh_ad = time.time() - start_time
    print("Total Time Mesh Adaptive, NN (s): ", tot_time_mesh_ad)
    print(count())
    bests = [x_best_cross_en, x_best_hooke_jeeves, x_best_mesh_ad]

    print(bests)
    print(d_cross_en, d_hooke_jeeves, d_mesh_ad) 
    return bests[np.argmin([d_cross_en, d_hooke_jeeves, d_mesh_ad])] 
    
def optimize_nn_batch(var, gamma):
    n_iters = 3
    n_samples = 5
    width = var
    d1_range = (0-(width/2),0+(width/2))
    d2_range = (0-(width/2),0+(width/2))
    start_time = time.time()
    for i in range(n_iters):
        x_best_batch, l_batch, d_batch = eval_nn_batch(d1_range, d2_range, n_samples)
        width *= gamma
        d1_range = (x_best_batch[0]-(width/2),x_best_batch[0]+(width/2))
        d2_range = (x_best_batch[1]-(width/2),x_best_batch[1]+(width/2))
        
    
    tot_time_batch = time.time() - start_time
    
    print("Total Time Batch, NN (s): ", tot_time_batch)

    return x_best_batch, l_batch, d_batch
    

def eval_nn_l(x):
    global counter, last_x_l
    counter +=1
    cfg.custom.d1 = float(x[0])
    cfg.custom.d2 = float(x[1])
    cfg.custom.Vinf = V
    cfg.custom.alpha = alpha
    cfg.run_mode='eval'
    if not np.all(np.equal(x,last_x_d)):
        solve_nn(cfg, domain, nodes)
    lift = np.loadtxt(os.path.expandvars("$SCRATCH/modulus/outputs/morph-wing_surf_big_results/monitors/lift.csv"), 
        skiprows=1, delimiter=',',usecols=(1))
    os.remove("$SCRATCH/modulus/outputs/morph-wing_surf_big_results/monitors/lift.csv")
        
    last_x_l = x
    print(lift.shape)
    if lift.shape:
        return float(lift[-1])
    else:
        return float(lift)
    
    
def eval_nn(x):
    global counter, last_x_d
    counter +=1
    cfg.custom.d1 = float(x[0])
    cfg.custom.d2 = float(x[1])
    cfg.custom.Vinf = V
    cfg.custom.alpha = alpha
    cfg.run_mode='eval'
    if not np.all(np.equal(last_x_l,x)):
        solve_nn(cfg, domain, nodes)
    drag = np.loadtxt(os.path.expandvars("$SCRATCH/modulus/outputs/morph-wing_surf_big_results/monitors/drag.csv"),
         skiprows=1, delimiter=',',usecols=(1))
    os.remove("$SCRATCH/modulus/outputs/morph-wing_surf_big_results/monitors/drag.csv")
         
    last_x_d = x
    if drag.shape:
        return float(drag[-1])
    else:
        return float(drag)
        
def eval_nn_batch(d1_range, d2_range, n_samples):
    cfg.custom.Vinf = V
    cfg.custom.alpha = alpha
    cfg.run_mode='eval'
    solve_nn_batch(cfg, domain, nodes, d1_range, d2_range, n_samples)
    
    res_dir = os.path.expandvars("$SCRATCH/modulus/outputs/morph-wing_surf_big_results/monitors")
    i_drag = 0
    i_lift = 0
    drag = np.zeros((n_samples**2,1)); lift = np.zeros((n_samples**2,1));
    d1 = np.zeros((n_samples**2,1)); d2 = np.zeros((n_samples**2,1));
    for file in os.listdir(res_dir):
        if file.startswith("drag") & file.endswith("mon.csv"):
            drag[i_drag,0] = float(np.loadtxt(os.path.join(res_dir,file),
                skiprows=1, delimiter=',',usecols=(1,)))
            d1[i_drag,0] = float(file.split('_')[1])
            d2[i_drag,0] = float(file.split('_')[2])
            i_drag += 1
            os.remove(os.path.join(res_dir,file))
        if file.startswith("lift") & file.endswith("mon.csv"):
            lift[i_lift,0] = float(np.loadtxt(os.path.join(res_dir,file),
                skiprows=1, delimiter=',',usecols=(1,)))
            i_lift +=1
            os.remove(os.path.join(res_dir,file))
            
    x = np.concatenate((d1,d2,lift),axis=1)
    obj = np.zeros((x.shape[0],1))
    for i in range(x.shape[0]):
        c_x = constraint_nn_batch(x[i,:])
        obj[i,0] = drag[i,0] + quad_pen(c_x) + count_pen(c_x, 1000)
        print(x[i,0:2], " L: ", x[i,2], " D: ", drag[i,0], " Obj: ", obj[i,0])
    
    best_idx = np.argmin(obj)
    x_best = x[best_idx,0:2]
    print(x_best)
    print('lift: ', x[best_idx,2], " drag: ", drag[best_idx,0])
    return x_best, x[best_idx,2], drag[best_idx,0]

def constraint_nn(x):
    l_x = eval_nn_l(x)
    c1 = max([0,np.abs(l_x-L0)-0.1*L0])
    c2 = max([0,np.abs(x[0])-3])
    c3 = max([0,np.abs(x[1])-3])
    return np.array([c1, c2, c3])
    
def constraint_nn_batch(x):
    c1 = max([0,np.abs(x[2]-L0)-0.1*L0])
    c2 = max([0,np.abs(x[0])-3])
    c3 = max([0,np.abs(x[1])-3])
    c = np.array([c1,c2,c3])
    print(c)
    return c

if __name__ == "__main__":
    res_path = os.path.expandvars("${GROUP_HOME}/${USER}/vtk_res")
    csv_name = "morph-wing_dataset_single_2.csv"
    x,y = load_vtk_data(res_path, csv_name)
    surrogate_d, surrogate_l = gen_basis_func(x,y)
    print(L0)
    
    x0=np.array([-0.3,1.5])
    
    reset_count()
    x_best = optimize(eval_surrogate, constraints, x0)
    print(x_best)
    print("lift: ", eval_surrogate_l(x_best)," drag: ", eval_surrogate(x_best))
    V = 14.0
    alpha = 3.0
    reset_count()
    run()
    cfg = get_config()
    domain = get_domain()
    nodes = get_nodes()
    x_best_nn, l_batch, d_batch = optimize_nn_batch(6, 0.3)
    print("NN Result:")
    print(x_best_nn)
    print("lift: ", l_batch, " drag: ",d_batch)
    
