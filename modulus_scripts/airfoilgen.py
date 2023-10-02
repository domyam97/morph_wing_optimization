import numpy as np
import cadquery as cq
#import matplotlib.pyplot as plt
#from geomdl import fitting
#from geomdl.visualization import VisMPL as vis

c = 0.305
degree = 3

def load_from_file(filename="n0012.dat"):
    """
    Reads the base airfoil shape from a .dat file
    Inputs:
    - filename - type: string - relative path filename 
        of .dat file - default: "n0012.dat"

    Outputs
    - dat - type: numpy.ndarray - x-y coordinates of airfoil 
    """
    global c, degree
    # Read NACA 0012 coordinates from .dat file
    dat = np.genfromtxt(filename, skip_header = 3)

    # Reorder coordinates to form continuous path
    idx = int(dat.shape[0]/2)

    top, bottom = np.split(dat,[idx])
    bottom = np.flip(bottom, 0)
    dat = np.concatenate((top, bottom))

    dat = dat*c

    return dat

def make_deflected_airfoil(delta):
    """
    Calculate the deflected airfoil shape
    Inputs:
    - delta - type: float - deflection angle

    Outputs
    - coords - type: numpy.ndarray - x-y coordinates of airfoil 
    """
    global c
    (x,dx) = np.linspace(0, 1, 200, endpoint=True, retstep=True)
    yc = np.zeros(x.shape)
    yt = np.zeros(x.shape)

    for i in range(x.shape[0]):
        if x[i] < 2/3:
            yc[i] = 0
        else:
            yc[i] = (x[i]-2/3)*np.sin(np.radians(delta))
        yt[i] = (0.12/0.2)*(0.2969*np.sqrt(x[i]) - 0.126*x[i] - \
                0.3516*np.power(x[i],2) + 0.2843*np.power(x[i],3) \
                - 0.1015*np.power(x[i],4))

    dyc_x = np.diff(yc)/dx
    theta = np.arctan(dyc_x)
    theta = np.insert(theta, 0, 0)
    yt[-1] = 0.0

    coords = np.zeros((2*x.shape[0], 2))

    for i in range(x.shape[0]):
        coords[i,0] = x[i] - yt[i]*np.sin(theta[i])
        coords[i,1] = yc[i] + yt[i]*np.cos(theta[i])
        coords[-(i+1),0] = x[i] + yt[i]*np.sin(theta[i])
        coords[-(i+1),1] = yc[i] - yt[i]*np.cos(theta[i])
    '''
    xte = 2/3 + (1/3*np.cos(np.radians(delta)))
    yte = 2/3*np.sin(np.radians(delta))
    coords[x.shape[0]-1,1] = 0
    coords[x.shape[0], 1] = 0
    '''
    coords = coords*c
    return coords

def make_NURBS(coords, ctrlpts=21, rend=False):
    global degree
    
    points = []
    for i in range(coords.shape[0]):
        p = (coords[i,0],coords[i,1])
        points.append(p)

    curve = fitting.approximate_curve(points, degree, ctrlpts_size = ctrlpts)

    curve.vis = vis.VisCurve2D()
    if rend:
        print("rendering")
        curve.render()
    return curve

def make_STL(deflections, filename="def-morph-wing.stl"):
    global c
    
    dat = make_deflected_airfoil(0)
    points_base = []

    for i in range(dat.shape[0]):
        p = (dat[i,0],dat[i,1])
        points_base.append(p)
    
    idx = int(len(points_base)/2)
    points_base.pop(idx)
    
    # Units are m
    thick = .002
    base = 0.076
    result = (cq.Workplane("XZ")
        .workplane(offset=thick).polyline(points_base).close()
        .extrude(base,combine=True)
        .faces("<Y").workplane()
        .polyline(points_base).close()
    )

    for d, defl in enumerate(deflections):
        coords = make_deflected_airfoil(defl)
        points_def = []
        
        for i in range(coords.shape[0]):
            p = (coords[i,0],coords[i,1])
            points_def.append(p)

        idx = int(len(points_def)/2)
        points_def.pop(idx) # remove duplicate point.
        #print("Extruding deflected " + str(d))
        
        result = (result.workplane(offset=base)	
            .polyline(points_def).close()
            .loft(combine=True)
            .polyline(points_def).close()
            .extrude(base,combine=True)
        )

        if len(deflections) - d > 1:
            result = (result.faces("<Y").workplane()
                .polyline(points_def).close()
            )
    
    cq.exporters.export(result, filename, opt={'ascii':True})

if  __name__ == "__main__":
    defl = [-2.28,  0.765]
    filename = "../def-morph/def_nn_morphWing.stl"
    make_STL([defl[0], defl[1]], filename=filename)

    #disk = cq.Workplane("XZ").moveTo(c/2,0).circle(c*2).extrude(0.002)
    #cq.exporters.export(disk, "disk.stl", opt={'ascii':True})

