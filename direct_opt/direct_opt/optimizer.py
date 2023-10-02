#
# File: optimizer.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project2_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np

def check_feas(c, x):
    # Checks the feasibility for a given point.
    # Returns a boolean
    const = c(x)
    return np.less_equal(const,np.zeros(const.shape)).all()


def random_restart_sim(x, vari = 3):
    # Performs a true random restart with a uniform distribution.
    rng = np.random.default_rng(seed=x.shape[0])
    xr = np.zeros(x.shape)
    sams = np.linspace(-vari, vari, 50)
    for i in range(x.shape[0]):
        xr[i] = sams[rng.integers(50)] + x[i]
    return xr

def random_restart(f, g, c, x, count, vari = 3):
    # Performs a not-so-random restart with a limited local cross entropy search
    x_his= cross_entropy_pen(f, g, c, x, 300, count, var = vari, m0 = 50)
    return x_his[-1,:]

def cross_entropy_pen(f, g, c, x0, n, count, var = 3.0, m0 = 200, gam = 0.7):
    """
    Uses a Cross Entropy search to find optimum.
    Referenced Algorithm 8.7 in text
    Includes count and quadratic penalties.
    """
    m = m0 # Number of random sample points to start
    m_elite = 10 # Number of elite samples to keep
    dim = x0.shape[0]

    gamma_s = gam
    mu = x0
    x_his = [mu]

    sigma = var*np.eye(dim)
    
    dt = "O,O"
    dist_his = np.array([(mu, sigma)], dtype=dt)
    
    rng = np.random.default_rng(seed=x0.shape[0])
    
    x_best = random_restart_sim(mu, vari = var)

    while count() < n-2*m:
        samples = rng.multivariate_normal(mu,sigma,m)
        m = max(int(gamma_s*m), 2*m_elite) # Decays sample size for more efficient evals

        eval = np.zeros((0,1))
        for i in range(samples.shape[0]):
            g = c(samples[i])
            y = f(samples[i])+quad_pen(g)+count_pen(g,1000)
            eval = np.append(eval,[[y]],axis=0)

        sort_idx = np.argsort(eval, axis=0)
        samples = np.take_along_axis(samples, sort_idx, axis=0)

        if samples.shape[0] > m_elite:
            samples = samples[:m_elite]
            eval = np.sort(eval,axis=0)[:m_elite]
        else:
            eval = np.sort(eval, axis=0)
        x_best = samples[-1]

        if eval[-1]-eval[0] < 1e-8:
            print(eval[0])
            break
        mu = np.mean(samples, axis=0)
        # Some randomness is injected into the next covariance
        sigma = (m/20*rng.random()+1)*np.cov(samples.T)

        x_his = np.append(x_his, [x_best], axis=0)
        dist = np.array((mu, sigma), dtype=dt)
        dist_his = np.append(dist_his, [dist], axis =0)
       
    return x_his

def rand_pos_spanning_set(alpha, n):
    # Creates a random Positive Spanning Set for a given alpha and n-dim
    # Referenced Algorithm 8.2 in text
    rng = np.random.default_rng(seed=n)
    delta = np.rint(1/np.sqrt(alpha))
    L = np.eye(n)
    for i in range(n):
        L[i,i] = rng.integers(-1, 1, endpoint=True)*delta
        for j in range(i):
                L[i,j] = rng.integers(-delta+1,delta-1,endpoint=True)
        
    D = L[rng.permutation(n),:]
    D = D[:,rng.permutation(n)]
    D = np.concatenate((D, np.array([-np.sum(D,axis=1)]).T),axis=1)
    return D            

def mesh_adaptive_pen(f, g, c, x0, n, count, restart_var = 2):
    """
    Uses a Mesh Adaptive Direct Search (MADS) to find optimum.
    Referenced Algorithm 8.3 in text
    Includes count and quadratic penalties.
    """
    x_his = [x0]
    x = x0
    #if not check_feas(c,x):
        #x = random_restart(f, g, c, x0, count, vari = restart_var)
    gx = c(x)
    y = f(x) + quad_pen(gx) + count_pen(gx,100)
    #x_his = np.append(x_his, [x], axis=0)
    alpha = 1.0
    dim = x0.shape[0]
    ep = np.power(4.0,-15)

    while count() < n-(2*dim+4):
        if alpha > ep:
            improved = False
            D = rand_pos_spanning_set(alpha, dim)
            for i in range(D.shape[1]):
                x_n = x + alpha*D[:,i]
                gx = c(x_n)
                y_n = f(x_n) + quad_pen(gx) + count_pen(gx,100)      
                if y_n < y:
                    x_his = np.append(x_his, [x_n], axis=0)
                    y = y_n; x = x_n; improved = True
                    x_n = x + 3*alpha*D[:,i]
                    gx = c(x_n)
                    y_n = f(x_n) + quad_pen(gx) + count_pen(gx,100)
                    if y_n < y:
                        x_his = np.append(x_his, [x_n], axis=0)
                        y = y_n; x = x_n
                    break
            if improved:
                alpha = min(1.0, alpha*4)
            else:
                alpha = alpha/4
        else:
            break
    print(count())
    return x_his    

def hooke_jeeves(f, g, c, x0, n, count, alpha=1.0, gam = 0.5, epsilon = 1.0e-10):
    """
    Uses a Hooke-Jeeves search to find optimum.
    Referenced Algorithm 7.5 in text
    Includes count and quadratic penalties.
    """
    x = x0
    x_his = [x0]
    dim = x0.shape[0]
    ep = epsilon
    al = alpha
    D = np.eye(dim)

    g = c(x)
    y = f(x) + quad_pen(g) + count_pen(g,1000)
    x_best = x
    y_best = y

    while count() < n-(4*dim):
        if al > ep:
            improved = False
            for i in range(dim):
                for sgn in [-1, 1]:
                    x_n = x + al*sgn*D[:,i]
                    g = c(x_n)
                    y_n = f(x_n) + quad_pen(g) + count_pen(g,1000)
                    if y_n < y_best:
                        improved = True
                        y_best = y_n
                        x_best = x_n
            
            x_his = np.append(x_his, [x_best], axis=0)
            x = x_best; y = y_best
            if not improved:
                al = al*gam
        else:
            break

    return x_his

class Particle:
    # A container for particle information.
    def __init__(self, x):
        self.x = x
        self.v = np.zeros(x.shape)
        self.x_best = x
        self.y = 0
        self.y_best = 1000

def particle_swarm_pen(f, g, c, x0, n, count, var = 3, restart_var = 3):
    """
    Uses a Particle Swarm search to find optimum.
    Referenced Algorithm 9.12 in text
    Includes count and quadratic penalties.
    """
    dim = x0.shape[0]; 
    wid = max(int(20/dim),3)
    rng = np.random.default_rng(seed=dim)
    

    m = min(pow(wid,min(dim,4)),200)
    sams = np.linspace(-var, var, m)
    x=x0
    x_his = [x]

    # If initial position is not feasible, try to randomly
    # find a feasible point nearby
    while not check_feas(c,x):
        if count() < 10:
            x = random_restart_sim(x, vari = restart_var)
        else:
            x_his = np.append(x_his, [x], axis = 0)
            break

    w = 0.7; c1 = 0.8; c2 = 1
    popl = []

    # Use uniform projection to initialize population
    # Helps with higher dimensional problems
    D = np.zeros((dim,m))
    for i in range(dim):
        D[i,:] = rng.permutation(sams)
    for j in range(m):
        xn = D[:,j] + x            
        popl.append(Particle(xn))
    
    y_best = 10.0e10

    for P in popl:
        g = c(P.x)
        p_quad = quad_pen(g)
        p_cnt = count_pen(g,100)
        y = f(P.x) + p_quad + p_cnt
        P.y = y
        if y < y_best:
            y_best = y

    pop_his = [popl]
    best = sorted(pop_his[-1], key=get_y)[0].x_best
    x_his = np.append(x_his, [best], axis = 0)

    while count() < n-2*m:
        best = sorted(pop_his[-1],key=get_y)[0].x_best
        for P in popl:
            r1 = rng.random(); r2 = rng.random()
            P.x = P.x + P.v
            P.v = w*P.v + c1*r1*(P.x_best - P.x) + c2*r2*(best - P.x)
        for P in popl:
            g = c(P.x)
            p_quad = quad_pen(g)
            p_cnt = count_pen(g,100)
            
            y = f(P.x) + p_cnt + p_quad
            P.y = y               
            if y < P.y_best:
                    P.x_best = P.x
                    P.y_best = y
        x_his = np.append(x_his, [best], axis = 0)
        pop_his.append(popl)


    best = sorted(pop_his[-1],key=get_y).x_best
    x_his = np.append(x_his, [best], axis = 0)
    return x_his

def get_y(part):
    # Key-function for sorting the partivle swarm population
    return part.y_best

def count_pen(constr, rho):
    '''
    Takes in the output of an eval of p.c(x) and a value for rho.
    Returns the count penalty.
    '''
    count = 0
    for i in range(constr.shape[0]):
        if constr[i] > 0:
            count +=1
    return rho*count

def quad_pen(constr):
    '''
    Takes in the output of an eval of p.c(x) and a value for rho.
    Returns the weighted quad penalty.
    '''
    p_quad = 0
    for i in range(constr.shape[0]):
        p_quad += 1000*pow(max(constr[i],0),2)
    return p_quad

def nesterov_mom(f, g, c, x0, n, count, alpha=0.0003, restart_var = 1):
    """
    Uses Nesterov Momentum descent to find optimum.
    Referenced Algorithm 5.4 in text

    Adds a restart using cross entropy if initial position is not a valid point. 
    """
    # Optmizer Params
    # Learning Rate from function call
    beta = 0.7 # Momentum Decay
    rng = np.random.default_rng(seed=x0.shape[0])

    # Initialize Search
    x = x0
    while check_feas(c, x)==False:
        if count() < 10:
            x = random_restart(f, g, c ,-x, count, vari = restart_var)
        else:
            x = -np.absolute(x)
            break
    x_his = [x]
    v = np.zeros(x0.shape)
    it = 0
    while count() < n-2:
        gMom = g(x + beta*v)
 
        # Calculate momentum at next step
        v = beta*v - alpha*gMom
        # Calculate next x
        x = x + v
        x_his = np.append(x_his, [x], axis=0)
    
    return x_his 

def optimize(f, g, c, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        c (function): Function evaluating constraints
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `f` and `c` cost 1 and `g` costs 2
        count (function): takes no arguments are reutrns current count
        prob (str): Name of the problem. So you can use a different strategy 
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """
    if prob == "simple1":
        # Do MADS w/ penalties
        x_his = mesh_adaptive_pen(f, g, c, x0, n, count)
    elif prob == "simple2":
        # Start with MADS w/ penalties to get within feasible region
        x_his = mesh_adaptive_pen(f, g, c, x0, n-1000, count)
        x = x_his[-1,:]
        # Finish with a particle search around feasible point.
        x_his = particle_swarm_pen(f, g, c, x, count()+800, count, var = 1, restart_var = 0.1)
        '''
        for i in range(len(pop_his)):
            xb = sorted(pop_his[-1], key = get_y)[0].x_best
            x_his = np.append(x_his, [xb], axis=0)
        '''
    elif prob == "simple3":
        # Do MADS w/ penalties
        x_his = mesh_adaptive_pen(f, g, c, x0, n, count, restart_var = 4)
    elif prob == "secret2":
        # Start with Nesterov Mom. Doesn't have penalty func.
        x_his = nesterov_mom(f, g, c, x0, n-1000, count, restart_var = 3)
        x = x_his[-1,:]
        # Find valid point with MADS w/ penalties
        x_his = mesh_adaptive_pen(f, g, c, x, n, count, restart_var = 1)
    elif prob == "secret1":
        # Start with a big Hooke-Jeeves search with penalties to find valid region.
        x_his = hooke_jeeves(f, g, c, x0, n, count, alpha = 200, gam = 0.8, epsilon=1)
        xh = x_his[-1,:]; xr = xh

        # Do a local Cross Entropy search with penalties to make sure you're in valid region.
        while count() < n-300:
            if not check_feas(c, xr):
                xr = random_restart(f, g, c, xh, count, vari = 1) # Does Local Cross Entropy Search
            else:
                if not xr==xh:
                    x_his = np.append(x_his, [xr], axis=0)
                break      

    x_best = x_his[-1,:]

    return x_best
