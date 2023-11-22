""" This module generates artificial microtubule (MT) length data.
"""
from numpy.random import rand, poisson, randint
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

########################################## FUNCTIONS ##########################################

def timestep(l,dt):
    """Performs a time step of the MT simulation. 

    Args:
        - l (numpy.ndarray of float): Array of MT lengths at time t
        - dt (float): Time step. 

    Returns:
        - out (numpy.ndarray of float): Array of MT lengths at time t+dt.
        - count (int): Number of events that happend during the time step.
    """
    l,c1 = turnover(l,dt)  
    l,c2 = severing(l,dt)
    l = growing(l,dt) 
    l,c3 = nucleation(l,dt)
    count = c1+c2+c3
    out = l
    return out, count

def turnover(l,dt):
    """Performs the turnover events in a time step.

    Args:
        - l (numpy.ndarray of float): Array of MT lengths before turnover events.
        - dt (float): Time step.  

    Returns:
        - out (numpy.ndarray of float): Array of MT lengths after turnover events.
        - count (int): Number of turnover events that happend during the time step.
    """
    count = 0
    out = []
    n = len(l)
    n_r = poisson(n*R*dt)
    which = randint(n,size=n_r)
    for i in range(n):
        if i not in which:
            out.append(l[i])
        else: count += 1 
    return out, count


def severing(l,dt):
    """Performs the severing events in a time step. 

    Args:
        - l (numpy.ndarray of float): Array of MT lengths before severing events.
        - dt (float): Time step. 

    Returns:
        - out (numpy.ndarray of float): Array of MT lengths after severing events.
        - count (int): Number of severing events that happend during the time step.
    """
    count = 0
    out = []
    for li in l:
        n_kappa = poisson(li*KAPPA*dt)
        if n_kappa == 0:
            out.append(li)
        else:
            count += 1
            cutpos = li*rand()
            out.append(cutpos)
            if rand()<ALPHA:
                out.append(li-cutpos)
    return out, count


def growing(l,dt):
    """Performs the growth of each MT during a time step. 

    Args:
        - l (numpy.ndarray of float): List of MT lengths before growing.
        - dt (float): Time step.

    Returns:
        - out (numpy.ndarray of float): List of MT lengths after growth.
    """
    out = [li + dt*VG for li in l]
    return out


def nucleation(l,dt):
    """Performs the nucleation events in a time step.

    Args:
        - l (numpy.ndarray of float): Array of MT lengths before nucelation events.
        - dt (float): Time step. 

    Returns:
        - out (numpy.ndarray of float): Array of MT lengths after nucleation events.
        - count (int): Number of nucleation events that happend during the time step.
    """
    count = 0
    n_nuc = poisson(NUC*dt)
    out = l
    for i in range(n_nuc):
        out.append(0.0)
        count += 1
    return out, count


def warmup(l,dt):
    """Performs warmup and saves warmup analysis in out/sim_SIM_ID/warmup.

    Args:
        - l (numpy.ndarray of float): Array of MT lengths before warm-up.
        - dt (float): Time step during warm-up.

    Returns:
        - out (numpy.ndarray of float): Array of MT lengths after warm-up.
    """

    #### Warmup ####
    l_mu,counts,mt_nr = [],[],[]
    count = 0
    t = np.linspace(0, N_W, N_W, endpoint=True)
    for ti in t:
        mt_nr.append(np.size(l))
        l_mu.append(np.mean(l))
        counts.append(count)
        l,count = timestep(l,dt)
    out = l

    #### Output ####
    folder_path = os.path.join(FOLDER_PATH, "warmup")
    os.makedirs(folder_path, exist_ok=True)

    ## Analyis/Plots ##
    #Plot MTnumber
    plt.plot(t,mt_nr)
    plt.xlabel("Number of timesteps")
    plt.ylabel("Number of MT")
    plt.savefig(os.path.join(folder_path, "MTnumber.png"))
    plt.close()

    #Plot MTlength
    plt.plot(t,l_mu)
    plt.axhline(y=VG/R, color="orange", linestyle="--",label="mean MT length without severing")
    plt.xlabel("Number of timesteps")
    plt.ylabel("Mean MT length (µm)")
    plt.legend()
    plt.savefig(os.path.join(folder_path, "MTlength.png"))
    plt.close()

    #Plot MTdistritubtion
    count, bins, ignored = plt.hist(l, 60, density=True)
    plt.title("MT length distibution after warm-up")
    plt.xlabel("Length (µm)")
    plt.ylabel("Count (normalized)")
    plt.savefig(os.path.join(folder_path, "MTdistritubtion.png"))
    plt.close()

    #Plot Events
    values, counts = np.unique(counts, return_counts=True)
    plt.xlabel("Number of events per timestep")
    plt.ylabel("Count")
    plt.bar(values, counts, align="center")
    plt.savefig(os.path.join(folder_path, "Events.png"))
    plt.close()

    return out


def simulation(l,dt):
    """Performs simulation and saves simulation analysis and artificial data in out/sim_SIM_ID/simulation.

    Args:
        - l (numpy.ndarray of float): Array of MT lengths after warm-up/before simulation.
        - dt (float): Time step during simulation.

    Returns:
        - out (numpy.ndarray of float): Array of MT lengths after simulation.
    """

    #### Simulation ####
    l_mu,mt_nr,counts = [],[],[]
    count = 0
    i = 0
    out = np.array([])
    t = np.linspace(0, N_S, N_S, endpoint=True)

    while i < RHO:
        for ti in t:
            counts.append(count)
            mt_nr.append(np.size(l))
            l_mu.append(np.mean(l))
            l,count = timestep(l,dt)
        out = np.concatenate((out, l))
        i += 1

    #### Output ####
    folder_path = os.path.join(FOLDER_PATH, "simulation")
    os.makedirs(folder_path, exist_ok=True)

    ## Artificial data ##
    filename = os.path.join(folder_path, "artifical_data.dat")
    file = open(filename, "w")
    for l in out:
        file.write(str(l)+"\n")
    file.close()

    ## Analyis/Plots ## 
    t = np.linspace(0, N_S*RHO, N_S*RHO, endpoint=True)

    #Plot MTnumber
    i = 1
    while i < RHO:
        plt.plot(N_S*(i+1),mt_nr[N_S*(i+1)-1],color="red", marker="o",zorder=2)
        i+=1
    plt.legend(["MT length output"])
    plt.plot(t,mt_nr,zorder=1)
    plt.xlabel("Number of timesteps")
    plt.ylabel("Number of MT")
    plt.savefig(os.path.join(folder_path, "MTnumber.png"))
    plt.close()
    
    #Plot MTlength
    plt.axhline(y=VG/R, color="orange", linestyle="--",zorder=2)
    i = 1
    while i < RHO:
        plt.plot(N_S*(i+1),l_mu[N_S*(i+1)-1],color="red", marker="o",zorder=2)
        i+=1
    plt.legend(["mean MT length if without severing","MT length output"])
    plt.plot(t,l_mu,zorder=1)
    plt.xlabel("Number of timesteps")
    plt.ylabel("Mean MT length (µm)")
    plt.savefig(os.path.join(folder_path, "MTlength.png"))
    plt.close()

    #Plot MTdistritubtion
    count, bins, ignored = plt.hist(out, 60, density=True)
    plt.title("MT length distibution after simulation")
    plt.xlabel("Length (µm)")
    plt.ylabel("Count (normalized)")
    plt.savefig(os.path.join(folder_path, "MTdistritubtion.png"))
    plt.close()

    #Plot Events
    values, counts = np.unique(counts, return_counts=True)
    plt.xlabel("Number of events per timestep")
    plt.ylabel("Count")
    plt.bar(values, counts, align="center")
    plt.savefig(os.path.join(folder_path, "Events.png"))
    plt.close()

    return out

def scale_r_dim_max(l):
    """Performs paramerter transformation from of the turnover rate from r (1/s) to r_max (-).

    Args:
        - l (numpy.ndarray of float): Array of MT lengths.

    Returns:
        - r_max (float): Non dimensional turnover rate (-), scaled with l_max.
    """
    r_max = R*max(l)/VG
    return r_max

def scale_r_dim_mu(l):
    """Performs paramerter transformation from of the turnover rate from r (1/s) to r_mu (-).

    Args:
        - l (numpy.ndarray of float): Array of MT lengths.

    Returns:
        - r_mu (float): Non dimensional turnover rate (-), scaled with l_mu.
    """
    r_mu = R*np.mean(l)/VG
    return r_mu

def scale_kappa_dim_max(l):
    """Performs paramerter transformation from of the severing rate from kappa (1/µm s) to kappa_max (-).

    Args:
        - l (numpy.ndarray of float): Array of MT lengths.

    Returns:
        - kappa_max (float): Non dimensional turnover rate (-), scaled with l_max.
    """
    kappa_max = KAPPA*max(l)**2/VG
    return kappa_max

def scale_kappa_dim_mu(l):
    """Performs paramerter transformation from of the severing rate from kappa (1/µm s) to kappa_mu (-).

    Args:
        - l (numpy.ndarray of float): Array of MT lengths.

    Returns:
        - kappa_mu (float): Non dimensional turnover rate (-), scaled with l_mu.
    """
    kappa_mu = KAPPA*np.mean(l)**2/VG
    return kappa_mu

def document(l):
    """Saves all relevant parameters of the simulation in out/sim_documentation.dat.

    Args:
        - l (numpy.ndarray of float): Array of MT lengths after simulation.

    Returns:
        - None
    """
    variables = {
        "SIM_ID": SIM_ID,                                   #Simulation ID
        "mt_nr": len(l),                                    #Number of MT
        "l_max (\u03BCm)": round(max(l),3),                 #Max MT length of distribution (µm)
        "l_mu (\u03BCm)": round(np.mean(l),3),              #Mean MT length of distribution (µm)
        "ALPHA (-)": ALPHA,                                 #Stability parameter (-)
        "R (1/s)": R,                                       #Turnover rate (1/s)
        "KAPPA (1/\u03BCm s)": KAPPA,                       #Severing rate (1/µm s)
        "r_max (-)" : round(scale_r_dim_max(l),2),          #Non dimensional turnover rate (-), scaled with l_max
        "kappa_max (-)" : round(scale_kappa_dim_max(l),2),  #Non dimensional severing rate (-), scaled with l_max
        "r_mu (-)": round(scale_r_dim_mu(l),2),             #Non dimensional turnover rate (-), scaled with l_mu
        "kappa_mu (-)": round(scale_kappa_dim_mu(l),2),     #Non dimensional severing rate (-), scaled with l_mu
        "VG (\u03BCm/s)": VG,                               #Growth velocity (µm/s)
        "NUC (1/s)": NUC,                                   #Nucleation rate (1/s)
        "N0": N0,                                           #Initial number of MT
        "DT_W (s/timestep)": DT_W,                          #Timestep during warm up (s/timestep)
        "N_W": N_W,                                         #Number of timesteps during warm up
        "DT_S (s/timestep)": DT_S,                          #Timestep during simulation (s/timestep)
        "N_S": N_S,                                         #Number of timesteps between length distribution output(s)
        "RHO": RHO,                                         #Number of length distribution outputs
        "Comment": COMMENT                                  #Comment 
    }

    out_directory = os.path.join(os.path.abspath(os.getcwd()), "out")
    os.makedirs(out_directory, exist_ok=True)
    file_name = os.path.join(out_directory, "sim_documentation.dat")
    if os.path.exists(file_name):
        with open(file_name, "a") as file:
            values = [str(variables[var]) for var in variables]
            line = "\t".join(values)
            file.write(line + "\n")
    else:
        with open(file_name, "w", encoding="utf-8") as file:
            headings = "\t".join(variables.keys())
            values = "\t".join(map(str, variables.values()))
            file.write(headings + "\n")
            file.write(values + "\n")
            
    return None

########################################## MAIN ##########################################

"""Define MT parameters, Initals, Parameters of simulation and Comment here.
"""
    
### Make paths ###
SIM_ID = "sim_" + datetime.datetime.now().strftime("%y%m%d%H%M%S")
FOLDER_PATH = os.path.join(os.path.abspath(os.getcwd()), "out", SIM_ID)
os.makedirs(FOLDER_PATH, exist_ok=True)

#### MT parameters ####
R = 0.7                                 #Turnover rate (1/s)
KAPPA = 0                               #Severing rate (1/µm s)
ALPHA = 0.5                             #Stability parameter (-)
VG = 1                                  #Growth velocity (µm/s)
NUC = 1000                              #Nucleation rate (1/s)

#### Initals ####
N0 = 1                                  #Initial number of MT
l = np.zeros(N0)                        #Initial length distribution

#### Parameters of simulation #### 
DT_W = 0.01                             #Timestep during warm up (s/timestep)
N_W = 2                                 #Number of timesteps during warm up
DT_S = 0.001                            #Timestep during simulation (s/timestep)             
N_S = 2                                 #Number of timesteps between length distribution output(s)
RHO = 4                                 #Number of length distribution outputs

COMMENT = "Here could be your Comment"

l = warmup(l,DT_W)
l = simulation(l,DT_S)
document(l)
print("File Executed")

########################################## EOF ##########################################