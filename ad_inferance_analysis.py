""" This module performes a Baysian inferance analysis on artificial microtubule (MT) length data.
"""
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import lpd_numpy as lpd_n
import lpd_pytensor as lpd_p
import os
import datetime
import arviz.labels as azl

########################################## FUNCTIONS ##########################################

def ad_load_data(n=0):
    """Loads random all entries (MT lengths) from DATA_PATH if n=0, or n enrties if n != 0 and scales it to the max length.

    Args:
        - n (int): Number of entries to be loaded. If no arg is given, n defaults to 0 and all data is loaded.

    Returns:
        - dat (numpy.ndarray of float): n/all data entries (MT lengths) scaled to the max length.
        - l_mu (float): Mean MT length of n/all data entries (MT lengths).
        - l_max (float): Max MT length of n/all data entries (MT lengths).
        - mt_nr (int): Number of MT.
    """
    try:
        dat = np.loadtxt(DATA_PATH, dtype=float)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    dat = []
    file = np.loadtxt(DATA_PATH, dtype = float)
    for line in file:
        dat.append(line)
    if n == 0: n = len(dat)
    dat = np.random.choice(dat, size=int(n), replace=False)
    l_max = max(dat)
    l_mu = np.mean(dat)
    mt_nr = len(dat)
    dat = dat/l_max
    return dat, l_mu, l_max, mt_nr

def bin(dat):
    """Sorts the data (MT lengths) in to bins of equal size distributed from (scaled) length 0 to 1 (as specified in lpd_numpy.py and lpd_pytensor.py).

    Args:
        - dat (numpy.ndarray of float): Array of MT lengths scaled to the max length.

    Returns:
        - dat_b (numpy.ndarray of int): Binned data (MT lengths).
        - b (numpy.ndarray of float): Bin locations on x-axis.
    """
    dat_b, b = np.histogram(dat,bins=np.linspace(0,1,lpd_p.NL+1))
    return dat_b, b

def scale_r_max_dim(r_max,l_max):
    """Performs paramerter transformation from of the turnover rate from r_max (-) to r (1/s).

    Args:
        - r_max (float): Non dimensional turnover rate (-), scaled with l_max.
        - l_max (float): Max MT length of n/all data entries (MT lengths).

    Returns:
        - r (float): Turnover rate (1/s).
    """
    gt_vg = load_gt()[3]
    r = r_max*gt_vg/l_max
    return r

def scale_kappa_max_dim(kappa_max,l_max):
    """Performs paramerter transformation of the severing rate from kappa_max (-) to kappa (1/µm s).

    Args:
        - kappa_max (float): Non dimensional severing rate (-), scaled with l_max.
        - l_max (float): Max MT length of n/all data entries (MT lengths).

    Returns:
        - kappa (float): Severing rate (1/µm s).
    """
    gt_vg = load_gt()[3]
    kappa = kappa_max*gt_vg/l_max**2
    return kappa

def scale_r_max_mu(r_max,l_max,l_mu):
    """Performs paramerter transformation from of the turnover rate from r_max (-) to r_mu (-).

    Args:
        - r_max (float): Non dimensional turnover rate (-), scaled with l_max.
        - l_mu (float): Mean MT length of n/all data entries (MT lengths).
        - l_max (float): Max MT length of n/all data entries (MT lengths).

    Returns:
        - r_mu (float): Non dimensional turnover rate (-), scaled with l_mu.
    """
    r_mu = r_max*l_mu/l_max
    return r_mu

def scale_kappa_max_mu(kappa_max,l_max,l_mu):
    """Performs paramerter transformation from of the severing rate from kappa_max (-) to kappa_mu (-).

    Args:
        - kappa_max (float): Non dimensional severing rate (-), scaled with l_max.
        - l_mu (float): Mean MT length of n/all data entries (MT lengths).
        - l_max (float): Max MT length of n/all data entries (MT lengths).

    Returns:
        - kappa_mu (float): Non dimensional severing rate (-), scaled with l_mu.
    """
    kappa_mu = kappa_max*l_mu**2/l_max**2
    return kappa_mu

def load_gt():
    """Loads the ground truth parameters from sim_documentation.dat.

    Args: 
        - None

    Returns:
        - gt_r_max (float): Ground truth of non dimensional turnover rate (-), scaled with l_max.
        - gt_kappa_max (float): Ground truth of non dimensional severing rate (-), scaled with l_max.
        - gt_alpha (float): Ground truth of stability parameter (-).
        - gt_vg (float): Ground truth of qrowth velocity (µm/s).
    """
    sim_doc = os.path.join(os.path.abspath(os.getcwd()), "out", "sim_documentation.dat")
    with open(sim_doc, "r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split("\t")
        traget_id = str(parts[0])
        if  traget_id == SIM_ID:
            gt_vg = float(parts[11])
            gt_alpha = float(parts[4])
            gt_r_max = float(parts[7])
            gt_kappa_max = float(parts[8])
            break
    return gt_r_max, gt_kappa_max, gt_alpha, gt_vg

def ad_document(inf_dat,mt_nr,l_max,l_mu):
    """Saves the inferred and other relevant parameters in out/ad_inf_documentation.dat.

    Args:
        - inf_dat (arviz.data.inference_data.InferenceData): Object that contains the samples.
        - mt_nr (int): Number of MT.
        - l_mu (float): Mean MT length of n/all data entries (MT lengths).
        - l_max (float): Max MT length of n/all data entries (MT lengths).

    Returns:
        - None
    """
    stats = az.summary(inf_dat)

    variables = {
        "SIM_ID": SIM_ID,                                          #Simulation ID
        "INF_ID": INF_ID,                                          #Inferance ID
        "mt_nr": mt_nr,                                            #Number of MT in p% of data                 
        "l_max (\u03BCm)": round(l_max,3),                         #Max MT length in p% of data               
        "l_mu (\u03BCm)": round(l_mu,3),                           #Mean MT length in p% of data
        "alpha_inf (-)": round(stats.iloc[0,0],2),                 #Inferred stability parameter (-)
        "inf_r (1/s)": round(stats.iloc[5,0],2),                   #Inferred turnover rate (1/s)
        "inf_kappa (1/\u03BCm s)": round(stats.iloc[6,0],2),       #Inferred severing rate (1/µm s)
        "inf_r_max (-)": round(stats.iloc[1,0],2),                 #Inferred non dimensional turnover rate (-), scaled with l_max
        "inf_kappa_max (-)": round(stats.iloc[2,0],2),             #Inferred non dimensional severing rate (-), scaled with l_max
        "inf_r_mu (-)": round(stats.iloc[3,0],2),                  #Inferred non dimensional turnover rate (-), scaled with l_mu
        "inf_kappa_mu (-)": round(stats.iloc[4,0],2),              #Inferred non dimensional severing rate (-), scaled with l_mu
        "Comment": COMMENT                                         #Comment 
    }

    out_directory = os.path.join(os.path.abspath(os.getcwd()), "out")
    os.makedirs(out_directory, exist_ok=True)
    file_name = os.path.join(out_directory, "ad_inf_documentation.dat")
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

def ad_make_figures(inf_dat,type = "mu"):
    """Saves figures (Pair, Posterior, Trace_alpha, Trace_r_max, Trace_kappa_max) in out/ad_inf_INF_ID.

    Args:
        - inf_dat (arviz.data.inference_data.InferenceData): Object that contains the samples.
        - type (string): Indicates which sacling ("max", "mu", "dim") will be used for the figures.

    Returns:
        - None
    """
    os.makedirs(FOLDER_PATH, exist_ok=True)

    #Plot Pair
    if type == "max":
        labeller = azl.MapLabeller(var_name_map={"inf_r_max": "R (-)","inf_kappa_max": "K (-)","inf_alpha": r"$\alpha$ (-)"})
        variables = ["inf_r_max", "inf_kappa_max", "inf_alpha"]
    elif type == "mu":
        labeller = azl.MapLabeller(var_name_map={"inf_r_mu": r"$\overline{R} (-)$","inf_kappa_mu": r"$\overline{K} (-)$","inf_alpha": r"$\alpha$ (-)"})
        variables = ["inf_r_mu", "inf_kappa_mu", "inf_alpha"]
    else: 
        labeller = azl.MapLabeller(var_name_map={"inf_r": "r (1/s)","inf_kappa": r"$\kappa (1/\mu m s)$","inf_alpha": r"$\alpha (-)$"})
        variables = ["inf_r", "inf_kappa", "inf_alpha"]
    
    pm.plot_pair(inf_dat, 
                    kind="kde",
                    marginals=False,
                    point_estimate="mean",
                    labeller=labeller,
                    var_names=variables,
                    divergences=False,)
    plt.savefig(os.path.join(FOLDER_PATH, f"Pair_{type}.png"))
    plt.close()

    #Plot Posterior
    az.plot_posterior(inf_dat,
                    kind="kde", 
                    labeller=labeller,
                    var_names=variables,
                    hdi_prob=0.95,)
    plt.savefig(os.path.join(FOLDER_PATH, f"Posterior_{type}.png"))
    plt.close()

    #Plot Trace_alpha
    for var in variables: 
        az.plot_trace(inf_dat,
                        var_names=var, 
                        legend=True,
                        labeller=labeller,
                        divergences=False,)
        plt.savefig(os.path.join(FOLDER_PATH, f"Trace_{var}_{type}.png"))
        plt.close()

    return None

def ad_make_pdist(inf_dat,dat_b,b,l_max):
    """Saves figures (Pdist, Error) in out/ad_inf_INF_ID.

    Args:
        - inf_dat (arviz.data.inference_data.InferenceData): Object that contains the samples.
        - dat_b (numpy.ndarray of int): Binned data (MT lengths).
        - b (numpy.ndarray of float): Position of bins.
        - l_max (float): Max MT length in p% of data (MT lengths).

    Returns:
        - None
    """
    os.makedirs(FOLDER_PATH, exist_ok=True)
    
    #Plot Pdist
    stats = az.summary(inf_dat)
    inf_alpha = stats.iloc[0,0]
    inf_r_max = stats.iloc[1,0]
    inf_kappa_max = stats.iloc[2,0]
    inf = lpd_n.pdist(inf_r_max, inf_kappa_max, inf_alpha)
    plt.plot(lpd_n.l*l_max,inf/sum(inf*lpd_n.delta*l_max), "--r",label = "Inferred distribution")

    gt_r_max, gt_kappa_max, gt_alpha, gt_vg = load_gt()
    gt = lpd_n.pdist(gt_r_max, gt_kappa_max, gt_alpha)
    plt.plot(lpd_n.l*l_max,gt/sum(gt*lpd_n.delta*l_max), "-b", label = "Ground truth")

    plt.plot(b[0:-1]*l_max,dat_b/sum(dat_b*lpd_n.delta*l_max), ".g", label = "Binned data") 

    plt.legend()
    plt.xlabel("Length l $(\mu m)$")
    plt.ylabel("$\phi(l)$")
    plt.savefig(os.path.join(FOLDER_PATH, "Pdist.png"))
    plt.close()

    #Plot Error
    error = (inf-gt)**2
    plt.plot(lpd_n.l*l_max,error, "r")
    plt.xlabel("Length l $(\mu m)$")
    plt.ylabel("Squared error")
    plt.savefig(os.path.join(FOLDER_PATH, "Error.png"))
    plt.close()

    return None

########################################## MAIN ##########################################

#### ENTER DATA HERE  ####
SIM_ID = "Enter sim_SIM_ID here"
COMMENT = "Here could be your comment."

### Make paths ###
INF_ID = "ad_inf_" + datetime.datetime.now().strftime("%y%m%d%H%M%S")
DATA_PATH = os.path.join(os.path.abspath(os.getcwd()), f"out/{SIM_ID}/simulation/artifical_data.dat")
FOLDER_PATH = os.path.join(os.path.abspath(os.getcwd()), "out", INF_ID)

### Load and bin data ####
dat, l_mu, l_max, mt_nr = ad_load_data()
dat_b,b = bin(dat)

#### Inferance ####
with pm.Model() as mod_two:
    inf_alpha = pm.Uniform("inf_alpha", lower=0, upper=1)
    inf_r_max = pm.Uniform("inf_r_max", lower=0, upper=10)                   
    inf_kappa_max = pm.Uniform("inf_kappa_max", lower=0, upper=20)                 

    inf_r_mu = pm.Deterministic("inf_r_mu", scale_r_max_mu(inf_r_max,l_max,l_mu))
    inf_kappa_mu = pm.Deterministic("inf_kappa_mu", scale_kappa_max_mu(inf_kappa_max,l_max,l_mu))

    inf_r= pm.Deterministic("inf_r", scale_r_max_dim(inf_r_max,l_max))
    inf_kappa = pm.Deterministic("inf_kappa", scale_kappa_max_dim(inf_kappa_max,l_max))

    p = lpd_p.pdist(inf_r_max,inf_kappa_max,inf_alpha)                                
    p = p/p.sum()*mt_nr
    l = pm.Poisson("l", mu=p, observed=dat_b)
    inf_dat = pm.sample(draws=2, tune=2, target_accept=0.8)

#### Document ####
ad_document(inf_dat,mt_nr,l_max,l_mu)
ad_make_figures(inf_dat,type = "mu")
ad_make_pdist(inf_dat,dat_b,b,l_max)
print("File Executed")

########################################## EOF ##########################################