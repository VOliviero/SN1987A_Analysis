import numpy as np
from os import listdir
from os.path import isfile, join, dirname, abspath
import pandas as pd
import scipy
import scipy.interpolate

class calG_reader:
    """
    Class to read cal_G files. Usage:
    cal_G = calG_reader("k2","a") # Select experiment, and phase ("a" : accretion, "c" : cooling)
    The class raises an exception if the file is not found.

    This function is one or bi-dimensional depending on phase.
    "a" : 1 var. (t0_over_tau)      |  To call: cal_G.eval(t0_over_tau)
    "c" : 2 var.s (T0,t0_over_tau)  |  To call: cal_G.eval((T0,t0_over_tau))

    If a value is evaluated outside the bounds of the tabulated function, error is raised    """

    def __init__(self,experiment,phase,folder_path=None):

        if folder_path == None:
            folder_path = dirname(dirname(abspath(__file__)))+"/tables/cal_g_tables/"

        self.phase = phase
        self.folder_path = folder_path + "/"
        self.files = [f for f in listdir(self.folder_path) if isfile(join(self.folder_path, f))]

        self.filename = self.retrieve_filename(experiment,phase)

        if self.filename not in self.files:
            raise Exception(f"File not found for experiment {experiment} with phase {phase}")

        self.data   = pd.read_csv(self.folder_path+self.filename,header=0,sep="\t")
        """
        self.data per (a) ha come variabile t0/tau e ritorna G
        self.data per (c) ha come variabili T0 e t0/tau e ritorna G
        """

        # Retrieve data
        self.calG = self.data["cal_g_function_a"].to_numpy()
        self.calG_error = self.data["err"].to_numpy()

        # Retrieve bounds and points for grid
        self.retrieve_bounds()
        self.points = self.eval_points()

        # define interpolator
        if phase == "a":
            self.interpolator = scipy.interpolate.interp1d(self.points,
                                                            self.calG,
                                                            kind='cubic',
                                                            bounds_error=True)

        elif phase == "c":
            self.shape = [len(dimension) for dimension in self.points]
            # Reshape data for the grid
            data_reshaped = np.reshape(self.calG,self.shape)
            self.interpolator= scipy.interpolate.RegularGridInterpolator(self.points,
                                                                         data_reshaped,
                                                                         method='cubic',
                                                                         bounds_error=True,)
        
    def retrieve_filename(self,experiment,phase):
        if phase == "a":
            return f"cal_g_{phase}.tsv"
        if phase == "c":
            return f"cal_g_{phase}_{experiment}.tsv"
        else:
            raise Exception(f"Unknown phase in retrieve_filename: {phase}")

    def eval_points(self):
        if self.phase == "a":
            return np.unique(self.data["t0/tau"])
        if self.phase == "c":
            return [np.unique(self.data["T0"]),np.unique(self.data["t0/tau"])]
        else:
            raise Exception(f"Unknown phase in point definer: {self.phase}")
        
    def eval(self,t0_over_tau,T0=None):
        """ eval point:
            "a" : 1 var. (t0_over_tau)    
            "c" : 2 var.s (T0,t0_over_tau)
        """
        if self.phase == "a":
            if T0 != None:
                raise Exception("T0 is not used for accretion!")
            return self.interpolator(t0_over_tau)
        
        if self.phase == "c":
            if T0 == None:
                raise Exception("T0 is needed for cooling!")
            return self.interpolator((T0,t0_over_tau))
        

    def retrieve_bounds(self):
        """Retrieve bounds"""
        if self.phase == "c":
            self.T0     = self.data["T0"].to_numpy() #MeV
            self.T0_min = np.min(self.T0)
            self.T0_max = np.max(self.T0)

        self.t0_over_tau     = self.data["t0/tau"].to_numpy()
        self.t0_over_tau_min = np.min(self.t0_over_tau)
        self.t0_over_tau_max = np.max(self.t0_over_tau)
        
class manager():

    def __init__(self,folder_path=None):
        self.folder_path = folder_path
        self.readers = dict()
        pass

    def eval(self,experiment,phase,t0_over_tau,T0=None):
        key = (experiment,phase)
        
        if key not in self.readers.keys():
            self.readers[key] = calG_reader(experiment,phase,self.folder_path)

        try:
            #string = f"{type(np.isnan(T0))} // {type(t0_over_tau)}"
            return self.readers[key].eval(t0_over_tau=t0_over_tau,T0=T0)
        
        except Exception as e:
            string_error = "".join((f"Error with calG_reader {experiment} ({phase}) with t0_over_tau = {t0_over_tau} and T0 = {T0}.\n",
                     f"Bounds for t0_over_tau = {self.readers[key].t0_over_tau_min:.04f} - {self.readers[key].t0_over_tau_max:.04f}\n"))
            if phase == "c":
                string_error = string_error + f"Bounds for T0 = {self.readers[key].T0_min:.04f} - {self.readers[key].T0_max:.04f}\n"
            raise Exception(string_error+f"{e}")
        
if __name__ == "__main__":

    import expdata
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    
    expname = {"k2" : "Kamiokande2", "bk": "Baksan", "im": "IMB", "ls":"LSD"}

    print("="*50+"\nTesting reader of CalG...\n"+"="*50)

    calGread= manager()

    eval_points = (5,0.0)
    # experiment = "k2"
    # print(f"Example of cooling evaluation with points {eval_points} on experiment {experiment}")
    # print(calGread.eval(eval_points,experiment,"c"))

    # print(f"\nExample of accretion evaluation with t0/tau = {eval_points[-1]} on experiment {experiment}")
    # print(calGread.eval(eval_points[-1],experiment,"a"))

    fig = plt.figure(figsize=(15, 4))
    t0_over_tau_range = (1,9.9)
    t0_over_tau = np.linspace(*t0_over_tau_range,2000)

    # Big plot on the top
    ax1 = fig.add_subplot(1,4,1)

    ax1.plot(t0_over_tau, calGread.eval("k2","a",t0_over_tau), color="black", label='Accretion CalG')
    ax1.set_title('Accretion Phase (single G function)')
    ax1.set_xlabel("t0/tau")
    ax1.set_ylabel("CalG")
    ax1.set_xlim(*t0_over_tau_range)

    t0_over_tau_range = (1,9.9)
    t0_over_tau = np.linspace(*t0_over_tau_range,200)
    T0_range = (1,9.9)
    T0 = np.linspace(*T0_range,400)
    z2 = np.array([[calGread.eval("k2","c",t0_over_tau_value,T0_value) for T0_value in T0] for t0_over_tau_value in t0_over_tau ])

    # Small plots on the bottom half
    ax2 = fig.add_subplot(1,4,2, projection="3d")
    ax2.contourf(T0, t0_over_tau, np.log10(z2), 100)
    ax2.set_title('Kamiokande Cooling CalG')
    ax2.set_xlabel("T0")
    ax2.set_ylabel("t0/tau")
    ax2.set_zlabel("log_10(CalG)")


    z3 = np.array([[calGread.eval("im","c",t0_over_tau_value,T0_value) for T0_value in T0] for t0_over_tau_value in t0_over_tau ])

    ax3 = fig.add_subplot(1,4,3, projection='3d')
    ax3.contourf(T0, t0_over_tau, np.log10(z3), 100)
    ax3.set_title('IMB Cooling CalG')
    ax3.set_xlabel("T0")
    ax3.set_ylabel("t0/tau")
    ax3.set_zlabel("log_10(CalG)")
    #ax3.legend()

    z4 = np.array([[calGread.eval("bk","c",t0_over_tau_value,T0_value) for T0_value in T0] for t0_over_tau_value in t0_over_tau ])
    ax4 = fig.add_subplot(1,4,4, projection='3d')
    ax4.contourf(T0, t0_over_tau, np.log10(z4), 100)
    ax4.set_title('Baksan Cooling CalG')
    ax4.set_xlabel("T0")
    ax4.set_ylabel("t0/tau")
    ax4.set_zlabel("log_10(CalG)")
    #ax4.legend()

    plt.tight_layout()
    plt.show()