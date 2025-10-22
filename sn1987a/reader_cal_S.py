import numpy as np
from os import listdir
from os.path import isfile, join, dirname, abspath
import pandas as pd
import scipy
import scipy.interpolate



class calS_reader:
    """
    Class to read cal_S files. Usage:
    cal_S = calS_reader("k2","a") # Select experiment, and phase ("a" : accretion, "c" : cooling)
    The class raises an exception if the file is not found.

    To get a value based on the temperature T , use
    cal_S.eval(T) (T in [MeV])

    If a value is evaluated outside the bounds of the tabulated function, np.nan is returned
    """

    def __init__(self,experiment,phase,folder_path=None,):

        if folder_path == None:
            folder_path = dirname(dirname(abspath(__file__)))+"/tables/cal_s_tables/"
        
        self.folder_path = folder_path + "/"
        self.files = [f for f in listdir(self.folder_path) if isfile(join(self.folder_path, f))]

        filename = lambda experiment,phase : f"cal_S_{experiment}_{phase}.tsv"
        self.filename = filename(experiment,phase)

        if self.filename not in self.files:
            raise Exception(f"File not found for experiment {experiment} with phase {phase}")

        self.data   = pd.read_csv(self.folder_path+self.filename,header=1,sep="\t")
        self.T     = self.data["T[MeV]"].to_numpy()
        self.T_min = np.min(self.T)
        self.T_max = np.max(self.T)
        
        self.calS = self.cal_S = self.data["cal_S"].to_numpy()
        self.calS_error = self.cal_S_error = self.data["cal_S_error"].to_numpy()

    '''def eval(self,T):
        #if Ev < self.Ev_min or Ev > self.Ev_max:
        #    raise Exception(f"Ev out of bounds {Ev}, the minimum is {self.Ev_min}!")
        return np.interp(T,self.T,self.cal_S,left=0,right=0)'''     

    def eval(self, T):
    # Creiamo una funzione di interpolazione utilizzando interp1d
        interp_func = scipy.interpolate.interp1d(self.T, self.cal_S, kind='cubic',bounds_error=False, fill_value=0)
    
    # Applichiamo la funzione di interpolazione alla variabile T
        return interp_func(T)
        
class manager():

    def __init__(self,folder_path=None):
        self.folder_path = folder_path
        self.readers = dict()
        pass

    def eval(self,T,experiment,phase):
        key = (experiment,phase)
        
        if key in self.readers.keys():
            return self.readers[key].eval(T)

        else:
            self.readers[key] = calS_reader(experiment,phase,self.folder_path)
            return self.readers[key].eval(T)
    

if __name__ == "__main__":

    import expdata
    import matplotlib as mpl
    
    import matplotlib.pyplot as plt
    
    import numpy as np
    
    expname = {"k2" : "Kamiokande2", "bk": "Baksan", "im": "IMB", "ls": "LSD"}
    filename = [f for f in listdir("../tables/Aeff_tables/") if isfile(join("./Aeff_tables/", f))]


    fig,axs = plt.subplots(2,3,figsize=(15,8))
    
    for i in range(2):
        for j in range(3):

            ax = axs[i,j]
            experiment  = ("k2","bk","im")[j]
            phase       = ("a","c")[i]

            ax.set_xlabel("T [MeV]")
            ax.set_ylabel("cal_S")

            cm = [mpl.cm.Blues,mpl.cm.Greens,mpl.cm.OrRd]
            ax.set_prop_cycle('color', cm[j](np.linspace(0.5,0,20)))
            if i==0:
                ax.set_title(expname[experiment])
            
            ax.set_yscale("log")
            ax.grid(which="both",alpha=0.5)
            #x.set_ylim(1,1e14)
            #ax.set_xscale("log")

            cal_s_read = calS_reader(experiment,phase)
            ax.plot(cal_s_read.T,cal_s_read.cal_S,label="cal_S_"+experiment+"_"+phase)
            ax.legend()
            print(f"Plotted calS_{phase} with for experiment {experiment}")

    plt.show()

