import numpy as np
from os import listdir
from os.path import isfile, join, dirname, abspath
import pandas as pd
import scipy
import scipy.interpolate



class f_reader:
    """
    Class to read f_function files. Usage:
    f_func = f_reader("k2","a",1) # Select experiment, phase and event number
    The class throws an exception if the file is not found.

    To get a value based on the temperature, use
    f_func.eval(temperature)

    If a value is evaluated outside the bounds of the tabulated function, np.nan is returned
    """

    def __init__(self,experiment,phase,event,folder_path=None):

        if folder_path == None:
            folder_path = dirname(dirname(abspath(__file__)))+"/tables/f_functions/"

        self.folder_path = folder_path + "/"
        self.files = [f for f in listdir(self.folder_path) if isfile(join(self.folder_path, f))]

        filename = lambda experiment,phase,event : f"f_func_{experiment}_{phase}_ev{event:02d}.tsv"
        self.filename = filename(experiment,phase,event)

        if self.filename not in self.files:
            raise Exception(f"File not found for experiment {experiment} phase {phase} event number {event}")

        self.data   = pd.read_csv(self.folder_path+self.filename,header=0,sep="\t")
        self.T      = self.data["T[MeV]"].to_numpy()
        self.T_min  = np.min(self.T)
        self.T_max  = np.max(self.T)
        self.f_value= self.data["f_value"].to_numpy()
        

    '''def eval(self,T):
        return np.interp(T,self.T,self.f_value,left=np.nan,right=np.nan)'''
    
    def eval(self, T):
    # Creiamo una funzione di interpolazione utilizzando interp1d
        interp_func = scipy.interpolate.interp1d(self.T, self.f_value, kind='cubic',bounds_error=False, fill_value=0)
    
    # Applichiamo la funzione di interpolazione alla variabile T
        return interp_func(T)

class manager():

    def __init__(self,folder_path=None):
        self.folder_path = folder_path
        self.readers = dict()
        pass

    def eval(self,T,experiment,phase,event):
        if experiment == "ls":
            return 0

        key = (experiment,phase,event)
        
        if key in self.readers.keys():
            return self.readers[key].eval(T)

        else:
            self.readers[key] = f_reader(experiment,phase,event,self.folder_path)
            return self.readers[key].eval(T)



if __name__ == "__main__":

    import expdata
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    
    fig,axs = plt.subplots(2,3,figsize=(15,8))
    expname = {"k2" : "Kamiokande2", "bk": "Baksan", "im": "IMB"}

    for i in range(2):
        for j in range(3):
            ax = axs[i,j]
            phase = ("a","c")[i]
            experiment = ("k2","bk","im")[j]
            
            ax.set_xlabel("T_"+phase+" [MeV]")
            ax.set_ylabel("f_"+phase)

            cm = [mpl.cm.Blues,mpl.cm.YlGn,mpl.cm.OrRd]
            ax.set_prop_cycle('color', cm[j](np.linspace(1,0,20)))


            if i == 0:
                ax.set_title(expname[experiment])
            ax.set_yscale("log")
            ax.grid(which="both",alpha=0.5)
            for n,_,_,_,_ in expdata.data[experiment]:
                f_func = f_reader(experiment,phase,n)
                T = np.arange(f_func.T_min,f_func.T_max,0.01)
                ax.plot(T,f_func.eval(T))
                print(f"Plotted event {n:2d} for experiment {experiment} phase {phase}")

    plt.show()

