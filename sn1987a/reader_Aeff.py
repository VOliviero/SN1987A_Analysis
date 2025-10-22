import numpy as np
from os import listdir
from os.path import isfile, join, dirname, abspath
import pandas as pd
import scipy
import scipy.interpolate



class Aeff_reader:
    """
    Class to read A_eff files. Usage:
    A_eff = Aeff_reader("k2",E_thr) # Select experiment, and E_thr
    The class throws an exception if the file is not found.

    To get a value based on the Ev, use
    A_eff.eval(Ev)

    If a value is evaluated outside the bounds of the tabulated function, np.nan is returned
    """

    def __init__(self,experiment,E_thr,folder_path=None):

        if folder_path == None:
            folder_path = dirname(dirname(abspath(__file__)))+"/tables/Aeff_tables/"

        self.folder_path = folder_path + "/"
        self.files = [f for f in listdir(self.folder_path) if isfile(join(self.folder_path, f))]

        filename = lambda experiment,E_thr : f"Aeff_{experiment}_thr{E_thr:.1f}.tsv"
        self.filename = filename(experiment,E_thr)

        if self.filename not in self.files:
            raise Exception(f"File not found for experiment {experiment} with E_thr = {E_thr}")

        self.data   = pd.read_csv(self.folder_path+self.filename,header=0,sep="\t")
        self.Ev     = self.data["Ev[MeV]"].to_numpy()
        self.Ev_min = np.min(self.Ev)
        self.Ev_max = np.max(self.Ev)
        self.Aeff   = self.data["Aeff[MeV-2]"].to_numpy()
        

    '''def eval(self,Ev):
        #if Ev < self.Ev_min or Ev > self.Ev_max:
        #    raise Exception(f"Ev out of bounds {Ev}, the minimum is {self.Ev_min}!")
        return np.interp(Ev,self.Ev,self.Aeff,left=np.nan,right=np.nan)'''
        
    def eval(self, Ev):
    # Creiamo una funzione di interpolazione utilizzando interp1d
        interp_func = scipy.interpolate.interp1d(self.Ev, self.Aeff, kind='cubic',bounds_error=False, fill_value=0)
    
    # Applichiamo la funzione di interpolazione alla variabile T
        return interp_func(Ev)
    

if __name__ == "__main__":

    from sn1987a import expdata
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    expname = {"k2" : "Kamiokande2", "bk": "Baksan", "im": "IMB"}
    filename = [f for f in listdir(".\\Aeff_tables\\") if isfile(join(".\\Aeff_tables\\", f))]
    E_thr_values = {exp : [float(f.split("_")[-1].rstrip(".tsv").lstrip("thr")) for f in filename if f.split("_")[1]==exp] for exp in expname}
    fig,axs = plt.subplots(1,3,figsize=(15,4))
    

    for j in range(3):
        ax = axs[j]
        experiment = ("k2","bk","im")[j]
        
        ax.set_xlabel("Ev [MeV]")
        ax.set_ylabel(r"Aeff [MeV$^{-2}$]")

        cm = [mpl.cm.Blues,mpl.cm.Greens,mpl.cm.OrRd]
        ax.set_prop_cycle('color', cm[j](np.linspace(1,0,20)))
        ax.set_title(expname[experiment])
        
        #ax.set_yscale("log")
        ax.grid(which="both",alpha=0.5)
        #ax.set_ylim(1,1e14)
        #ax.set_xscale("log")
        for E_thr in E_thr_values[experiment]:
            Aeff_read = Aeff_reader(experiment,E_thr)
            Ev = np.arange(Aeff_read.Ev_min,Aeff_read.Ev_max,0.01)
            ax.plot(Ev,Aeff_read.eval(Ev))
            print(f"Plotted Aeff with E_thr = {E_thr} MeV for experiment {experiment}")


        for n_event,Ei,_ in expdata.data[experiment]:
            ax.axvline(Ei,color="red",lw=1,alpha=0.2)

    plt.show()

