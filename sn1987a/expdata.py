"""
Module containing experimental data.
How to use. Import this module into another module.
Then call expdata.get_df(experiment) to retrieve a pandas dataframe
where experiment is "bk","k2" or "im".
The dataframe contains the following columns
"exp","exp_id","E_MeV","angle_deg","cos_angle","time_ms","time_s","background"

"""
import pandas as pd
import numpy as np

# Note: the last four values in k2 were set to 90 angle
# Data Format (ID, exp_id,          Time[ms],Ei[MeV],angle[deg],bkgd )
data_bk = ((    "bk_01",     1,     0,      12.0,   90,     0.00084,    ),
           (    "bk_02",     2,     435,    17.9,   90,     0.0013,     ),
           (    "bk_03",     3,     1710,   23.5,   90,     0.0012,     ),
           (    "bk_04",     4,     7687,   17.5,   90,     0.0013,     ),
           (    "bk_05",     5,     9099,   20.3,   90,     0.0013,     ))
data_k2 = ((    "k2_01",     1,     0,      20.0,   18,     1e-05,      ),
           (    "k2_02",     2,     107,    13.5,   40,     0.00054,    ),
           (    "k2_03",     3,     303,    7.5,    108,    0.024,      ),
           (    "k2_04",     4,     324,    9.2,    70,     0.0028,     ),
           (    "k2_05",     5,     507,    12.8,   135,    0.00053,    ),
           (    "k2_06",     6,     686,    6.3,    68,     0.079,      ),
           (    "k2_07",     7,     1541,   35.4,   32,     5e-06,      ),
           (    "k2_08",     8,     1728,   21.0,   30,     1e-05,      ),
           (    "k2_09",     9,     1915,   19.8,   38,     1e-05,      ),
           (    "k2_10",    10,     9219,   8.6,    122,    0.0042,     ),
           (    "k2_11",    11,     10433,  13.0,   49,     0.0004,     ),
           (    "k2_12",    12,     12439,  8.9,    91,     0.0032,     ),
           (    "k2_13",    13,     17641,  6.5,    90,     0.073,      ),
           (    "k2_14",    14,     20257,  5.4,    90,     0.053,      ),
           (    "k2_15",    15,     21355,  4.6,    90,     0.018,      ),
           (    "k2_16",    16,     23814,  6.5,    90,     0.073,      ))
data_im = ((    "im_01",     1,     0,      38.0,   80,     1e-05,      ),
           (    "im_02",     2,     412,    37.0,   44,     1e-05,      ),
           (    "im_03",     3,     650,    28.0,   56,     1e-05,      ),
           (    "im_04",     4,     1141,   39.0,   65,     1e-05,      ),
           (    "im_05",     5,     1562,   36.0,   33,     1e-05,      ),
           (    "im_06",     6,     2684,   36.0,   52,     1e-05,      ),
           (    "im_07",     7,     5010,   19.0,   42,     1e-05,      ),
           (    "im_08",     8,     5582,   22.0,   104,    1e-05,      ))

data = {
            "bk" : data_bk,
            "k2" : data_k2,
            "im" : data_im,
}

datalist = []
for exp in data:
    for id,exp_id,time,Ei,angle,bkgd in data[exp]:
        datalist.append([exp,id,exp_id,Ei,angle,time,bkgd])

# Create pandas dataframe with all data
df = pd.DataFrame(datalist,columns=["exp","id","exp_id","E_MeV","angle_deg","time_ms","background"])
df["cos_angle"] = np.cos(df["angle_deg"]*np.pi/180)
df["time_s"] = df["time_ms"]/1000

df_bk = df[df["exp"]=="bk"]
df_k2 = df[df["exp"]=="k2"]
df_im = df[df["exp"]=="im"]
df_ls = pd.DataFrame()

def get_df(experiment) -> pd.DataFrame:
    if experiment==None:
        return df
    elif experiment=="bk":
        return df_bk
    elif experiment=="k2":
        return df_k2
    elif experiment=="im":
        return df_im
    elif experiment=="ls":
        return df_ls
    else:
        raise Exception("Wrong experiment name")




def print_dataframe():
    """Prints the dataframe"""
    print("\n\n"+"="*50)
    print("Printing Dataframe:")
    print("="*50+"\n\n")
    print(df)

def print_data():
    """Print all the raw data"""
    print("\n\n"+"="*50)
    print("Printing Raw Data:")
    print("="*50+"\n\n")
    for exp in data.keys():
        print("****************")
        print("Experiment " + exp)
        print("****************")
        print("ID   Time[ms]    Energy[MeV]       Angle[°]")
        for id,expid,time,Ei,angle,bkgd in data[exp]:
            print(f"{id}\t{expid:>2d}\t{time:>5d}\t\t{Ei:>4}\t\t{angle:>3}\t\t{bkgd:>5}")
            

if __name__ == "__main__":
 
    print_data()
    print_dataframe()