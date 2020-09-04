import pandas as pd
import os


dataset = "camcan"
if os.path.exists("/home/parietal/"):
    results_path = "/home/parietal/hjanati/csvs/%s/" % dataset
else:
    data_path = "~/Dropbox/neuro_transport/code/mtw_experiments/meg/"
    data_path = os.path.expanduser(data_path)
    results_path = data_path + "results/%s/" % dataset


savedir_names = ["ico4_camcan_depth"]
for savedir_name in savedir_names:
    df = []
    datadir = results_path + savedir_name + "/"

    for root, dirs, files in os.walk(datadir):
        for f in files:
            if f.split('.')[-1] == "csv":
                try:
                    d = pd.read_csv(root + f, header=0, index_col=0)
                    df.append(d)
                except:
                    pass

    if len(df):
        df = pd.concat(df, ignore_index=True)
        df.to_csv("data/%s.csv" % savedir_name)
    else:
        print("No data for %s" % savedir_name)
