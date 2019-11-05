import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from .plt_kmkm import plt_km

def plt_kmeans(pred,df):
    plt.figure(figsize=(5,5))
    plt.scatter(df.iloc[:,0],df.iloc[:,1],c=pred,cmap="tab20",s=5)
    plt.title("Kmeans with same k color")
    ax=plt.colorbar()
    plt.show()
    
def do_kmeans(k,df,time,pred_):
    # k    : set k of k-means
    # df   : (lat,lon)
    # time : input for k-means (lat,lon,time)
    # pred_: prepro label of run() with k-means
    
    if k==1:
        pass
    else:
        df_time = pd.DataFrame(time)
        df_time = df_time.reset_index(drop=True)
        tmp=pd.concat([df,df_time],axis=1)
        tmp_ = tmp.copy()
        tmp = tmp.dropna()
        tmp = tmp.reset_index(drop=True)
        #print(tmp.shape,tmp.values.shape)
        pred = KMeans(n_clusters=k).fit_predict(tmp.values)
        #print(pred.shape)
        #plt_kmeans(pred,tmp)
        plt_km(pred,tmp,pred_,tmp_)
        
        plt.figure(figsize=(5,5))
        plt.hist(pred,bins=30)
        plt.ylabel("freq")
        plt.xlabel("no.label")
        plt.title("freq of label only kmeans with same k")
        plt.show()
    