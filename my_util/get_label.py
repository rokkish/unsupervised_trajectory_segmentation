import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



"""def plot_Freqofnumlabel_Loss(df_freq,loss):
    #get freq
    freq = {}
    for label_i in df_freq:
        freq[label_i] = freq.get(label_i, 0) + 1
    #print("{Label_i: Freq}=",freq)
    
    #view
    fig = plt.figure(figsize=(15,5))
    plt.style.use('dark_background')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(df_freq,bins=30)
    ax1.set_ylabel("freq")
    ax1.set_xlabel("no.label")
    ax1.set_title("freq of label")
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(loss)
    ax2.set_ylabel("loss")
    ax2.set_xlabel("epochs")
    ax2.set_title("loss")
    plt.show()
    return freq"""
    
"""def hash_latlon_into_label(a,b,c):
    df_label = a*b*c+a*0.5+0.4*b+0.2*c+1
    return df_label"""

"""def get_label(show,time_dim):
    df_ret = pd.DataFrame(show[:,0,:])

    #重複消去 lat,lon
    df_label = df_ret[~df_ret.duplicated()]
    df_label = df_label.reset_index()
    #a,b,c=df_label.iloc[:,0],df_label.iloc[:,1],df_label.iloc[:,2]
    #df_label = hash_latlon_into_label(a,b,c)
    print(df_ret.shape)
    print(df_ret)
    print(df_label.shape)
    print(df_label)
    
    #hash用dir作成={hash値:label}
    dir_label={}
    for i,x in enumerate(df_label.loc["index"]):
        print(i,x)
        dir_label[x]=i
        
    #len==time,hash作成
    #a_,b_,c_=df_ret.iloc[:,0],df_ret.iloc[:,1],df_ret.iloc[:,2]
    #df_hash = hash_latlon_into_label(a_,b_,c_)
    df_hash
    #label:a
    a=[]
    for h in df_hash:
        a.append(dir_label[h])
        
    #concat
    df_label_ = pd.Series(a)
    df_ret = pd.conc"""at([df_ret,df_label_],axis=1)
    if time_dim:
        df_ret.columns=["lat","lon","t","label"]
    else:
        df_ret.columns=["lat","lon","N","label"]
    
    return df_ret

def plot_seg_result(df_ret,df_traj_i,Id_trajectory,lat,lon,result_dir):
    plt.figure(figsize=(15,10))
    plt.style.use('dark_background')
    
    ls_label = sorted(pd.unique(df_ret["label"]))
    n_split = len(ls_label)+4
    for n,j in enumerate(ls_label):
        Itraj_Jlabel = df_traj_i.loc[df_ret["label"]==j]
        plt.subplot(4,int(n_split/4),n+1)
        plt.plot(df_traj_i[lat].values,df_traj_i[lon].values,color="white",alpha=0.4)
        plt.scatter(Itraj_Jlabel[lat],Itraj_Jlabel[lon],c="r",s=8)
        plt.text(0,0,Itraj_Jlabel.iloc[0,0])
        plt.title(str(j))
    plt.show()
    
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.scatter(df_traj_i[lat],df_traj_i[lon],c=df_ret["label"].iloc[:df_traj_i.shape[0]],cmap="tab20",s=16)
    plt.title("Segmentation")
    ax = plt.colorbar()
    plt.subplot(1,2,2)
    plt.plot(df_traj_i[lat].values,df_traj_i[lon].values,color="white",alpha=0.2)
    plt.scatter(df_traj_i[lat],df_traj_i[lon],c=np.linspace(0,df_traj_i.shape[0],df_traj_i.shape[0]),cmap="magma",s=16)
    plt.title("Time color")
    ax = plt.colorbar()
    plt.savefig("./"+result_dir+"/segmentation_"+"{:0=3}".format(Id_trajectory)+".png")
    plt.show()
    """
    label_max = max(df_ret["label"].values)
    params = np.random.rand(label_max, 3)
    df_traj_i = pd.concat([df_traj_i,df_ret["label"]],axis=1)
    plt.subplot(1,n_split,3)
    for i,param in enumerate(params):
        plt.plot(df_traj_i.loc[df_traj_i["label"]==i][lat].values,df_traj_i.loc[df_traj_i["label"]==i][lon].values,color=param,label=str(i),marker="o")
    plt.title("Seg color by ")
    plt.legend()
    
    plt.savefig("./"+result_dir+"/segmentation_"+"{:0=3}".format(Id_trajectory)+".png")
    plt.show()
    #"""