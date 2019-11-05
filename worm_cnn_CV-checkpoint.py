#!/usr/bin/env python
# coding: utf-8

# pip
#pip install keras,xlrd,pandas,pydot,sklearn,matplotlib

# import
import tensorflow as tf
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
from numpy.random import * 
from sklearn import metrics
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import os
#os.environ['CUDA_VISIBLE_DEVICES'] ="1"
import sys
from worm_util import get_x
from worm_util import get_model
from worm_util import my_callback as cb
from worm_util import get_batch
import datetime
import os
from worm_util import my_plt
from sklearn.model_selection import KFold
from multiprocessing import Process,Pool
import multiprocessing

#Hyper
st,fin=10,310
n_splits = 10
# # cnn model : Hyper Param
epochs = 20
kernel = 3
#model_type = "cnn"
#model_type = "hmm"
model_type = "lstm"


# # data read
# trajectory(x,y)
def get_(x,y):
    x_ = pd.read_csv(x)
    y_ = pd.read_csv(y)
    df = pd.concat([x_.iloc[:,2:],y_.iloc[:,2:]],axis=1)
    return df
xy_ash = get_("data_prepro/pre_ASH_x.csv","data_prepro/pre_ASH_y.csv")
xy_awb = get_("data_prepro/pre_AWB_x.csv","data_prepro/pre_AWB_y.csv")
xy_ash = xy_ash.fillna(0).astype("float")
xy_awb = xy_awb.fillna(0).astype("float")

# turn
turn_ash = pd.read_csv("data_prepro/pre_ASH_turn.csv",index_col="Unnamed: 0")
turn_awb = pd.read_csv("data_prepro/pre_AWB_turn.csv",index_col="Unnamed: 0")

# data shape
#print("(t,x:y):",xy_ash.shape,xy_awb.shape) print("(t,turn):",turn_ash.shape,turn_awb.shape)

# # Resize
xy_ash = xy_ash.iloc[st:fin,:]# (time,n)
xy_ash = xy_ash.reset_index(drop=True)
xy_awb = xy_awb.iloc[st:fin,:]# (time,n)
xy_awb = xy_awb.reset_index(drop=True)
turn_ash = turn_ash.iloc[st:fin,:]# (time,n)
turn_ash = turn_ash.reset_index(drop=True)
turn_awb = turn_awb.iloc[st:fin,:]# (time,n)
turn_awb = turn_awb.reset_index(drop=True)

#make id array
#X = list(range(turn_ash.shape[1]))
X_odor_45_id = list(range(0,32))
X_odor_90_id = list(range(32,32+39))
X_odor_180_id = list(range(32+39,32+39+35))
X_odor_no_id = list(range(32+39+35,32+39+35+41))

y = turn_ash.T
#from sklearn.model_selection import KFold
kf = KFold(n_splits=n_splits)

from worm_util import my_split
list_odor = ["45","90","180","no"]
list_odor_data_id = [X_odor_45_id,X_odor_90_id,X_odor_180_id,X_odor_no_id]
dir_train_id,dir_test_id = my_split.my_split(list_odor,list_odor_data_id,n_splits)

def multi_dnn(k_fold):
#def multi_dnn(k_fold,X_train_id, X_test_id):
    print(k_fold)
    X_train_id, X_test_id = dir_train_id[k_fold], dir_test_id[k_fold]
    if k_fold<4:
        os.environ['CUDA_VISIBLE_DEVICES'] =str(k_fold)
    else:
        print("Use ",k_fold," GPU!!")
        return
        #sys.exit()
    
    y_train = pd.DataFrame(None)
    y_test = pd.DataFrame(None)
    for i in X_train_id:
        y_train = pd.concat([y_train,y.iloc[i,:]],axis=1)
    for i in X_test_id:
        y_test = pd.concat([y_test,y.iloc[i,:]],axis=1)
    
    #from worm_util import get_x
    X_train,X_test = get_x.get_x(xy_ash,X_train_id,X_test_id)
    y_train = y_train.values.T
    y_test = y_test.values.T
    
    Y_train = y_train
    Y_test = y_test
    Y_train = Y_train[:,:,np.newaxis]
    Y_test = Y_test[:,:,np.newaxis]
    #y_train_onehot = to_categorical(y_train)
    #y_test_onehot = to_categorical(y_test)
    #Y_train = y_train_onehot#Y_test = y_test_onehot
    print(Y_train.shape,Y_test.shape)
    
    # ## Plot
    plt.figure(figsize=(5,5))
    for i in range(X_train.shape[0]):
        plt.scatter(X_train[i,:,0],X_train[i,:,1],s=3,alpha=0.5)
    plt.savefig("./fig/train_trajectory_"+str(k_fold)+".png")

    plt.figure(figsize=(5,5))
    for i in range(X_test.shape[0]):
        plt.scatter(X_test[i,:,0],X_test[i,:,1],s=3,alpha=0.5)
    plt.savefig("./fig/test_trajectory_"+str(k_fold)+".png")

    # turn
    x=np.linspace(0,300,300)
    y_=np.linspace(0,0,300)
    #for i in range(y_test.shape[0]):
    plt.figure(figsize=(5,5))
    plt.fill_between(x,y_,y_train.mean(axis=0),alpha=0.75,label="train")
    plt.fill_between(x,y_,y_test.mean(axis=0),alpha=0.75,label="test")
    plt.xlabel("time")
    plt.legend()
    plt.savefig("./fig/turn_train_test_"+str(k_fold)+".png")

    # # Norm
    #X_train
    X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())
    X_test = (X_test-X_test.min())/(X_test.max()-X_test.min())

    # model
    #from worm_util import get_model
    if model_type=="cnn":
        model = get_model.get_model(X_train.shape[1],kernel)
    elif model_type=="lstm":
        model = get_model.get_lstm(X_train.shape[1])
    elif model_type=="hmm":
        pass

    # Callback for f1
    #from worm_util import my_callback as cb
    # compile
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=1/1000),
                  metrics=['accuracy',cb.true_positives, cb.possible_positives, cb.predicted_positives])
    model.summary()


    #from worm_util import get_batch
    # # train on batch
    met_=pd.DataFrame(None)
    for e in range(epochs):
        # train
        for x_batch, y_batch, i in get_batch.get_batch(X_train, Y_train, shuffle=True):
            if i%20==0:
                print("batch "+str(k_fold)+":",str(i),"\r")
            x_batch = x_batch[np.newaxis,:,:]
            y_batch = y_batch[np.newaxis,:,:]

            # plot sample
            if 0:
            #if e%5==0 and i==0:
                plt.scatter(x=x_batch[0,:,0],y=x_batch[0,:,1],
                            c=y_batch[0,:,1],cmap=cm.jet)
                ax = plt.colorbar()
                plt.show()
            #print(x_batch.shape,y_batch.shape)
            model.train_on_batch(x_batch,y_batch)

        # evaluate model with predicting 
        ls_metrics =[]
        for x_batch_test, y_batch_test, i in get_batch.get_batch(X_test, Y_test, shuffle=True):
            ls_metrics.append(model.test_on_batch(x_batch_test[np.newaxis,:,:],y_batch_test[np.newaxis,:,:]))

        df_met = pd.DataFrame(ls_metrics)
        df_tmp = df_met.mean()
        df_tmp.index = ["loss","acc","TP", "TP+FN", "TP+FP"]
        p = df_tmp["TP"]/df_tmp["TP+FP"]
        r = df_tmp["TP"]/df_tmp["TP+FN"]
        f = 2*p*r/(p+r)
        met_ = pd.concat([met_,df_tmp],axis=1)
        print(df_met.head())
        print(df_tmp.head())

        print("epoch "+str(k_fold)+":",e+1,"/",epochs,", metrics:[loss,acc]=",df_tmp[:2].values)
        print("             metrics:[Precision,Recall,F1]=(","{:.3f}".format(p),"{:.3f}".format(r),"{:.3f}".format(f),")")
    # dir
    #import datetime
    now_ = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_dir = "./fig/model_"+now_+"_"+model_type+"_"+str(k_fold)
    os.mkdir(result_dir)

    # predict
    #from worm_util import my_plt
    #plot
    my_plt.plt_metrics(met_,result_dir,epochs)
    df_pred = my_plt.plot_predict(model,X_test,Y_test,result_dir)
    my_plt.mean_plot_predict(turn_ash,df_pred,result_dir)
#"""
#multi
#from multiprocessing import Process
print('parents process id:%s'%(os.getpid()))
print('cpu count:%s'%multiprocessing.cpu_count())
#p = Pool(int(16))
p = Pool(int(multiprocessing.cpu_count()))
print("start process")

#p = Process(target=multi_dnn(1,dir_train_id[1], dir_test_id[1]))
#p.start()
p.map(multi_dnn,(0,1))
#multi_dnn(2,dir_train_id[2], dir_test_id[2])
p.close()