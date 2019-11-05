""""this is setting hyper parameter for segnet with kmeans."""
import os
import sys

# time_dim       : True(add time dim)
# crossonly      : True(only crossentropy without my loss)
# sec_argmax     : set second of argmax
# alpha          : hypara
# lambda_penalty : loss + lambda_p * penalty
# Tau            : samller tau more dt ,exp^(dt/T) in penalty()

# trajectory     : dir{df=(t,(x,y)),df,...}
# traj           : np.array=(t,n,(x,y,t))

class Args(object):
    """this is hyper parameter setting module."""
    train_epoch = 2 ** 6
    mod_dim1 = 64 #
    mod_dim2 = 32 # 32
    GPU_ID = 1

    MIN_LABEL_NUM = 4  # if the label number small than it, break loop

    animal_ = ["cel", "bird"]
    animal = animal_[1]
    lat, lon, result_dir = "", "", ""

    time_dim = True
    crossonly = False
    sec_argmax = False

    alpha = 10
    lambda_p = 0.01
    window = 20
    Tau = 100

    START_ID = 0
    END_ID = 5

    network = "segnet" # "unet"

def get_numdata(animal):
    """Define max number by animal"""
    if animal == "cel":
        num_data = 300
    elif animal == "bird":
        num_data = 30
    return num_data

def hypara_check(start, end, num_data):
    """true:ok, false:out"""
    ans = True
    if start > end:
        ans = False
        print("####### INVERSE start <=> end ??")
    elif (start > num_data) or (end > num_data):
        ans = False
        print("####### PLEASE INPUT NUMBER:(start,end) UNDER "+str(num_data))
    return ans

def set_hypara(args, tmp):
    """this is module for setting parameters from tmp."""
    args.animal, args.train_epoch, args.alpha = tmp.animal, tmp.epoch, tmp.alpha
    args.lambda_p, args.Tau = tmp.lambda_p, tmp.tau
    args.time_dim, args.crossonly, args.sec_argmax = tmp.time, tmp.myloss, tmp.secmax
    args.network = tmp.net
    args.lat, args.lon, args.result_dir = get_para(args)

    if hypara_check(int(tmp.start), int(tmp.end), get_numdata(args.animal)):
        args.START_ID, args.END_ID = int(tmp.start), int(tmp.end)
    else:
        sys.exit()
    return args

def get_para(args):
    """this is module for getting dir, lat, lon."""

    # set lattitude, longitude
    if args.animal == "bird":
        lat, lon = "latitude", "longitude"
    elif args.animal == "car":
        lat, lon = "lat", "lon"
    elif args.animal == "cel":
        lat, lon = "x", "y"

    # result directory
    result_dir = "result_slide/"+args.animal+"_xy"
    if args.time_dim:
        result_dir += "t"
    else:
        result_dir += "window"
    if args.sec_argmax:
        result_dir += "_sec_arg"
    else:
        result_dir += "_arg"
    if args.crossonly:
        result_dir += "_nonmyloss"
    else:
        result_dir += "_myloss_a"+str(args.alpha)+"_p"+str(args.lambda_p)+"_tau"+str(args.Tau)
    if not os.path.isdir("./"+result_dir):
        try:
            os.mkdir("./"+result_dir)
        except:
            current_path = os.getcwd()
            #modified_path = current_path.replace(" ", "/ ")
            modified_path = "\""+current_path+"\""
            os.mkdir(modified_path+"/"+result_dir)

    print("Save figure, " + result_dir)
    return lat, lon, result_dir
