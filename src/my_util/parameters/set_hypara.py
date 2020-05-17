""""this is setting hyper parameter for segnet with kmeans."""
import os
import sys
import my_util.parameters.config as config


def get_numdata(animal):
    """Define max number by animal"""
    if animal == "cel":
        num_data = 300
    elif animal == "bird":
        num_data = 203
    return num_data

def hypara_check(start, end, num_data):
    """true:ok, false:out"""
    ans = True
    if start > end:
        ans = False
        print("####### INVERSE start <=> end ??")
    elif (start > num_data) or (end > num_data):
        ans = False
        print("####### PLEASE INPUT NUMBER:(start,end) UNDER " + str(num_data))
    return ans

def fit_name(args):
    """this is module for fit parameter name to legacy code"""
    args.train_epoch= args.epoch
    args.Tau        = args.tau
    args.time_dim   = args.time
    args.crossonly  = args.myloss
    args.sec_argmax = args.secmax
    args.network    = args.net
    args.GPU_ID     = args.gpu
    return args

def add_params(args):
    """add params from config, get_para()"""
    args.lat, args.lon, args.result_dir = get_para(args)

    args.mod_dim1       = config.mod_dim1
    args.mod_dim2       = config.mod_dim2
    args.MIN_LABEL_NUM  = config.MIN_LABEL_NUM
    args.START_ID, args.END_ID = args.start, args.end
    args.window = config.window
    return args

def chk(args):
    if hypara_check(args.START_ID, args.END_ID, get_numdata(args.animal)):
        return
    sys.exit()

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
    result_dir = args.d + "_" + args.animal + "_xy"
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
        result_dir += "_myloss_a" + str(args.alpha) + "_p" + str(args.lambda_p) + "_tau" + str(args.Tau)
    if not os.path.isdir("./result/" + result_dir):
        try:
            os.mkdir("./result/" + result_dir)
        except:
            current_path = os.getcwd()
            modified_path = "\"" + current_path + "\""
            os.mkdir(modified_path + "/" + result_dir)

    print("Save figure, " + result_dir)
    return lat, lon, result_dir


def main(args):
    args = fit_name(args)
    args = add_params(args)
    chk(args)
    return args
