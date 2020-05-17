import argparse
import my_util.parameters.config as config

PARSER = argparse.ArgumentParser(description="this is Segnet with Kmeans")

# data
PARSER.add_argument("--animal", choices=["cel", "bird"], default=config.animal)

# train
PARSER.add_argument("-e", "--epoch", type=int, default=config.epoch, help="BATCH: num of trainging with one trajectory")
PARSER.add_argument("--epoch_all", type=int, default=config.epoch_all, help="EPOCH: num of training with all trajectories")

# hypara of custom loss
PARSER.add_argument("--alpha", type=float, default=config.alpha, help="to be bigger, enlarge d. To be smaller, ensmaller d")
PARSER.add_argument("--lambda_p", type=float, default=config.lambda_p)
PARSER.add_argument("--tau", type=float, default=config.tau, help="to be smaller, enlarge w.")

# on/off custom module
PARSER.add_argument("--time", action="store_true")
PARSER.add_argument("--myloss", action="store_false")
PARSER.add_argument("--secmax", action="store_true")

# set traj id
PARSER.add_argument("--start", type=int, default=config.start, help="this is start_id")
PARSER.add_argument("--end", type=int, default=config.end, help="this is end_id")

# select network
PARSER.add_argument("--net", type=str, default=config.net)

# train hyparam
PARSER.add_argument("--lr", type=float, default=config.lr, help="learing rate")
PARSER.add_argument("--momentum", type=float, default=config.momentum, help="learing rate")

PARSER.add_argument("-d", default=config.dir_name, help="header of result dir name, result/d~")
PARSER.add_argument("-k", type=int, default=config.K, help="set k of kmeans for compared method")
PARSER.add_argument("--plot_mode", action="store_true")
PARSER.add_argument("--label_dir", default=config.label_dir, help="set dir name of load label")

# gpu
PARSER.add_argument("--gpu", type=int, default=config.GPU_ID)

args = PARSER.parse_args()
