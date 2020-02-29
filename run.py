import subprocess
import argparse

if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description="this is shell scripts of Segnet with Kmeans")

    # data
    PARSER.add_argument("--animal", choices=["cel", "bird"], default="bird")

    # set traj id
    PARSER.add_argument("-s", type=int, default=0, help="this is start_id")

    PARSER.add_argument("-k", type=int, default=10, help="set k of kmeans for compared method")

    args = PARSER.parse_args()

    # proposed
    subprocess.call(["python", "segnet_with_kmeans.py",
        "--myloss",
        "--secmax",
        "--time",
        "--start", str(args.s),
        "--end", str(args.s + 1),
        "--net", "segnet",
        "--animal", str(args.animal),
        "--alpha", "0.1",
        "--tau", "10000",
        "--epoch", "32",
        "--epoch_all", "4",
        "-d", "run_proposed",
        "-k", str(args.k)])

    # without Penalty, secargmax
    subprocess.call(["python", "segnet_with_kmeans.py",
        "--time",
        "--start", str(args.s),
        "--end", str(args.s + 1),
        "--net", "segnet",
        "--animal", str(args.animal),
        "--alpha", "0.1",
        "--tau", "10000",
        "--epoch", "32",
        "--epoch_all", "4",
        "-d", "run_ablation",
        "-k", str(args.k)])

    # without secargmax
    subprocess.call(["python", "segnet_with_kmeans.py",
        "--myloss",
        "--time",
        "--start", str(args.s),
        "--end", str(args.s + 1),
        "--net", "segnet",
        "--animal", str(args.animal),
        "--alpha", "0.1",
        "--tau", "10000",
        "--epoch", "32",
        "--epoch_all", "4",
        "-d", "run_ablation",
        "-k", str(args.k)])

    # without Penalty
    subprocess.call(["python", "segnet_with_kmeans.py",
        "--secmax",
        "--time",
        "--start", str(args.s),
        "--end", str(args.s + 1),
        "--net", "segnet",
        "--animal", str(args.animal),
        "--alpha", "0.1",
        "--tau", "10000",
        "--epoch", "32",
        "--epoch_all", "4",
        "-d", "run_ablation",
        "-k", str(args.k)])
