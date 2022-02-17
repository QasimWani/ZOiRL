import subprocess
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--mismatch", type=float, default=0.0, help="mismatch rate")
args = parser.parse_args()


def get_results(seed):
    cmd = f"python3 cocp_supply_chain.py --seed={seed} --mismatch={args.mismatch}"
    subprocess.check_output(cmd, shell=True)


for i in tqdm(range(20)):
    get_results(i)
