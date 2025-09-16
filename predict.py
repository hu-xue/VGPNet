from ast import arg
from sympy import false
import torch
import pyrtklib as prl
import rtk_util as util
import json
import sys
import numpy as np
import pandas as pd
import pymap3d as p3d
from model import VGPNet
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms
import argparse
import datetime

DEVICE = "cuda"

# use argparse to get config
parser = argparse.ArgumentParser(description="Train the model")
parser.add_argument(
    "--config_file",
    type=str,
    default="config/train_img/klt3_train.json",
    help="Path to the config file",
)
parser.add_argument(
    "--bool_gnss",
    action="store_true",
    help="Boolean flag for GNSS usage",
    default=False,
)
parser.add_argument(
    "--bool_fisheye",
    action="store_true",
    help="Boolean flag for fisheye usage",
    default=False,
)
parser.add_argument(
    "--bool_ccffm",
    action="store_true",
    help="Boolean flag for CCFFM usage",
    default=False,
)
parser.add_argument(
    "--model_name",
    type=str,
    help="Model name to load",
    default="image_3d",
    required=True,
)
args = parser.parse_args()
print("Arguments:", args)
config_file = args.config_file
print("Config file:", config_file)

dataset_name = config_file.split("/")[-1].split(".json")[0].split("_")[0]
print(dataset_name)
# 后缀
if "klt" in dataset_name:
    ends = "png"
elif "rw" in dataset_name:
    ends = "png"
# klt1_predict klt2_predict urban_medium urban_harsh
# config = "config/image/urban_harsh.json"

with open(config_file) as f:
    conf = json.load(f)

mode = conf["mode"]
if mode not in ["train", "predict"]:
    raise RuntimeError("%s is not a valid option" % mode)

result = config_file.split("/")[-1].split(".json")[0]  # klt1_predict klt2_predict
result = result.split("_")[0]  # klt1 klt2

bool_gnss = args.bool_gnss
bool_fisheye = args.bool_fisheye
bool_ccffm = args.bool_ccffm

prefix = "_"
if bool_gnss:
    prefix += "g"
if bool_fisheye:
    prefix += "f"
prefix += "_ccffm" if bool_ccffm else ""

now = datetime.datetime.now()
prefix += f"_{now.strftime('%Y%m%d_%H%M%S')}"

print(prefix)
result_path = "result/pred_major/" + result + prefix
print(result_path)

os.makedirs(result_path, exist_ok=True)
os.makedirs(result_path + "/bw_VGL", exist_ok=True)


net = VGPNet(
    bool_gnss=bool_gnss,
    bool_fisheye=bool_fisheye,
    bool_ccffm=bool_ccffm,
)
net.double()
net.load_state_dict(
    torch.load(f"model/image_rw1_gf_ccffm_20250911_163814/{args.model_name}.pth")
)
net = net.to(DEVICE)


obs, nav, sta = util.read_obs(conf["obs"], conf["eph"])
prl.sortobs(obs)
prcopt = prl.prcopt_default
obss = util.split_obs(obs)

tmp = []

if conf.get("gt", None):
    gt = None
    if dataset_name == "klt3" or dataset_name == "klt1" or dataset_name == "klt2":
        gt = pd.read_csv(
            conf["gt"],
            skiprows=30,
            header=None,
            sep=" +",
            skipfooter=4,
            engine="python",
        )
        gt[0] = gt[0] + 18  # leap seconds
        time_ref = pd.read_csv(conf["ref"], header=0)
    elif "rw" in dataset_name:
        gt = pd.read_csv(conf["gt"], header=None, skiprows=1)
        gt[1] = gt[1] + 18
    gts = []


# load image infos
img_fishs = []
if conf.get("img", None) and (args.bool_fisheye):
    if "rw" in dataset_name:
        img_fish = os.listdir(conf["img"])
    else:
        img_fish = os.listdir(os.path.join(conf["img"], "fisheye"))
    for i in img_fish:
        if i.endswith(".png") or i.endswith(".jpg"):
            img_fishs.append(float(i[:-4]))
    img_fishs.sort()
    img_fishes = []


for o in obss:
    t = o.data[0].time
    t = t.time + t.sec
    if t > conf["start_time"] and (
        conf["end_time"] == -1 and 1 or t < conf["end_time"]
    ):
        tmp.append(o)
        if conf.get("gt", None):
            if "rw" in dataset_name:
                gt_row = gt.loc[(gt[1] - t).abs().argmin()]
                gt_time = gt_row[1]
                gts.append([gt_row[3], gt_row[4], gt_row[5]])
            elif "klt" in dataset_name:
                gt_row = gt.loc[(gt[6] - t).abs().argmin()]
                gts.append(
                    [
                        gt_row[3] + gt_row[4] / 60 + gt_row[5] / 3600,
                        gt_row[6] + gt_row[7] / 60 + gt_row[8] / 3600,
                        gt_row[9],
                    ]
                )
        if conf.get("img", None) and bool_fisheye:
            img_fish_row = min(img_fishs, key=lambda x: abs(x - gt_time - 18))
            img_row_f_now = os.path.join(conf["img"], f"{img_fish_row:.6f}.{ends}")
            img_fishes.append(img_row_f_now)

f_preprocess = (
    transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=conf["f_norm_std"][0], std=conf["f_norm_std"][1]),
        ]
    )
    if bool_fisheye
    else None
)


obss = tmp
net.eval()
errors = []
gt_pos = []
TDL_bw_pos = []
ecef_pos = []
samples = 0
# gps_times = []
# img_f_nows = []
with tqdm(range(len(obss)), ncols=80) as t:
    for i in t:
        o = obss[i]
        if conf.get("gt", None):
            gt_row = gts[i]
        try:
            ret = util.get_ls_pnt_pos(o, nav)
            if not ret["status"]:
                print(ret["msg"])
                continue
        except Exception as e:
            print(e)
            continue

        rs = ret["data"]["eph"]
        dts = ret["data"]["dts"]
        sats = ret["data"]["sats"]
        exclude = ret["data"]["exclude"]
        prs = ret["data"]["prs"]
        resd = np.array(ret["data"]["residual"])
        SNR = np.array(ret["data"]["SNR"])
        azel = np.delete(
            np.array(ret["data"]["azel"]).reshape((-1, 2)), exclude, axis=0
        )
        in_data = torch.tensor(
            np.hstack([SNR.reshape(-1, 1), azel[:, 1:], resd]), dtype=torch.float64
        ).to(DEVICE)

        # load images and normalize
        img_f_now = None
        if conf.get("img", None) and (args.bool_fisheye):
            img_row_f_now = img_fishes[i]
            # img_f_nows.append(img_row_f_now)

            img_f_now = (
                Image.open(img_row_f_now).resize((224, 224)) if bool_fisheye else None
            )

            img_f_now = (
                f_preprocess(img_f_now).unsqueeze(0).to(DEVICE, dtype=torch.float64)
                if bool_fisheye
                else None
            )

        samples += in_data.shape[0]
        if bool_fisheye:
            predict = net(in_data, img_f_now)  # [weight, bias]

        weight = predict[0]
        bias = predict[1]

        sats_used = np.delete(np.array(sats), exclude, axis=0)
        snp = sats_used
        wnp = weight.detach().cpu().numpy()
        bnp = bias.detach().cpu().numpy()
        ep = pd.DataFrame(np.vstack([snp, wnp, bnp]).T)
        ep.columns = ["sat", "weight", "bias"]
        ep.to_csv(result_path + "/bw_VGL/%d.csv" % i, index=None)

        ret = util.get_ls_pnt_pos_torch(
            o, nav, torch.diag(weight), bias.reshape(-1, 1), p_init=ret["pos"]
        )
        result_wls = ret["pos"][:3].detach().cpu().numpy()
        TDL_bw_pos.append(p3d.ecef2geodetic(*result_wls))
        gt_pos.append([gt_row[0], gt_row[1], gt_row[2]])
        errors.append(p3d.geodetic2enu(*TDL_bw_pos[-1], *gt_pos[-1]))
        ecef_pos.append(ret["pos"].detach().cpu().numpy())

# result_gpstime = np.array(gps_times)
# result_fisheye = np.array(img_f_nows)
# result_gpstime.tofile("" +result + "/gps_time.txt", sep="\n")
# result_fisheye.tofile(result_path + "/fisheye.txt", sep="\n")

ecef_pos = np.array(ecef_pos)
gt_pos = np.array(gt_pos)
TDL_bw_pos = np.array(TDL_bw_pos)
errors = np.array(errors)
np.savetxt(
    result_path + "/ecef_VGP.csv",
    ecef_pos,
    delimiter=",",
    header="x,y,z,t1,t2,t3,t4",
    comments="",
)
np.savetxt(
    result_path + "/gt_VGP.csv",
    gt_pos,
    delimiter=",",
    header="lat,lon,height",
    comments="",
)
np.savetxt(
    result_path + f"/VGP{prefix}.csv",
    TDL_bw_pos,
    delimiter=",",
    header="lat,lon,height",
    comments="",
)
print(
    f"2D mean: {np.linalg.norm(errors[:,:2],axis=1).mean():.2f}, 3D mean: {np.linalg.norm(errors,axis=1).mean():.2f}"
)
print(f"Samples {samples}")
