from ast import arg
import re
import torch
import pyrtklib as prl
import rtk_util as util
import json
import sys
import numpy as np
import pandas as pd
import pymap3d as p3d
from model import VGPNet
from torch.nn import MSELoss
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms
import argparse
import datetime

DEVICE = "cuda"

# use argparse to get config
parser = argparse.ArgumentParser(description="Train the model")
parser.add_argument(
    "--config_file",
    type=str,
    default="config/train_img/rw4_train.json",
    help="Path to the config file",
)
parser.add_argument(
    "--bool_gnss", action="store_true", help="Boolean flag for GNSS usage"
)
parser.add_argument(
    "--bool_fisheye", action="store_true", help="Boolean flag for fisheye usage"
)
parser.add_argument(
    "--bool_surrounding", action="store_true", help="Boolean flag for surrounding usage"
)
parser.add_argument(
    "--bool_mask", action="store_true", help="Boolean flag for mask usage"
)
parser.add_argument(
    "--bool_ccffm", action="store_true", help="Boolean flag for CCFFM usage"
)
parser.add_argument(
    "--resume", type=int, default=0, help="Resume training from this epoch"
)
parser.add_argument("--epoch", type=int, default=200, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
parser.add_argument(
    "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
)
args = parser.parse_args()
print("Arguments:", args)
config_file = args.config_file
print("Config file:", config_file)


dataset_name = config_file.split("/")[-1].split(".json")[0].split("_")[0]
print(dataset_name)
# 后缀
if dataset_name in [
    "klt1",
    "klt2",
    "klt3",
    "rw4",
    "rw5",
    "rw6",
    "rw1",
    "rw2",
    "rw3",
]:
    ends = "png"
# elif dataset_name in ["rw1", "rw2", "rw3"]:
#     ends = "jpg"
print("ends: ", ends)
# urban_deep or klt3_train
# config = "config/image/klt3_train.json"

with open(config_file) as f:
    conf = json.load(f)

mode = conf["mode"]
if mode not in ["train", "predict"]:
    raise RuntimeError("%s is not a valid option" % mode)

result = config_file.split("/")[-1].split(".json")[0]


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

print("prefix: ", prefix)

result_path = "result/train_img/" + result + prefix
print(result_path)
model_dir = conf["model"] + prefix
print(model_dir)

os.makedirs(result_path, exist_ok=True)  # dir for result
os.makedirs(model_dir, exist_ok=True)  # dir for model


obs, nav, sta = util.read_obs(conf["obs"], conf["eph"])
prl.sortobs(obs)

obss = util.split_obs(obs)
print(f"obs num: {len(obss)}")

tmp = []

#! diff about the dataset
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
        # elif dataset_name == "rw1" or dataset_name == "rw2":
        #     gt = pd.read_csv(conf["gt"], header=None)
        #     gt[6] = gt[6] + 18  # leap seconds
        # read ros time ref
        time_ref = pd.read_csv(conf["ref"], header=0)
    elif dataset_name == "deep" or dataset_name == "medium" or dataset_name == "harsh":
        gt = pd.read_csv(conf["gt"], skiprows=2, header=None, sep=" +", engine="python")
        gt[0] = gt[0] + 18  # leap seconds
    elif dataset_name in ["rw1", "rw2", "rw3", "rw4", "rw5", "rw6"]:
        gt = pd.read_csv(
            conf["gt"],
            skiprows=1,
            header=None,
        )
        gt[1] = gt[1] + 18  # leap seconds

    gts = []

# load image infos
img_fishs = []
if conf.get("img", None) and (bool_fisheye):
    if dataset_name in ["rw4", "rw1"]:
        img_fish = os.listdir(conf["img"])
    # elif dataset_name in ["rw1", "klt3"]:
    #     img_fish = os.listdir(os.path.join(conf["img"], "fisheye"))

    if img_fish is not None:
        for i in img_fish:
            if i.endswith(".png") or i.endswith(".jpg"):
                img_fishs.append(float(i[:-4]))
    img_fishs.sort()
    img_fish_row_path_list = []

# filter and normalize
gather_data = []
for o in obss:
    t = o.data[0].time
    t = t.time + t.sec  #! t is gps time
    if t > conf["start_time"] and (
        conf["end_time"] == -1 and 1 or t < conf["end_time"]
    ):
        tmp.append(o)
        if conf.get("gt", None):
            if dataset_name in [
                "klt1",
                "klt2",
                "klt3",
            ]:
                gt_row = gt.loc[(gt[0] - t).abs().argmin()]
                # print("gt_row time:", gt_row[0], "t:", t)
                gts.append(
                    [
                        gt_row[3] + gt_row[4] / 60 + gt_row[5] / 3600,
                        gt_row[6] + gt_row[7] / 60 + gt_row[8] / 3600,
                        gt_row[9],
                    ]
                )
            elif dataset_name in ["rw4", "rw5", "rw6", "rw1", "rw2", "rw3"]:
                gt_row = gt.loc[(gt[1] - t).abs().argmin()]
                gt_time = gt_row[1]
                # print("gt_row time:", gt_row[1], "t:", t)
                gts.append(
                    [
                        gt_row[3],
                        gt_row[4],
                        gt_row[5],
                    ]
                )
                img_fish_row = min(img_fishs, key=lambda x: abs(x - gt_time - 18))
                img_fish_row_path = os.path.join(
                    conf["img"], f"{img_fish_row:.6f}.{ends}"
                )
                img_fish_row_path_list.append(img_fish_row_path)
            # elif dataset_name in ["rw1", "rw2", "rw3"]:
            #     gt_row = gt.loc[(gt[6] - t).abs().argmin()]
            #     # print("gt_row time:", gt_row[6], "t:", t)
            #     gts.append(
            #         [
            #             gt_row[3] + gt_row[4] / 60 + gt_row[5] / 3600,
            #             gt_row[6] + gt_row[7] / 60 + gt_row[8] / 3600,
            #             gt_row[9],
            #         ]
            #     )

        # if conf.get("img", None) and bool_fisheye:
        #     if dataset_name in ["rw1", "rw2", "rw3", "klt1", "klt2", "klt3"]:
        #         # 在time_ref根据t找到对应的t1
        #         time_tmp = (time_ref["time_ref"] - (gt_time - 18)).abs().idxmin()
        #         ros_img_time = time_ref.loc[time_tmp, "ros_time"]
        #         ref_gt_time = time_ref.loc[time_tmp, "time_ref"]
        #     else:
        #         ros_img_time = gt_time - 18
        #         ref_gt_time = gt_time - 18

        #     print("imt-time: ", ros_img_time)
        #     img_fish_row = min(img_fishs, key=lambda x: abs(x - ros_img_time))
        #     img_row_f = os.path.join(
        #         conf["img"], "fisheye", f"{img_fish_row:.6f}.{ends}"
        #     )
        #     img_fishes.append(img_row_f)

        ret = util.get_ls_pnt_pos(o, nav)  # 计算最小二乘解
        if not ret["status"]:
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
        gather_data.append(np.hstack([SNR.reshape(-1, 1), azel[:, 1:], resd]))

print(f"gather data num: {len(gather_data)}")
norm_data = np.vstack(gather_data)
imean = norm_data.mean(axis=0)
istd = norm_data.std(axis=0)

print(f"preprocess done, mean:{imean}, std:{istd}")


net = VGPNet(
    torch.tensor(imean, dtype=torch.float32),
    torch.tensor(istd, dtype=torch.float32),
    bool_gnss,
    bool_fisheye,
    # bool_surrounding,
    # bool_mask,
    bool_ccffm,
)
net.double()
net = net.to(DEVICE)

resume_ep = args.resume
model_path = model_dir + f"/image_ep{resume_ep}.pth"
if os.path.exists(model_path) and resume_ep > 0:
    net.load_state_dict(torch.load(model_path))
    print(f"load from {model_path}.")
else:
    print(
        f"model path {model_path} not found or not config resume, starting from scratch."
    )
    resume_ep = 0


obss = tmp

pos_errs = []


opt = torch.optim.AdamW(
    net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
)
sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

epoch = args.epoch

loss_log_path = result_path + f"/loss_{resume_ep}.csv"
if os.path.exists(loss_log_path):
    vis_loss = list(np.loadtxt(loss_log_path).reshape(-1))
else:
    vis_loss = []


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

batch_size = 128

for k in range(resume_ep, epoch):
    net.train()
    loss = 0  # 计算每个epoch的loss
    epoch_loss = 0
    indices = list(range(len(obss)))
    # random.shuffle(indices)  # 随机打乱数据
    with tqdm(indices, desc=f"Epoch {k+1}", ncols=80) as t:
        batch_loss = 0  # 计算每个batch的loss
        for i in t:  # 遍历所有的obss
            o = obss[i]
            if conf.get("gt", None):
                gt_row = gts[i]  # lat lon hgt
                gt_before = gts[i - 1] if i > 0 else gts[i]
            ret = util.get_ls_pnt_pos(o, nav)  # 计算最小二乘解，得到初步的位置解
            if not ret["status"]:
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

            if conf.get("img", None) and bool_fisheye:
                img_row_f = img_fish_row_path_list[i]
                img_f = (
                    Image.open(img_row_f).resize((224, 224)) if bool_fisheye else None
                )
                img_f = (
                    f_preprocess(img_f).unsqueeze(0).to(DEVICE, dtype=torch.float64)
                    if bool_fisheye
                    else None
                )

            in_data = torch.tensor(
                np.hstack([SNR.reshape(-1, 1), azel[:, 1:], resd]), dtype=torch.float32
            ).to(DEVICE)
            if bool_fisheye:
                predict = net(in_data, img_f)  # [weight, bias]

            weight = predict[0]
            bias = predict[1]

            # 用网络预测的权重和偏置来计算最小二乘解
            ret = util.get_ls_pnt_pos_torch(
                o, nav, torch.diag(weight), bias.reshape(-1, 1), p_init=ret["pos"]
            )
            result_wls = ret["pos"][:3]
            enu = p3d.ecef2enu(*result_wls, gt_row[0], gt_row[1], gt_row[2])
            loss = torch.norm(torch.hstack(enu[:3]))  # 每个sample的loss
            batch_loss += loss
            if (i + 1) % batch_size == 0:
                # batch_loss = torch.stack(batch_loss).mean()
                opt.zero_grad()
                batch_loss.backward()
                opt.step()
                epoch_loss += batch_loss
                t.set_postfix({"batch_loss": batch_loss.item()})
                batch_loss = 0
            elif (i + 1) == len(obss):
                # batch_loss = torch.stack(batch_loss).mean()
                opt.zero_grad()
                batch_loss.backward()
                opt.step()
                epoch_loss += batch_loss
                t.set_postfix({"batch_loss": batch_loss.item()})
                batch_loss = 0

        print("mean loss: ", epoch_loss.item() / len(obss), "epoch loss: ", epoch_loss.item())
        vis_loss.append(epoch_loss.item() / len(obss))
    if k % 10 == 0 and k > 0:
        torch.save(
            net.state_dict(),
            os.path.join(model_dir, f"image_ep{k}.pth"),
        )
        vis_loss_300 = np.array(vis_loss)
        plt.plot(vis_loss)
        plt.savefig(result_path + f"/loss_{k}.png")
        np.savetxt(result_path + f"/loss_{k}.csv", vis_loss_300.reshape(-1, 1))
    sch.step()
torch.save(net.state_dict(), os.path.join(model_dir, "image_3d.pth"))
vis_loss = np.array(vis_loss)
plt.plot(vis_loss)
plt.savefig(result_path + f"/loss_{epoch}.png")
np.savetxt(result_path + f"/loss_{epoch}.csv", vis_loss.reshape(-1, 1))
