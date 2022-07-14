from __future__ import division, print_function, absolute_import
import numpy as np
import scipy.io
from .utils import *


FEAT_DICT = {
    "R0": "G0_R0_FEAT.dat",
    "L0": "G0_L0_FEAT.dat",
    "R1": "G0_L1_FEAT.dat",
    "L1": "G0_R1_FEAT.dat",
}

COSTVOL_DICT = {
    "R0": "G1_R0_COSTV.dat",
    "L0": "G1_L0_COSTV.dat",
    "R1": "G1_L1_COSTV.dat",
    "L1": "G1_R1_COSTV.dat",
}


def __kernel_feat(out_dir, block, suffix):
    with open(os.path.join(out_dir, suffix), "w") as f:
        for i in range(block.item().shape[0]):
            for j in range(block.item().shape[1]):
                f.write("(%3d, %4d) : " % (i, j))
                for k in range(block.item().shape[2]):
                    t = np.uint32(block.item()[i, j, 1 - k])
                    hex_ = hex(t).split("x")[-1].zfill(8)
                    f.write("%s" % hex_)
                f.write("\n")


def __print_feat(path, out_dir, order):
    mode = load_json(os.path.join(path, "config.json"))["mode"]
    info = scipy.io.loadmat(os.path.join(path, "info.mat"))
    if order == "R0L0":
        if mode == 3:
            feat_l = info["info"]["R0R1L0L1"][0, 0]["R0L0"][0, 0]["featL"]
            feat_r = info["info"]["R0R1L0L1"][0, 0]["R0L0"][0, 0]["featR"]
        print(feat_l.item().shape)
        __kernel_feat(out_dir, feat_l, FEAT_DICT["R0"])
        __kernel_feat(out_dir, feat_r, FEAT_DICT["L0"])
    elif order == "R1L1":
        if mode == 3:
            feat_l = info["info"]["R0R1L0L1"][0, 0]["R1L1"][0, 0]["featL"]
            feat_r = info["info"]["R0R1L0L1"][0, 0]["R1L1"][0, 0]["featR"]
        print(feat_l.item().shape)
        __kernel_feat(out_dir, feat_l, FEAT_DICT["L1"])
        __kernel_feat(out_dir, feat_r, FEAT_DICT["R1"])
    return


def print_feat(path, out_dir, order):
    mode = load_json(os.path.join(path, "config.json"))["mode"]
    info = scipy.io.loadmat(os.path.join(path, "info.mat"))
    if order == "R0L0":
        if mode == 3:
            feat_l = info["info"]["R0R1L0L1"][0, 0]["R0L0"][0, 0]["featL"]
            feat_r = info["info"]["R0R1L0L1"][0, 0]["R0L0"][0, 0]["featR"]
        with open(os.path.join(out_dir, "G0_R0_FEAT.dat"), "w") as f:
            for i in range(feat_r.item().shape[0]):
                for j in range(feat_r.item().shape[1]):
                    f.write("(%3d, %4d) : " % (i, j))
                    for k in range(feat_r.item().shape[2]):
                        t = np.uint32(feat_r.item()[i, j, 1 - k])
                        # remove '0x' and make it 8 digits aligned
                        hex_ = hex(t).split("x")[-1].zfill(8)
                        f.write("%s" % hex_)
                    f.write("\n")
        with open(os.path.join(out_dir, "G0_L0_FEAT.dat"), "w") as f:
            for i in range(feat_l.item().shape[0]):
                for j in range(feat_l.item().shape[1]):
                    f.write("(%3d, %4d) : " % (i, j))
                    for k in range(feat_l.item().shape[2]):
                        t = np.uint32(feat_l.item()[i, j, 1 - k])
                        hex_ = hex(t).split("x")[-1].zfill(8)
                        f.write("%s" % hex_)
                    f.write("\n")
    elif order == "R1L1":
        if mode == 3:
            feat_l = info["info"]["R0R1L0L1"][0, 0]["R1L1"][0, 0]["featL"]
            feat_r = info["info"]["R0R1L0L1"][0, 0]["R1L1"][0, 0]["featR"]
        with open(os.path.join(out_dir, "G0_L1_FEAT.dat"), "w") as f:
            for i in range(feat_r.item().shape[0]):
                for j in range(feat_r.item().shape[1]):
                    f.write("(%3d, %4d) : " % (i, j))
                    for k in range(feat_r.item().shape[2]):
                        t = np.uint32(feat_r.item()[i, j, 1 - k])
                        hex_ = hex(t).split("x")[-1].zfill(8)
                        f.write("%s" % hex_)
                    f.write("\n")
        with open(os.path.join(out_dir, "G0_R1_FEAT.dat"), "w") as f:
            for i in range(feat_l.item().shape[0]):
                for j in range(feat_l.item().shape[1]):
                    f.write("(%3d, %4d) : " % (i, j))
                    for k in range(feat_l.item().shape[2]):
                        t = np.uint32(feat_l.item()[i, j, 1 - k])
                        hex_ = hex(t).split("x")[-1].zfill(8)
                        f.write("%s" % hex_)
                    f.write("\n")


def __kernel_costvol(out_dir, block, suffix):
    with open(os.path.join(out_dir, suffix), "w") as f:
        for i in range(block.item().shape[0]):
            for j in range(block.item().shape[1]):
                f.write("(%3d, %4d) : " % (i, j))
                for k in range(block.item().shape[2]):
                    t = np.int8(block.item()[i, j, k])
                    hex_ = hex(t).split("x")[-1].zfill(2)
                    f.write("%s" % hex_)
                f.write("\n")


def __print_costvol(path, out_dir, order):
    mode = load_json(os.path.join(path, "config.json"))["mode"]
    info = scipy.io.loadmat(os.path.join(path, "info.mat"))
    if order == "R0L0":
        if mode == 3:
            cost_l = info["info"]["R0R1L0L1"][0, 0]["R0L0"][0, 0]["costVolL"]
            cost_r = info["info"]["R0R1L0L1"][0, 0]["R0L0"][0, 0]["costVolR"]
        print(cost_l.item().shape)
        # print(cost_l.item()[:,:,0])
        __kernel_costvol(out_dir, cost_l, COSTVOL_DICT["L0"])
        __kernel_costvol(out_dir, cost_r, COSTVOL_DICT["R0"])
    elif order == "R1L1":
        if mode == 3:
            cost_l = info["info"]["R0R1L0L1"][0, 0]["R1L1"][0, 0]["costVolL"]
            cost_r = info["info"]["R0R1L0L1"][0, 0]["R1L1"][0, 0]["costVolR"]
        print(cost_l.item().shape)
        # print(cost_l.item()[:,:,0])
        __kernel_costvol(out_dir, cost_l, COSTVOL_DICT["L1"])
        __kernel_costvol(out_dir, cost_r, COSTVOL_DICT["R1"])


def print_costvol(path, out_dir, order):
    mode = load_json(os.path.join(path, "config.json"))["mode"]
    info = scipy.io.loadmat(os.path.join(path, "info.mat"))
    if order == "R0L0":
        if mode == 3:
            cost_l = info["info"]["R0R1L0L1"][0, 0]["R0L0"][0, 0]["costVolL"]
            cost_r = info["info"]["R0R1L0L1"][0, 0]["R0L0"][0, 0]["costVolR"]
        print(cost_l.item().shape, cost_r.item().shape)
        # print(cost_l.item()[:,:,0])
        with open(os.path.join(out_dir, "G1_R0_COSTV.dat"), "w") as f:
            for i in range(cost_r.item().shape[0]):
                for j in range(cost_r.item().shape[1]):
                    f.write("(%3d, %4d) : " % (i, j))
                    for k in range(cost_r.item().shape[2]):
                        t = np.int8(cost_r.item()[i, j, k])
                        # f.write("%2d," % t)
                        hex_ = hex(t).split("x")[-1].zfill(2)
                        f.write("%s," % hex_)
                    f.write("\n")
        with open(os.path.join(out_dir, "G1_L0_COSTV.dat"), "w") as f:
            for i in range(cost_l.item().shape[0]):
                for j in range(cost_l.item().shape[1]):
                    f.write("(%3d, %4d) : " % (i, j))
                    for k in range(cost_l.item().shape[2]):
                        t = np.int8(cost_l.item()[i, j, k])
                        # f.write("%2d," % t)
                        hex_ = hex(t).split("x")[-1].zfill(2)
                        f.write("%s," % hex_)
                    f.write("\n")
    elif order == "R1L1":
        if mode == 3:
            cost_l = info["info"]["R0R1L0L1"][0, 0]["R1L1"][0, 0]["costVolL"]
            cost_r = info["info"]["R0R1L0L1"][0, 0]["R1L1"][0, 0]["costVolR"]
        print(cost_l.item().shape, cost_r.item().shape)
        # print(cost_l.item()[:,:,0])
        with open(os.path.join(out_dir, "G1_L1_COSTV.dat"), "w") as f:
            for i in range(cost_r.item().shape[0]):
                for j in range(cost_r.item().shape[1]):
                    f.write("(%3d, %4d) : " % (i, j))
                    for k in range(cost_r.item().shape[2]):
                        t = np.int8(cost_r.item()[i, j, k])
                        # f.write("%2d," % t)
                        hex_ = hex(t).split("x")[-1].zfill(2)
                        f.write("%s," % hex_)
                    f.write("\n")
        with open(os.path.join(out_dir, "G1_R1_COSTV.dat"), "w") as f:
            for i in range(cost_l.item().shape[0]):
                for j in range(cost_l.item().shape[1]):
                    f.write("(%3d, %4d) : " % (i, j))
                    for k in range(cost_l.item().shape[2]):
                        t = np.int8(cost_l.item()[i, j, k])
                        # f.write("%2d," % t)
                        hex_ = hex(t).split("x")[-1].zfill(2)
                        f.write("%s," % hex_)
                    f.write("\n")


if __name__ == "__main__":
    # ROOT_DIR = os.getcwd()
    # folders = get_cur_folder_name()
    ROOT_DIR = "D:/gerrit2/CV3D/depth/chishui/dvcases"
    folders = ["case41"]
    print(folders)
    for folder in folders:
        __print_feat(
            os.path.join(ROOT_DIR, folder), os.path.join(ROOT_DIR, folder), "R1L1"
        )
        __print_feat(
            os.path.join(ROOT_DIR, folder), os.path.join(ROOT_DIR, folder), "R0L0"
        )

        __print_costvol(
            os.path.join(ROOT_DIR, folder), os.path.join(ROOT_DIR, folder), "R1L1"
        )
        __print_costvol(
            os.path.join(ROOT_DIR, folder), os.path.join(ROOT_DIR, folder), "R0L0"
        )

