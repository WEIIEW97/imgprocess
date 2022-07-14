from __future__ import division, print_function, absolute_import

import numpy as np
import cv2
import os
import glob
import re
import matplotlib.pyplot as plt


def group_converter(in_dir, in_format, key, mark, out_dir):
    """convert file format in directory

    Args:
        in_dir (str): input directory
        in_format (str): input file format
        key (str): key element
        mark (str): mark for 'L' --> left, 'R' --> right
        out_dir (str): output directory

    Raises:
        ValueError: _description_
    """
    i = 0
    file_names = os.listdir(in_dir)
    os.makedirs(out_dir, exist_ok=True)
    for file in glob.glob(os.path.join(in_dir, f"*.{in_format}")):
        if file.find(key) != -1:
            img_name = file_names[i]
            print(f"Original name is: {img_name}")
            img = cv2.imread(os.path.join(in_dir, img_name))
            # should be `2` but not `0` since opencv read img as `BGR`
            img = img[:, :, 2]
            # get image index
            # idx = [_.start() for _ in re.finditer('-', img_name)]
            idx = [_.start() for _ in re.finditer("_", img_name)]
            # renam = img_name[idx[-2]+1: idx[-1]]
            renam = img_name[idx[0] + 1 : -4]
            # renam = img_name
            print(f"Convert name to: {renam}")
            with open(os.path.join(out_dir, f"rect{mark}_{renam}.yuv"), "wb") as f:
                np.asarray(img, dtype=np.uint8).tofile(f)
            i += 1
        else:
            raise ValueError("You have been mixed up right/left position.")


def yuv2rgb(dim, key, dep_dir, out_dir, format, dtype=np.uint16):
    """target on the output of NVP dv4 chip

    Args:
        dim (list): image width, height
        key (str): key word of the image name
        dep_dir (str): input directory
        out_dir (str): output directory
        format (str): target suffix
        dtype (numpy type, optional): storage format. Defaults to np.uint16.

    Raises:
        ValueError: _description_
    """
    width = dim[0]
    height = dim[1]
    i = 0
    file_names = os.listdir(dep_dir)
    if "out_config.txt" in file_names:
        file_names.remove("out_config.txt")
    os.makedirs(out_dir, exist_ok=True)
    for file in glob.glob(os.path.join(dep_dir, "*.yuv")):
        if file.find(key) != -1:
            img_name = file_names[i]
            name = img_name.replace("yuv", f"{format}")
            img = np.fromfile(os.path.join(dep_dir, img_name), dtype=dtype)
            img = np.reshape(img, [height, width])
            cv2.imwrite(os.path.join(out_dir, name), img)
            i += 1
        else:
            raise ValueError("You have been mixed up right/left position.")


def png2colormap(png_dir, color_dir):
    """apply colormap on images

    Args:
        png_dir (_type_): _description_
        color_dir (_type_): _description_
    """
    os.makedirs(color_dir, exist_ok=True)
    file_names = os.listdir(png_dir)
    i = 0
    for file in glob.glob(os.path.join(png_dir, "*.png")):
        img_name = file_names[i]
        print(img_name)
        print(file)
        img = plt.imread(file)
        if len(img) == 3:
            img = img[:, :, 0]
        sc = plt.imshow(img)
        sc.set_cmap("jet")
        # plt.colorbar()
        plt.colorbar(sc)
        # plt.show()
        plt.savefig(os.path.join(color_dir, img_name))
        i += 1


def yuv2colormap(dep_dir, dim, out_dir):
    """apply colormap on yuv format file

    Args:
        dep_dir (_type_): _description_
        dim (_type_): _description_
        out_dir (_type_): _description_
    """
    width = dim[0]
    height = dim[1]
    os.makedirs(out_dir, exist_ok=True)
    file_names = os.listdir(dep_dir)
    file_names.remove("out_config.txt")
    i = 0
    for file in glob.glob(os.path.join(dep_dir, "*.yuv")):
        img_name = file_names[i]
        print(img_name)
        img = np.fromfile(file, dtype=np.uint16)
        img = np.reshape(img, [height, width])
        img = img.astype(np.uint8)
        img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imshow("colormap", img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        name = img_name.replace("yuv", "png")
        cv2.imwrite(os.path.join(out_dir, name), img_color)
        i += 1


def conv2d(data, kernel_size, pooling):
    kernel = np.ones((kernel_size, kernel_size))
    h_raw, w_raw = data.shape[0], data.shape[1]
    out_dim = np.zeros((h_raw - kernel_size + 1, w_raw - kernel_size + 1))
    # print(out_dim.shape[0], out_dim.shape[1])
    h, w = out_dim.shape[0], out_dim.shape[1]
    out = np.zeros(shape=(h, w))
    if pooling == "avg":
        for i in range(h):
            for j in range(w):
                out[i, j] = np.round(
                    np.mean(data[i : i + kernel_size, j : j + kernel_size] * kernel)
                )
    elif pooling == "max":
        for i in range(h):
            for j in range(w):
                out[i, j] = np.max(
                    data[i : i + kernel_size, j : j + kernel_size] * kernel
                )
    return out


def plot_grid(
    x,
    y,
    xrange,
    yrange,
    grid_size,
    suptitle,
    ylabel,
    xlabel,
    pooling,
    kernel,
    type,
    pos,
):
    path = os.getcwd()
    save_path = os.path.join(path, f"kernel_{kernel}")
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(suptitle)
    x_range = xrange
    y_sub = yrange
    i = 1
    for _y in y_sub:
        ax = plt.subplot(grid_size[0], grid_size[1], i)
        ax.scatter(x[x_range[0] : x_range[1], _y], y[x_range[0] : x_range[1], _y])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        i += 1
    plt.savefig(os.path.join(save_path, f"{pos}_{pooling}_{type}.png"))


def plot_color_combine(
    x, y, xrange, yrange, suptitle, ylabel, xlabel, pooling, kernel, type, pos
):
    path = os.getcwd()
    save_path = os.path.join(path, f"kernel_{kernel}")
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(suptitle)
    x_range = xrange
    y_sub = yrange
    i = 1
    for _y in y_sub:
        ax = plt.subplot(1, 1, i)
        ax.scatter(x[x_range[0] : x_range[1], _y], y[x_range[0] : x_range[1], _y])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    plt.savefig(os.path.join(save_path, f"{pos}_{pooling}_{type}.png"))

