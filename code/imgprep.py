import numpy as np
import cv2
import os
import glob
import re
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from random import sample

def converter(in_dir, in_format, to_fromat, key, type, out_dir):
    cnt = 0
    for img in glob.glob(os.path.join(in_dir, f'*.{in_format}')):
        Image.open(img).convert(type).save(os.path.join(
            out_dir,  f'rect{key}_000{cnt}.{to_fromat}'))
        cnt += 1


def yuv_converter(in_dir, in_format, key, mark, out_dir):
    i = 0
    file_names = os.listdir(in_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    for file in glob.glob(os.path.join(in_dir, f'*.{in_format}')):
        if file.find(key) != -1:
            img_name = file_names[i]
            print(f'Original name is: {img_name}')
            img = cv2.imread(os.path.join(in_dir, img_name))
            # should be `2` but not `0` since opencv read img as `BGR`
            img = img[:, :, 2]
            # get image index
            idx = [_.start() for _ in re.finditer('-', img_name)]
            renam = img_name[idx[-2]+1: idx[-1]]
            print(f'Convert name to: {renam}')
            with open(os.path.join(out_dir, f'rect{mark}_{renam}.yuv'), 'wb') as f:
                np.asarray(img, dtype=np.uint8).tofile(f)
            i += 1
        else:
            raise ValueError("You have been mixed up right/left position.")


def yuv2bmp(dim, src_dir, dst_dir=None, dtype=np.uint8, save=False):
    width, height = dim[0], dim[1]
    with open(src_dir, 'rb') as f:
        img = np.fromfile(f, dtype=dtype, count=width*height)
        img = img.reshape((height, width)).astype(dtype)
        cv2.imshow('bmp image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if save:
            cv2.imwrite(dst_dir, img)


def yuv2bmpng(dim, key, dep_dir, out_dir, format, dtype=np.uint16):
    width = dim[0]
    height = dim[1]
    i = 0
    file_names = os.listdir(dep_dir)
    if 'out_config.txt' in file_names:
        file_names.remove('out_config.txt')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    for file in glob.glob(os.path.join(dep_dir, '*.yuv')):
        if file.find(key) != -1:
            img_name = file_names[i]
            name = img_name.replace('yuv', f'{format}')
            img = np.fromfile(os.path.join(dep_dir, img_name), dtype=dtype)
            img = np.reshape(img, [height, width])
            cv2.imwrite(os.path.join(out_dir, name), img)
            i += 1
        else:
            raise ValueError("You have been mixed up right/left position.")


def png2colormap(png_dir, color_dir):
    if os.path.exists(color_dir):
        shutil.rmtree(color_dir)
    os.makedirs(color_dir)
    file_names = os.listdir(png_dir)
    i = 0
    for file in glob.glob(os.path.join(png_dir, '*.png')):
        img_name = file_names[i]
        print(img_name)
        print(file)
        img = plt.imread(file)
        if len(img) == 3:
            img = img[:, :, 0]
        sc = plt.imshow(img)
        sc.set_cmap('jet')
        # plt.colorbar()
        plt.colorbar(sc)
        # plt.show()
        plt.savefig(os.path.join(color_dir, img_name))
        i += 1


def yuv2colormap(dep_dir, dim, out_dir):
    width = dim[0]
    height = dim[1]
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    file_names = os.listdir(dep_dir)
    file_names.remove('out_config.txt')
    i = 0
    for file in glob.glob(os.path.join(dep_dir, '*.yuv')):
        img_name = file_names[i]
        print(img_name)
        img = np.fromfile(file, dtype=np.uint16)
        img = np.reshape(img, [height, width])
        img = img.astype(np.uint8)
        img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imshow('colormap', img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        name = img_name.replace('yuv', 'png')
        cv2.imwrite(os.path.join(out_dir, name), img_color)
        i += 1


def rgb2yuv(in_path, out_path, name, channel):
    os.makedirs(out_path, exist_ok=True)
    img = cv2.imread(in_path)
    if channel == 'Y':
        img = img[:,:,2]
    elif channel == 'U':
        img = img[:,:,1]
    elif channel == 'V':
        img = img[:,:,0]
    with open(os.path.join(out_path, name), 'wb') as f:
        np.asarray(img, dtype=np.uint8).tofile(f)
    return


def file_opener(path, dtype, channels=1):
    with open(path) as f:
        out = np.fromfile(f, dtype=dtype)
    if channels == 3:
        out = cv2.imread(path)
    return out


def img2disp(img, dim):
    h, w = dim[0], dim[1]
    disp = img.reshape((h, w))
    _disp = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            _disp[i, j] = float(disp[i, j]) / 64.0
    return _disp


def raw2disp(raw, dim):
    if len(raw.shape) == 3:
        img = raw[:, :, 0]
    elif len(raw.shape) == 2:
        img = raw
    else:
        raise ValueError('Input dimension must be placed in right shape.')
    h, w = dim[0], dim[1]
    disp = img.reshape((h, w))
    _disp = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            _disp[i, j] = float(disp[i, j]) / 64.0
    return _disp


def reshaper(input, dim):
    if len(dim) <= 2:
        h, w = dim[0], dim[1]
        out = input.reshape((h, w))
    elif len(dim) == 3:
        h, w, c = dim[0], dim[1], dim[2]
        out = input.shape((h, w, c))
    return out


def calc_foc_baseline(disparity, depth, size=10, is_sample=True):
    if disparity.shape != depth.shape:
        raise ValueError('Disparity and depth must remain same dimension.')
    disp_temp = disparity.reshape(-1)
    depth_temp = depth.reshape(-1)
    idx = np.where(disp_temp == 0)
    _disp = np.delete(disp_temp, idx)
    _depth = np.delete(depth_temp, idx)
    if is_sample is  False:
        fb = np.mean(disparity * depth)
    else:
        total_idx = np.array(range(len(disp_temp)))
        remain_idx = np.delete(total_idx, idx)
        sample_idx = np.random.choice(remain_idx, size, replace=False)
        disp_sample = _disp[sample_idx]
        depth_sample = _depth[sample_idx]
        fb = np.mean(disp_sample * depth_sample)
    return fb


def disp2depth(disp, fb):
    if len(disp.shape) > 1:
        disp = disp.reshape(-1)
    depth = fb / disp
    depth[np.where(disp == 0)] = 0
    return depth


def conv2d(data, kernel_size, pooling):
    kernel = np.ones((kernel_size, kernel_size))
    h_raw, w_raw = data.shape[0], data.shape[1]
    out_dim = np.zeros((h_raw-kernel_size+1, w_raw-kernel_size+1))
    # print(out_dim.shape[0], out_dim.shape[1])
    h, w = out_dim.shape[0], out_dim.shape[1]
    out = np.zeros(shape=(h, w))
    if pooling == 'avg':
        for i in range(h):
            for j in range(w):
                out[i, j] = np.round(
                    np.mean(data[i:i+kernel_size, j:j+kernel_size] * kernel))
    elif pooling == 'max':
        for i in range(h):
            for j in range(w):
                out[i, j] = np.max(
                    data[i:i+kernel_size, j:j+kernel_size] * kernel)
    return out


def plot_grid(x, y, xrange, yrange, grid_size, suptitle, ylabel, xlabel, pooling, kernel, type, pos):
    path = os.getcwd()
    save_path = os.path.join(path, f'kernel_{kernel}')
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(suptitle)
    x_range = xrange
    y_sub = yrange
    i = 1
    for _y in y_sub:
        ax = plt.subplot(grid_size[0], grid_size[1], i)
        ax.scatter(
            x[x_range[0]:x_range[1], _y],
            y[x_range[0]:x_range[1], _y])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        i += 1
    plt.savefig(os.path.join(save_path, f'{pos}_{pooling}_{type}.png'))


def plot_color_combine(x, y, xrange, yrange, grid_size, suptitle, ylabel, xlabel, pooling, kernel, type, pos):
    path = os.getcwd()
    save_path = os.path.join(path, f'kernel_{kernel}')
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(suptitle)
    x_range = xrange
    y_sub = yrange
    i = 1
    for _y in y_sub:
        ax = plt.subplot(1, 1, i)
        ax.scatter(
            x[x_range[0]:x_range[1], _y],
            y[x_range[0]:x_range[1], _y])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    plt.savefig(os.path.join(save_path, f'{pos}_{pooling}_{type}.png'))


def imgcrop(in_path:str, out_path:str, crop_size:list):
    img = cv2.imread(in_path)
    w, h = crop_size[0], crop_size[1]
    if len(img.shape) == 3:
        crop_r = img[:,:,2][0:h, 0:w]
        crop_g = img[:,:,1][0:h, 0:w]
        crop_b = img[:,:,0][0:h, 0:w]
        crop = np.dstack((crop_r, crop_g, crop_b))
    elif len(img.shape) == 1:
        crop = img[0:h, 0:w]
    cv2.imwrite(out_path, crop)
    return

def rect_lr_visualizer(left_img_path, right_img_path, thr, out_path, thickness=1, line_type=8, save=False):
    img_l = cv2.imread(left_img_path)
    img_r = cv2.imread(right_img_path)

    lr_r = np.hstack((img_l[:,:,2], img_r[:,:,2]))
    lr_g = np.hstack((img_l[:,:,1], img_r[:,:,1]))
    lr_b = np.hstack((img_l[:,:,0], img_r[:,:,0]))

    lr = np.dstack((lr_b, lr_g, lr_r))
    h, w, _ = lr.shape
    NUM_SAMP = 5
    POINT_COLOR = (0, 0, 255)

    samp_line = sample(range(thr, h-thr), NUM_SAMP)
    for samp in samp_line:
        cv2.line(lr, (0, samp), (w, samp), POINT_COLOR, thickness, line_type)
    cv2.imshow('left-right rectified visualization', lr)
    cv2.waitkey(0)
    cv2.destroyAllWindows()
    if save:
        cv2.imwrite(out_path, lr)
    return


if __name__ == '__main__':
    # IN_DIR = 'D:/data/118/img/R'
    # OUT_DIR = 'D:/data/118/img/RYUV'
    # yuv_converter(in_dir=IN_DIR, in_format='bmp', key='right', mark='R', out_dir=OUT_DIR)

    # src_dir = 'D:/data/118/img/rect_yuv/rectL_2070.yuv'
    # dst_dir = 'D:/data/118/img/rect_yuv/rectL_2070.bmp'
    # yuv2bmp(dim=[1280, 800], src_dir=src_dir, dst_dir=dst_dir, save=True)

    # CANNOT SAVE IMAGE IN THIS HEIGHT!!
    src_dir = 'D:/data/118/img/out_depth/out_depth_1871.yuv'
    dst_dir = 'D:/data/118/img/out_depth/colorpy/out_depth_1871.png'
    img = file_opener(src_dir, dtype=np.uint16)
    disp = img2disp(img, [800, 1280])
    depth = disp2depth(disp, 679.306486*54.476586)
    os.mkdir('D:/data/118/img/out_depth/colorpy')
    cv2.imwrite(dst_dir, np.uint16(depth))
