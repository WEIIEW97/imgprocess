from __future__ import division, print_function, absolute_import
from random import sample

import numpy as np
import cv2
import os
import glob
from PIL import Image

# def converter(in_dir, in_format, to_fromat, key, type, out_dir):
#     cnt = 0
#     for img in glob.glob(os.path.join(in_dir, f'*.{in_format}')):
#         Image.open(img).convert(type).save(os.path.join(
#             out_dir,  f'rect{key}_000{cnt}.{to_fromat}'))
#         cnt += 1


def bin2rgb(dim, src_dir, dst_dir=None, dtype=np.uint8, save=False):
    """convert binary to rgb, only support y channel.

    Args:
        dim (list): width, height
        src_dir (str): input directory
        dst_dir (str, optional): output directory. Defaults to None.
        dtype (_type_, optional): _description_. Defaults to np.uint8.
        save (bool, optional): _description_. Defaults to False.
    """
    width, height = dim[0], dim[1]
    # os.makedirs(dst_dir, exist_ok=True)
    with open(src_dir, "rb") as f:
        img = np.fromfile(f, dtype=dtype, count=width * height)
        img = img.reshape((height, width)).astype(dtype)
        cv2.imshow("rgb image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if save:
            cv2.imwrite(dst_dir, img)


def rgb2yuv(in_path, out_path, name, channel, is_save):
    """convert rgb file to yuv format

    Args:
        in_path (_type_): _description_
        out_path (_type_): _description_
        name (_type_): file name
        channel (str): choose one of channels = ['Y', 'U', 'V']
        is_save (bool): whether to save

    Returns:
        np.array: yuv file stream
    """
    os.makedirs(out_path, exist_ok=True)
    img = cv2.imread(in_path)
    print(img.shape)
    if channel == "Y":
        img = img[:, :, 2]
    elif channel == "U":
        img = img[:, :, 1]
    elif channel == "V":
        img = img[:, :, 0]
    if is_save:
        with open(os.path.join(out_path, name), "wb") as f:
            np.asarray(img, dtype=np.uint8).tofile(f)
    return img


def rgb2nv12(in_path, out_path, is_save):
    """convert rgb to nv12 format, include 3 channels(YUV)

    Args:
        in_path (_type_): _description_
        out_path (_type_): _description_
        is_save (bool): _description_

    Returns:
        nv12: yuv 3 channels stream
    """
    img = cv2.imread(in_path)
    h, w, c = img.shape
    img = img.astype(np.double)

    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    V = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    idx = 0
    nv12 = np.zeros((int(w * h * 3 / 2), 1), np.uint8)
    for i in range(h):
        for j in range(w):
            nv12[idx] = Y[i, j]
            idx += 1

    for k in range(0, h, 2):
        for m in range(0, w, 2):
            nv12[idx] - U[k, m]
            idx += 1
            nv12[idx] = V[k, m]
            idx += 1

    if is_save:
        with open(out_path, "wb") as f:
            np.asarray(nv12, dtype=np.uint8).tofile(f)
    return nv12


def nv12rgb(dim, in_path, out_path, dtype=np.uint8, is_save=True):
    """convert nv12 to rgb format
    -----------------------------------------------------
    # B = 1.164(Y - 16)                  + 2.018(U - 128)
    # G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
    # R = 1.164(Y - 16) + 1.596(V - 128)
    -----------------------------------------------------

    Args:
        dim (_type_): _description_
        in_path (_type_): _description_
        out_path (_type_): _description_
        dtype (_type_, optional): _description_. Defaults to np.uint8.
        is_save (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    w, h = dim[0], dim[1]
    _buffer_end = w * h

    __yuv = np.fromfile(in_path, dtype=dtype)
    y = __yuv[0:_buffer_end].reshape((h, w))
    u = __yuv[_buffer_end::2].reshape((h // 2, w // 2))
    v = __yuv[_buffer_end + 1 :: 2].reshape((h // 2, w // 2))

    u = u.repeat(2, axis=0).repeat(2, axis=1)
    v = v.repeat(2, axis=0).repeat(2, axis=1)

    yuv = np.dstack((y, u, v))

    # convert from uint8 to float32 before subtraction
    yuv = yuv.astype(np.float32)
    yuv[:, :, 0] = yuv[:, :, 0].clip(16, 235) - 16
    yuv[:, :, 1:] = yuv[:, :, 1:].clip(16, 140) - 128

    __convert = np.array(
        [[1.164, 0.000, 2.018], [1.164, -0.813, -0.391], [1.164, 1.596, 0.000]]
    )
    __rgb = np.matmul(yuv, __convert.T).clip(0, 255).astype(dtype)
    rgb = Image.fromarray(__rgb)
    if is_save:
        rgb.save(out_path)
    return rgb


def bidiff(src1, src2, thr, out_path=None, is_save=True):
    """compute the binary-difference between two images, images dimension must be matched.

    Args:
        src1 (_type_): _description_
        src2 (_type_): _description_
        thr (_type_): tolerance threshold
        out_path (_type_, optional): _description_. Defaults to None.
        is_save (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    img1 = cv2.imread(src1)
    img2 = cv2.imread(src2)
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    diff = diff.astype(np.uint8)
    index = np.where(diff <= thr)
    out = np.zeros(img1.shape, dtype=np.uint8)
    out[index] = 255
    if is_save:
        cv2.imwrite(out_path, out)
    return out


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
        raise ValueError("Input dimension must be placed in right shape.")
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
        raise ValueError("Disparity and depth must remain same dimension.")
    disp_temp = disparity.reshape(-1)
    depth_temp = depth.reshape(-1)
    idx = np.where(disp_temp == 0)
    _disp = np.delete(disp_temp, idx)
    _depth = np.delete(depth_temp, idx)
    if is_sample == False:
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


def ind2sub(index, w, fixed_bits=16):
    """convert index to subscript

    Args:
        index (int): element index
        w (int): current position
        fixed_bits (int, optional): _description_. Defaults to 16.

    Returns:
        h, w: destinated subscript
    """
    height = np.floor(index * fixed_bits / w)
    width = index * fixed_bits - height * w
    return int(height), int(width)


def imgcrop(in_path, out_path, up_left, down_right, save=True):
    """crop image from known crop coordinates

    Args:
        in_path (_type_): _description_
        out_path (_type_): _description_
        up_left (list): start crop coord. e.g. --> [0, 0]
        down_right (list): end crop coord. e.g. --> [w, h]
        save (bool, optional): _description_. Defaults to True.

    Returns:
        crop: cropped image array
    """
    img = cv2.imread(in_path)
    w_start, h_start = up_left[0], up_left[1]
    w_end, h_end = down_right[0], down_right[1]
    if len(img.shape) == 3:
        crop_r = img[:, :, 2][h_start:h_end, w_start:w_end]
        crop_g = img[:, :, 1][h_start:h_end, w_start:w_end]
        crop_b = img[:, :, 0][h_start:h_end, w_start:w_end]
        crop = np.dstack((crop_r, crop_g, crop_b))
    elif len(img.shape) == 1:
        crop = img[h_start:h_end, w_start:w_end]
    if save:
        cv2.imwrite(out_path, crop)
    return crop


def imgcrop_f(in_path, out_path, ind, w, crop_w, crop_h, save=True):
    """crop image by a given rdma address.

    Args:
        in_path (_type_): _description_
        out_path (_type_): _description_
        ind (_type_): _description_
        w (_type_): _description_
        crop_w (_type_): _description_
        crop_h (_type_): _description_
        save (bool, optional): _description_. Defaults to True.

    Returns:
        crop: cropped image array
    """
    img = cv2.imread(in_path)
    h_start, w_start = ind2sub(ind, w)
    h_end, w_end = h_start + crop_h, w_start + crop_w
    print(f"start idx{h_start, w_start}, end idx{h_end, w_end}")
    if len(img.shape) == 3:
        crop_r = img[:, :, 2][h_start:h_end, w_start:w_end]
        crop_g = img[:, :, 1][h_start:h_end, w_start:w_end]
        crop_b = img[:, :, 0][h_start:h_end, w_start:w_end]
        crop = np.dstack((crop_r, crop_g, crop_b))
    elif len(img.shape) == 1:
        crop = img[h_start:h_end, w_start:w_end]
    if save:
        cv2.imwrite(out_path, crop)
    return crop


def rect_lr_visualizer(left_img_path, right_img_path, thr, thickness=1, line_type=8):
    """check paired rectified stereo images are being horizonal aligned

    Args:
        left_img_path (_type_): _description_
        right_img_path (_type_): _description_
        thr (int): line begin offset
        thickness (int, optional): _description_. Defaults to 1.
        line_type (int, optional): _description_. Defaults to 8.
    """
    img_l = cv2.imread(left_img_path)
    img_r = cv2.imread(right_img_path)
    lr_r = np.hstack((img_l[:, :, 2], img_r[:, :, 2]))
    lr_g = np.hstack((img_l[:, :, 1], img_r[:, :, 1]))
    lr_b = np.hstack((img_l[:, :, 0], img_r[:, :, 0]))

    lr = np.dstack((lr_b, lr_g, lr_r))
    h, w, _ = lr.shape
    NUM_SAMP = 5
    POINT_COLOR = (0, 0, 255)
    samp_line = sample(range(thr, h - thr), NUM_SAMP)
    for samp in samp_line:
        cv2.line(lr, (0, samp), (w, samp), POINT_COLOR, thickness, line_type)
    cv2.imshow("left-right rectified visualization", lr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def convert_bmp_to_bin(img_path, out_path):
    img = cv2.imread(img_path)
    if len(img.shape) == 3:
        img = img[:, :, 0]
    with open(out_path, "wb") as f:
        img.tofile(f)
    return


def rename(path, base_name, format):
    """rename file name in directory

    Args:
        path (_type_): _description_
        base_name (_type_): _description_
        format (_type_): _description_
    """
    files = os.listdir(path)
    for idx, file in enumerate(files):
        os.rename(
            os.path.join(path, file),
            os.path.join(path, f"{base_name}_{str(idx)}.{format}"),
        )
    return


if __name__ == "__main__":
    # IN_DIR =  'D:/maskFromScale3/source/internal/fusion3/black_block/LRAW'
    # OUT_DIR = 'D:/maskFromScale3/source/internal/fusion3/black_block/YUV'
    # yuv_converter(in_dir=IN_DIR, in_format='bmp', key='L', mark='L', out_dir=OUT_DIR)
    # rgb2yuv(IN_DIR, OUT_DIR, 'rectL_0.yuv', 'Y')

    # src_dir = 'D:/data/118/img/rect_yuv/rectL_2070.yuv'
    # dst_dir = 'D:/data/118/img/rect_yuv/rectL_2070.bmp'
    # yuv2bmp(dim=[1280, 800], src_dir=src_dir, dst_dir=dst_dir, save=True)

    # CANNOT SAVE IMAGE IN THIS HEIGHT!!
    # src_dir = 'D:/data/118/img/out_depth/out_depth_1871.yuv'
    # dst_dir = 'D:/data/118/img/out_depth/colorpy/out_depth_1871.png'
    # img = file_opener(src_dir, dtype=np.uint16)
    # disp = img2disp(img, [800, 1280])
    # depth = disp2depth(disp, 679.306486*54.476586)
    # os.mkdir('D:/data/118/img/out_depth/colorpy')
    # cv2.imwrite(dst_dir, np.uint16(depth))

    IN_DIR = "D:/tmpData/toweiwei0704/input.png"
    OUT_DIR = "D:/tmpData/toweiwei0704/"
    OUR_DIR1 = "D:/tmpData/toweiwei0704/rtl_out1.png"
    # BIN_DIR = 'D:/gerrit2/CV3D/depth/chishui/128x64/case33/hex_src128x64_imgR.bin'
    # yuv_converter(IN_DIR, 'bmp', 'right', 'R', OUT_DIR)
    # rename(OUT_DIR, 'rectR', 'yuv')
    # bin2rgb(dim=[1920, 1080], src_dir=IN_DIR, dst_dir=OUT_DIR, save=True)
    # nv12rgb([1920, 1080], IN_DIR, OUR_DIR1)
    rgb2yuv(IN_DIR, OUT_DIR, "inputY.yuv", "Y", True)
    # rgb2nv12(IN_DIR, OUT_DIR, True)
    # imgcrop(IN_DIR, OUT_DIR, [288, 256], [288+128, 256+64])
    # print(ind2sub(10258, 640))
    # imgcrop_f(IN_DIR, OUT_DIR, 10258, 640, 128, 64)
    # convert_bmp_to_bin(OUT_DIR, BIN_DIR)
    # left = 'D:/CV3D/calibration/nvpMatlabHubble/genCase/case/param1/output/rectL_Y.png'
    # right = 'D:/CV3D/calibration/nvpMatlabHubble/genCase/case/param1/output/rectR_Y.png'
    # rect_lr_visualizer(left, right, 30)
