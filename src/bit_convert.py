from __future__ import division, print_function, absolute_import

import numpy as np
from numba import jit
import scipy.io
from .utils import *


@jit(nopython=True)
def mask_iter(input):
    """Use numba to accelerate the mask iteration

    Args:
        input (np.array): input array

    Returns:
        np.array: mask
    """
    msk = np.zeros(int(input.shape[0] / 8), dtype=np.uint8)
    index = 0
    for i in range(msk.shape[0]):
        msk[i] = (
            2 ** 0 * input[index]
            + 2 ** 1 * input[index + 1]
            + 2 ** 2 * input[index + 2]
            + 2 ** 3 * input[index + 3]
            + 2 ** 4 * input[index + 4]
            + 2 ** 5 * input[index + 5]
            + 2 ** 6 * input[index + 6]
            + 2 ** 7 * input[index + 7]
        )
        index += 8
    return msk


@jit(nopython=True)
def disp_iter(input):
    """Use numba to accelerate the disp iteration

    Args:
        input (np.array): input array

    Returns:
        np.array: disp
    """
    dsp = np.zeros(int(input.shape[0]), dtype=np.uint16)
    # for i in range(input.shape[0]):
    #     dsp[i] = input[i]
    dsp = input
    return dsp


@jit(nopython=True)
def disp_iter_6bit(input):
    dsp = np.zeros(int(input.shape[0]), dtype=np.uint8)
    # for i in range(input.shape[0]):
    #     dsp[i] = input[i]
    dsp = input
    return dsp


def mask2bit_jit(path):
    # since there not exists mode 0 or mode 1 for mask, therefore
    # we cannot just return globally, otherwise it will cause
    # reference issue before assignment
    mode = load_json(os.path.join(path, "config.json"))["mode"]
    info = scipy.io.loadmat(os.path.join(path, "info.mat"))
    if mode == 2:
        mask = info["info"]["R0L0"][0, 0]["maskL"][0, 0]
        mask = mask.reshape(-1)
        return mask_iter(mask)
    elif mode == 3:
        mask = info["info"]["R0R1L0L1"][0, 0]["maskL"][0, 0]
        mask = mask.reshape(-1)
        return mask_iter(mask)


def disp2bit_jit(path):
    mode = load_json(os.path.join(path, "config.json"))["mode"]
    info = scipy.io.loadmat(os.path.join(path, "info.mat"))
    if mode == 0:
        disp = info["info"]["R0"][0, 0]["med_sub_dispR_12bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 1:
        disp = info["info"]["R0R1"][0, 0]["med_sub_dispR_12bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 2:
        disp = info["info"]["R0L0"][0, 0]["med_sub_dispL_12bit"][0, 0]
        disp = disp.reshape(-1)
    else:
        disp = info["info"]["R0R1L0L1"][0, 0]["med_sub_dispL_12bit"][0, 0]
        disp = disp.reshape(-1)
    return disp_iter(disp)


def disp2bit_jit_6(path):
    """retrive low 6 bit disp

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    mode = load_json(os.path.join(path, "config.json"))["mode"]
    info = scipy.io.loadmat(os.path.join(path, "info.mat"))
    if mode == 0:
        disp = info["info"]["R0"][0, 0]["med_sub_dispR_low6bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 1:
        disp = info["info"]["R0R1"][0, 0]["med_sub_dispR_low6bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 2:
        disp = info["info"]["R0L0"][0, 0]["med_sub_dispL_low6bit"][0, 0]
        disp = disp.reshape(-1)
    else:
        disp = info["info"]["R0R1L0L1"][0, 0]["med_sub_dispL_low6bit"][0, 0]
        disp = disp.reshape(-1)
    return disp_iter_6bit(disp)


def disp2bit_jit_mdf(path):
    """retrive median filter disp

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    mode = load_json(os.path.join(path, "config.json"))["mode"]
    info = scipy.io.loadmat(os.path.join(path, "info.mat"))
    if mode == 0:
        disp = info["info"]["R0"][0, 0]["med_dispR_6bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 1:
        disp = info["info"]["R0R1"][0, 0]["med_dispR_6bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 2:
        disp = info["info"]["R0L0"][0, 0]["med_dispL_6bit"][0, 0]
        disp = disp.reshape(-1)
    else:
        disp = info["info"]["R0R1L0L1"][0, 0]["med_dispL_6bit"][0, 0]
        disp = disp.reshape(-1)
    return disp_iter_6bit(disp)


def disp2bit_jit_enh(path):
    """retrive enh disp(being md filtered and intercept to high 6 bit)

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    mode = load_json(os.path.join(path, "config.json"))["mode"]
    info = scipy.io.loadmat(os.path.join(path, "info.mat"))
    if mode == 0:
        disp = info["info"]["R0"][0, 0]["med_sub_dispR_high6bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 1:
        disp = info["info"]["R0R1"][0, 0]["med_sub_dispR_high6bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 2:
        disp = info["info"]["R0L0"][0, 0]["med_sub_dispL_high6bit"][0, 0]
        disp = disp.reshape(-1)
    else:
        disp = info["info"]["R0R1L0L1"][0, 0]["med_sub_dispL_high6bit"][0, 0]
        disp = disp.reshape(-1)
    return disp_iter_6bit(disp)


def mask2bit(path):
    # since there not exists mode 0 or mode 1 for mask, therefore
    # we cannot just return globally, otherwise it will cause
    # reference issue before assignment
    mode = load_json(os.path.join(path, "config.json"))["mode"]
    info = scipy.io.loadmat(os.path.join(path, "info.mat"))
    if mode == 2:
        mask = info["info"]["R0L0"][0, 0]["maskL"][0, 0]
        mask = mask.reshape(-1)
        msk = np.zeros(int(mask.shape[0] / 8), dtype=np.uint8)
        index = 0
        for i in range(msk.shape[0]):
            msk[i] = (
                2 ** 0 * mask[index]
                + 2 ** 1 * mask[index + 1]
                + 2 ** 2 * mask[index + 2]
                + 2 ** 3 * mask[index + 3]
                + 2 ** 4 * mask[index + 4]
                + 2 ** 5 * mask[index + 5]
                + 2 ** 6 * mask[index + 6]
                + 2 ** 7 * mask[index + 7]
            )
            index += 8
        return msk
    elif mode == 3:
        mask = info["info"]["R0R1L0L1"][0, 0]["maskL"][0, 0]
        mask = mask.reshape(-1)
        msk = np.zeros(int(mask.shape[0] / 8), dtype=np.uint8)
        index = 0
        for i in range(msk.shape[0]):
            msk[i] = (
                2 ** 0 * mask[index]
                + 2 ** 1 * mask[index + 1]
                + 2 ** 2 * mask[index + 2]
                + 2 ** 3 * mask[index + 3]
                + 2 ** 4 * mask[index + 4]
                + 2 ** 5 * mask[index + 5]
                + 2 ** 6 * mask[index + 6]
                + 2 ** 7 * mask[index + 7]
            )
            index += 8
        return msk


def disp2bit(path):
    mode = load_json(os.path.join(path, "config.json"))["mode"]
    info = scipy.io.loadmat(os.path.join(path, "info.mat"))
    if mode == 0:
        disp = info["info"]["R0"][0, 0]["med_sub_dispR_12bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 1:
        disp = info["info"]["R0R1"][0, 0]["med_sub_dispR_12bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 2:
        disp = info["info"]["R0L0"][0, 0]["med_sub_dispL_12bit"][0, 0]
        disp = disp.reshape(-1)
    else:
        disp = info["info"]["R0R1L0L1"][0, 0]["med_sub_dispL_12bit"][0, 0]
        disp = disp.reshape(-1)
    dsp = np.zeros(int(disp.shape[0]), dtype=np.uint16)
    dsp = disp
    return dsp


def disp2bit_6(path):
    mode = load_json(os.path.join(path, "config.json"))["mode"]
    info = scipy.io.loadmat(os.path.join(path, "info.mat"))
    if mode == 0:
        disp = info["info"]["R0"][0, 0]["med_sub_dispR_low6bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 1:
        disp = info["info"]["R0R1"][0, 0]["med_sub_dispR_low6bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 2:
        disp = info["info"]["R0L0"][0, 0]["med_sub_dispL_low6bit"][0, 0]
        disp = disp.reshape(-1)
    else:
        disp = info["info"]["R0R1L0L1"][0, 0]["med_sub_dispL_low6bit"][0, 0]
        disp = disp.reshape(-1)
    dsp = np.zeros(int(disp.shape[0]), dtype=np.uint8)
    dsp = disp
    return dsp


def disp2bit_mdf(path):
    mode = load_json(os.path.join(path, "config.json"))["mode"]
    info = scipy.io.loadmat(os.path.join(path, "info.mat"))
    if mode == 0:
        disp = info["info"]["R0"][0, 0]["med_dispR_6bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 1:
        disp = info["info"]["R0R1"][0, 0]["med_dispR_6bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 2:
        disp = info["info"]["R0L0"][0, 0]["med_dispL_6bit"][0, 0]
        disp = disp.reshape(-1)
    else:
        disp = info["info"]["R0R1L0L1"][0, 0]["med_dispL_6bit"][0, 0]
        disp = disp.reshape(-1)
    dsp = np.zeros(int(disp.shape[0]), dtype=np.uint8)
    dsp = disp
    return dsp


def disp2bit_enh(path):
    mode = load_json(os.path.join(path, "config.json"))["mode"]
    info = scipy.io.loadmat(os.path.join(path, "info.mat"))
    if mode == 0:
        disp = info["info"]["R0"][0, 0]["med_sub_dispR_high6bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 1:
        disp = info["info"]["R0R1"][0, 0]["med_sub_dispR_high6bit"][0, 0]
        disp = disp.reshape(-1)
    elif mode == 2:
        disp = info["info"]["R0L0"][0, 0]["med_sub_dispL_high6bit"][0, 0]
        disp = disp.reshape(-1)
    else:
        disp = info["info"]["R0R1L0L1"][0, 0]["med_sub_dispL_high6bit"][0, 0]
        disp = disp.reshape(-1)
    dsp = np.zeros(int(disp.shape[0]), dtype=np.uint8)
    dsp = disp
    return dsp


if __name__ == "__main__":
    # ROOT_DIR = os.getcwd()
    # folders = get_cur_folder_name()
    ROOT_DIR = "D:/gerrit2/CV3D/depth/chishui/dvcases"
    folders = ["case41"]
    print(folders)
    for folder in folders:
        maskk = mask2bit_jit(os.path.join(ROOT_DIR, folder))
        dispp = disp2bit_jit(os.path.join(ROOT_DIR, folder))
        # dispp6 = disp2bit_jit_6(os.path.join(ROOT_DIR, folder))
        # disp_mdf = disp2bit_jit_mdf(os.path.join(ROOT_DIR, folder))
        # disp_enh = disp2bit_enh(os.path.join(ROOT_DIR, folder))
        # exclude the NoneType maskk and make sure maskk is not null
        if type(maskk) == np.ndarray and len(maskk) != 0:
            file_writer(os.path.join(ROOT_DIR, folder, "out_mask.bin"), maskk)
        if len(dispp) != 0:  # make sure dispp is not null
            file_writer(os.path.join(ROOT_DIR, folder, "out_disp.bin"), dispp)
        # if len(disp_mdf) != 0:
        #     file_writer(os.path.join(ROOT_DIR, folder, 'out_disp_mdf.bin'), disp_mdf)
        # if len(dispp6) != 0:
        #     file_writer(os.path.join(ROOT_DIR, folder, 'out_disp_low6bit.bin'), dispp6)
        # if len(disp_enh) != 0:  # make sure dispp is not null
        #     file_writer(os.path.join(ROOT_DIR, folder,
        #                 'out_disp_enh.bin'), disp_enh)

    # maskk = mask2bit_jit(os.path.join(ROOT_DIR, folder))
    # dispp = disp2bit_jit(os.path.join(ROOT_DIR, folder))
    # if type(maskk) == np.ndarray and len(maskk) != 0:   # exclude the NoneType maskk and make sure maskk is not null
    #     file_writer(os.path.join(ROOT_DIR, folder, 'out_mask.bin'), maskk)
    # if len(dispp) != 0: # make sure dispp is not null
    #     file_writer(os.path.join(ROOT_DIR, folder, 'out_disp.bin'), dispp)
