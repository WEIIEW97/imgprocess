import numpy as np
import scipy.io
from numba import jit
import json
import os

def load_json(path):
    with open(path) as j:
        __json = json.load(j)
    return __json


def get_cur_folder_name():
    __name = [name for name in os.listdir(".") if os.path.isdir(name)]
    return __name


@jit(nopython=True)
def mask_iter(input):
    msk = np.zeros(int(input.shape[0]/8), dtype=np.uint8)
    index = 0
    for i in range(msk.shape[0]):
        msk[i] = 2**0 * input[index] + 2**1 * input[index+1] + 2**2 * input[index+2] + 2**3 * input[index+3] + \
                    2**4 * input[index+4] + 2**5 * input[index+5] + 2**6 * input[index+6] + 2**7 * input[index+7]
        index += 8
    return msk

@jit(nopython=True)
def disp_iter(input):
    dsp = np.zeros(int(input.shape[0]), dtype=np.uint16)
    dsp = input     # remove '\r\n' to '\n'
    return dsp


def mask2bit_jit(path):
    # global mask_iter, load_json
    mode = load_json(os.path.join(path, 'config.json'))['mode']
    info = scipy.io.loadmat(os.path.join(path, 'info.mat'))
    if mode == 2:
        mask = info['info']['R0L0'][0,0]['maskL'][0, 0]
        mask = mask.reshape(-1)
        return mask_iter(mask)
    elif mode == 3:
        mask = info['info']['R0L0'][0,0]['maskL'][0, 0]
        mask = mask.reshape(-1)
        return mask_iter(mask)


def disp2bit_jit(path):
    # global disp_iter, load_json
    mode = load_json(os.path.join(path, 'config.json'))['mode']
    info = scipy.io.loadmat(os.path.join(path, 'info.mat'))
    if mode == 0:
        disp = info['info']['R0'][0,0]['med_sub_dispR_12bit'][0, 0]
        disp = disp.reshape(-1)
    elif mode == 1:
        disp = info['info']['R0R1'][0,0]['med_sub_dispR_12bit'][0, 0]
        disp = disp.reshape(-1)
    elif mode == 2:
        disp = info['info']['R0L0'][0,0]['med_sub_dispL_12bit'][0, 0]
        disp = disp.reshape(-1)
    else:
        disp = info['info']['R0L0'][0,0]['med_sub_dispL_12bit'][0, 0]
        disp = disp.reshape(-1)
    return disp_iter(disp)


def disp2bit_jit_6(path):
    # global disp_iter, load_json
    mode = load_json(os.path.join(path, 'config.json'))['mode']
    info = scipy.io.loadmat(os.path.join(path, 'info.mat'))
    if mode == 0:
        disp = info['info']['r0'][0,0]['med_sub_dispr_low6bit'][0, 0]
        disp = disp.reshape(-1)
    elif mode == 1:
        disp = info['info']['r0r1'][0,0]['med_sub_dispr_low6bit'][0, 0]
        disp = disp.reshape(-1)
    elif mode == 2:
        disp = info['info']['r0l0'][0,0]['med_sub_displ_low6bit'][0, 0]
        disp = disp.reshape(-1)
    else:
        disp = info['info']['r0l0'][0,0]['med_sub_displ_low6bit'][0, 0]
        disp = disp.reshape(-1)
    return disp_iter(disp)


def mask2bit(path):
    mode = load_json(os.path.join(path, 'config.json'))['mode']
    info = scipy.io.loadmat(os.path.join(path, 'info.mat'))
    if mode == 2:
        mask = info['info']['R0L0'][0,0]['maskL'][0, 0]
        mask = mask.reshape(-1)
        msk = np.zeros(int(mask.shape[0]/8), dtype=np.uint8)
        index = 0
        for i in range(msk.shape[0]):
            msk[i] = 2**0 * mask[index] + 2**1 * mask[index+1] + 2**2 * mask[index+2] + 2**3 * mask[index+3] + \
                        2**4 * mask[index+4] + 2**5 * mask[index+5] + 2**6 * mask[index+6] + 2**7 * mask[index+7]
            index += 8
        return msk
    elif mode == 3:
        mask = info['info']['R0L0'][0,0]['maskL'][0, 0]
        mask = mask.reshape(-1)
        msk = np.zeros(int(mask.shape[0]/8), dtype=np.uint8)
        index = 0
        for i in range(msk.shape[0]):
            msk[i] = 2**0 * mask[index] + 2**1 * mask[index+1] + 2**2 * mask[index+2] + 2**3 * mask[index+3] + \
                        2**4 * mask[index+4] + 2**5 * mask[index+5] + 2**6 * mask[index+6] + 2**7 * mask[index+7]
            index += 8
        return msk


def disp2bit(path):
    mode = load_json(os.path.join(path, 'config.json'))['mode']
    info = scipy.io.loadmat(os.path.join(path, 'info.mat'))
    if mode == 0:
        disp = info['info']['R0'][0,0]['med_sub_dispR_12bit'][0, 0]
        disp = disp.reshape(-1)
    elif mode == 1:
        disp = info['info']['R0R1'][0,0]['med_sub_dispR_12bit'][0, 0]
        disp = disp.reshape(-1)
    elif mode == 2:
        disp = info['info']['R0L0'][0,0]['med_sub_dispL_12bit'][0, 0]
        disp = disp.reshape(-1)
    else:
        disp = info['info']['R0L0'][0,0]['med_sub_dispL_12bit'][0, 0]
        disp = disp.reshape(-1)
    dsp = np.zeros(disp.shape[0], dtype=np.uint16)
    dsp = disp
    return dsp



def disp2bit_6(path):
    # global disp_iter, load_json
    mode = load_json(os.path.join(path, 'config.json'))['mode']
    info = scipy.io.loadmat(os.path.join(path, 'info.mat'))
    if mode == 0:
        disp = info['info']['r0'][0,0]['med_sub_dispr_low6bit'][0, 0]
        disp = disp.reshape(-1)
    elif mode == 1:
        disp = info['info']['r0r1'][0,0]['med_sub_dispr_low6bit'][0, 0]
        disp = disp.reshape(-1)
    elif mode == 2:
        disp = info['info']['r0l0'][0,0]['med_sub_displ_low6bit'][0, 0]
        disp = disp.reshape(-1)
    else:
        disp = info['info']['r0l0'][0,0]['med_sub_displ_low6bit'][0, 0]
        disp = disp.reshape(-1)
    dsp = np.zeros(int(disp.shape[0]), dtype=np.uint16)
    dsp = disp
    return dsp


def file_writer(path, input):
    with open(path, 'wb') as f:
        f.write(input)


if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    folders = get_cur_folder_name()
    for folder in folders:
        maskk = mask2bit_jit(os.path.join(ROOT_DIR, folder))
        dispp = disp2bit_jit(os.path.join(ROOT_DIR, folder))
        if type(maskk) == np.ndarray:
            file_writer(os.path.join(ROOT_DIR, folder, 'out_mask.bin'), maskk)
        if len(dispp) != 0:
            file_writer(os.path.join(ROOT_DIR, folder, 'out_disp.bin'), dispp)
