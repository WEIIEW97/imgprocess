import numpy as np
import scipy.io
from numba import jit
import os
import shutil

@jit(nopython=True)
def mask_iter(mask):
    msk = np.zeros(int(mask.shape[0]/8), dtype=np.uint8)
    index = 0
    for i in range(msk.shape[0]):
        msk[i] = 2**0 * mask[index] + 2**1 * mask[index+1] + 2**2 * mask[index+2] + 2**3 * mask[index+3] + \
                    2**4 * mask[index+4] + 2**5 * mask[index+5] + 2**6 * mask[index+6] + 2**7 * mask[index+7]
        index += 8
    return msk

@jit(nopython=True)
def disp_iter(disp):
    dsp = np.zeros(disp.shape[0], dtype=np.uint16)
    for i in range(dsp.shape[0]):
        dsp[i] = disp[i]
    return dsp


def mask2bit_jit(path):
    info = scipy.io.loadmat(path)
    mask = info['info']['R0L0'][0,0]['maskL'][0, 0]
    mask = mask.reshape(-1)
    return mask_iter(mask)


def disp2bit_jit(path):
    info = scipy.io.loadmat(path)
    disp = info['info']['R0L0'][0,0]['med_sub_dispL_12bit'][0, 0]
    disp = disp.reshape(-1)
    return disp_iter(disp)


def mask2bit(path):
    info = scipy.io.loadmat(path)
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
    info = scipy.io.loadmat(path)
    disp = info['info']['R0L0'][0,0]['med_sub_dispL_12bit'][0, 0]
    disp = disp.reshape(-1)
    dsp = np.zeros(disp.shape[0], dtype=np.uint16)
    for i in range(dsp.shape[0]):
        dsp[i] = disp[i]
    return dsp


def file_writer(file, path, mode='wb'):
    with open(os.path.join(path), mode=mode) as f:
        f.write(file)
    return



if __name__ == '__main__':
    IN_DIR = ''
    OUT_DIR = ''
    mask = mask2bit_jit(os.path.join(IN_DIR, 'info.mat'))
    disp = disp2bit_jit(os.path.join(IN_DIR, 'info.mat'))
    file_writer(mask, path=os.path.join(OUT_DIR))
