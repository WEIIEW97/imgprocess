import numpy as np
import cv2
import os
import glob 
import re
import shutil
from PIL import Image
import matplotlib.pyplot as plt




def converter(in_dir, in_format, to_fromat, key, type, out_dir):
    cnt = 0
    for img in glob.glob(os.path.join(in_dir,f'*.{in_format}')):
        Image.open(img).convert(type).save(os.path.join(out_dir,  f'rec{key}_000{cnt}.{to_fromat}'))
        cnt += 1


def yuv_converter(in_dir,in_format, key, mark, out_dir):
    i = 0
    file_names = os.listdir(in_dir)
    for file in glob.glob(os.path.join(in_dir, f'*.{in_format}')):
        if file.find(key) != -1:
            img_name = file_names[i]
            img = cv2.imread(os.path.join(in_dir, img_name))
            img = img[:,:,0]
            # get image index
            idx = [_.start() for _ in re.finditer('-', img_name)]
            renam = img_name[idx[-2]+1 : idx[-1]-1]

            with open(os.path.join(out_dir, f'rec{mark}_{renam}.yuv'),'wb') as f:
                np.asarray(img.T, dtype=np.uint8).tofile(f)
            i += 1
        else:
            raise ValueError("You have been mixed up right/left position.")


def yuv2png(dim, key, dep_dir, out_dir):
    width = dim[0]
    height = dim[1]
    i = 0
    file_names = os.listdir(dep_dir)
    file_names.remove('out_config.txt')
    ## if do this process recursively, it will lead to the output being deleted recursively
    # if os.path.exists(out_dir):
    #     shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for file in glob.glob(os.path.join(dep_dir, '*.yuv')):
        if file.find(key) != -1:
            img_name = file_names[i]
            name = img_name.replace('yuv', 'png')
            img = np.fromfile(os.path.join(dep_dir, img_name), dtype=np.uint16)
            img = np.reshape(img, [height, width])
            cv2.imwrite(os.path.join(out_dir, name), img)
            i += 1
        else:
            raise ValueError("You have been mixed up right/left position.")


def png2colormap(png_dir, color_dir, clim=[0, 0.01]):
    # if os.path.exists(color_dir):
    #     shutil.rmtree(color_dir)
    os.makedirs(color_dir, exist_ok=True)
    file_names = os.listdir(png_dir)
    i = 0
    cmin = clim[0]
    cmax = clim[1]
    for file in glob.glob(os.path.join(png_dir, '*.png')):
        img_name = file_names[i]
        print(img_name)
        print(file)
        img = plt.imread(file)
        if len(img) == 3:
            img = img[:,:,0]
        sc = plt.imshow(img)
        sc.set_cmap('jet')
        # plt.colorbar()
        plt.colorbar(sc)
        plt.clim(cmin, cmax)
        plt.savefig(os.path.join(color_dir, img_name))
        i += 1



def yuv2colormap(dep_dir, dim, out_dir):
    width = dim[0]
    height = dim[1]
    # if os.path.exists(out_dir):
    #     shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
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
    
        
# def disp2depth(path, dtype, dim):
#     with open(path) as f:
#         disp = np.fromfile(f, dtype=dtype)
#     print(disp.shape)
#     h = dim[0]
#     w = dim[1]
#     disp = disp.reshape((h, w))    
#     depth = np.zeros(disp.shape)
#     for i in range(disp.shape[0]):
#         for j in range(disp.shape[1]):
#             depth[i,j] = float(disp[i,j]) / 64.0
#     return depth


# def read_raw(path, dtype, dim):
#     with open(path) as f:
#         raw = np.fromfile(f, dtype=dtype)
#     h, w = dim[0], dim[1]
#     raw = raw.reshape((h, w))
#     return raw


# function to display the coordinates of
# of the points clicked on the image
# def click_event(event, x, y, flags, params):

# 	# checking for left mouse clicks
# 	if event == cv2.EVENT_LBUTTONDOWN:

# 		# displaying the coordinates
# 		# on the Shell
# 		print(x, ' ', y)

# 		# displaying the coordinates
# 		# on the image window
# 		font = cv2.FONT_HERSHEY_SIMPLEX
# 		cv2.putText(img, str(x) + ',' +
# 					str(y), (x,y), font,
# 					1, (255, 0, 0), 2)
# 		cv2.imshow('image', img)

# 	# checking for right mouse clicks	
# 	if event==cv2.EVENT_RBUTTONDOWN:

# 		# displaying the coordinates
# 		# on the Shell
# 		print(x, ' ', y)

# 		# displaying the coordinates
# 		# on the image window
# 		font = cv2.FONT_HERSHEY_SIMPLEX
# 		b = img[y, x, 0]
# 		g = img[y, x, 1]
# 		r = img[y, x, 2]
# 		cv2.putText(img, str(b) + ',' +
# 					str(g) + ',' + str(r),
# 					(x,y), font, 1,
# 					(255, 255, 0), 2)
# 		cv2.imshow('image', img)
    
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
            _disp[i,j] = float(disp[i,j]) / 64.0
    return _disp


def raw2disp(raw, dim):
    if len(raw.shape) == 3:
        img = raw[:,:,0]
    elif len(raw.shape) == 2:
        img = raw
    else:
        raise ValueError('Input dimension must be placed in right shape.')
    h, w = dim[0], dim[1]
    disp = img.reshape((h, w))
    _disp = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            _disp[i,j] = float(disp[i,j]) / 64.0
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
    _disp  = np.delete(disp_temp, idx)
    _depth = np.delete(depth_temp, idx)
    if is_sample==False:
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
    return (fb / disp)


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
                out[i,j] = np.round(np.mean(data[i:i+kernel_size, j:j+kernel_size] * kernel))
    elif pooling == 'max':
        for i in range(h):
            for j in range(w):
                out[i,j] = np.max(data[i:i+kernel_size, j:j+kernel_size] * kernel)
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
        x[x_range[0]:x_range[1],_y],
        y[x_range[0]:x_range[1],_y])
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
        x[x_range[0]:x_range[1],_y],
        y[x_range[0]:x_range[1],_y])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    plt.savefig(os.path.join(save_path, f'{pos}_{pooling}_{type}.png'))

    


    
    