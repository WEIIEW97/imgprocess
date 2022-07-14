from __future__ import division, print_function, absolute_import
import json
import os
import yaml
import scipy.io


def load_json(path):
    """load json file

    Args:
        path (str): path to json file

    Returns:
        _type_: json object
    """
    with open(path) as j:
        __json = json.load(j)
    return __json


def safeparse_yaml(path):
    with open(path, 'r') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return data


def parse_yaml(path):
    with open(path, 'r') as f:
        data = yaml.load(f.read(), yaml.FullLoader)
    return data


def loadmat(path):
    m = scipy.io.loadmat(path)
    return m


def savemat(path, m):
    scipy.io.savemat(path, m)


def get_cur_folder_name():
    __name = [name for name in os.listdir(".") if os.path.isdir(name)]
    return __name


def file_writer(path, input):
    with open(path, 'wb') as f:
        f.write(input)
