
"""
日志管理
"""

import os
import logging
from typing import List

proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_path = os.path.join(proj_path, "log")

# 管理项目路径
os.putenv('proj_path', proj_path)


# 动态添加日志方法
def add_logging_methods(cls):
    log_levels = ['debug', 'info', 'warning', 'error', 'critical']
    for level in log_levels:
        def log_method(self, message, level=level):
            self.log(level, message)
        setattr(cls, level, log_method)
    return cls


@add_logging_methods
class SimpleLogger:
    def __init__(self, instance, cmd_level=logging.DEBUG, file_level=logging.DEBUG):
        name = instance.__class__.__name__
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        # 设置CMD日志
        self.set_handler("StreamHandler", handler_level=cmd_level)
        # 设置文件日志
        self.set_handler("FileHandler", handler_level=file_level, handler_kwargs={"filename": log_path + f"/{name}.log"})

    def log(self, level, message):
        """级别日志统一入口"""
        log_method = getattr(self.logger, level)
        log_method(message)

    def set_handler(self, handler_name, handler_level, handler_kwargs={}):
        """
        handler_name: "FileHandler", "StreamHandler", etc
        handler_kwargs： handler类的实例化参数， fh = logging.FileHandler(**handler_kwargs)
        """
        fmt = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        handler = getattr(logging, handler_name)(**handler_kwargs)
        handler.setFormatter(fmt)
        handler.setLevel(handler_level)
        self.logger.addHandler(handler)


"""
特殊结构
"""

from collections import defaultdict


# 可以嵌套的 defaultdict，防止多级键不存在，KeyError
def nested_dict():
    return defaultdict(nested_dict)


# 让多个变量指向同一个地址，多处修改，同时更新
# 类似于Vue中ref
class Ref:
    def __init__(self, initial_data):
        self.data = initial_data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return f"{self.data}"

    def value(self):
        return self.data


"""
数据清理
"""
import json


def write_json_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def read_urls_txt(remove_dup=False):
    lines = open(proj_path + "/data/top_websites/urls.txt", "r").readlines()
    urls = [line.strip() for line in lines]
    if remove_dup:
        urls = set(urls)
    return list(urls)


def get_value(nested_structure, path):
    if not path:
        return None

    key = path[0]
    if len(path) == 1:
        return nested_structure[key]

    if isinstance(nested_structure, dict):
        if key not in nested_structure:
            return None
        return get_value(nested_structure[key], path[1:])
    elif isinstance(nested_structure, list):
        if not isinstance(key, int) or key >= len(nested_structure):
            return None
        return get_value(nested_structure[key], path[1:])
    else:
        raise TypeError("Unsupported nested structure type.")


def set_value(nested_structure, path, value):
    """
    修改深度嵌套字典中的值，如果路径中不存在则创建。

    :param nested_structure: 嵌套的字典或列表
    :param path: 路径列表
    :param value: 要设置的值
    :return: 修改后的嵌套结构
    """
    if not path:
        return value

    key = path[0]

    if len(path) == 1:
        # 如果路径长度为1，直接设置值
        if isinstance(nested_structure, dict):
            nested_structure[key] = value
        elif isinstance(nested_structure, list):
            while len(nested_structure) <= key:
                nested_structure.append(None)
            nested_structure[key] = value
        return nested_structure

    if isinstance(nested_structure, dict):
        if key not in nested_structure:
            nested_structure[key] = {} if isinstance(path[1], str) else []
        nested_structure[key] = set_value(nested_structure[key], path[1:], value)
    elif isinstance(nested_structure, list):
        if not isinstance(key, int) or key >= len(nested_structure):
            while len(nested_structure) <= key:
                nested_structure.append({} if isinstance(path[1], str) else [])
        nested_structure[key] = set_value(nested_structure[key], path[1:], value)
    else:
        raise TypeError("Unsupported nested structure type.")

    return nested_structure
