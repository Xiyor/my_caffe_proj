# coding:utf-8
import glob
import os
import numpy as np
import random
import lmdb
import cv2
from caffe.proto import caffe_pb2
from data_prepare import preprocess_helper


preprocess_obj = preprocess_helper()

def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=preprocess_obj.img_w,
        height=preprocess_obj.img_h,
        label=label,
        data=np.rollaxis(img, 2).tostring())



# 基于结构化的data目录生成lmdb数据

def generate_lmdb_by_path(dir_path, train_lmdb_path, val_lmdb_path, shuffle_flag = True):
    in_db_train = lmdb.open(train_lmdb_path, map_size=int(1e12))
    in_txn_train = in_db_train.begin(write=True)
    in_db_val = lmdb.open(val_lmdb_path, map_size=int(1e12))
    in_txn_val = in_db_val.begin(write=True)

    cur_label = 0
    for sub_dir_path in os.listdir(dir_path):
        cur_img_path_list = os.listdir(os.path.join(dir_path, sub_dir_path))
        if shuffle_flag:
            cur_img_path_list = random.shuffle(cur_img_path_list)

        for in_idx, img_path in enumerate(cur_img_path_list):
            if in_idx % 5 == 0:
                # 构建验证集
                img_val = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img_val = preprocess_helper.prerocess(img_val)
                datum_val = make_datum(img_val, cur_label)
                in_txn_val.put('{:0>8d}'.format(in_idx), datum_val.SerializeToString())

            else:
                # 构建训练集
                img_train = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img_train = preprocess_helper.prerocess(img_train)
                datum_train = make_datum(img_train, cur_label)
                in_txn_train.put('{:0>8d}'.format(in_idx), datum_train.SerializeToString())

        cur_label += 1

    in_db_train.close()
    in_db_val.close()