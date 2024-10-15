#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@文件        :converter.py
@说明        :用于将样本转为png图片
@时间        :2024/10/15 17:17:04
@作者        :winnower
@版本        :1.0
'''

from utils import convert

TRAIN_SET = './DataSets/MNIST/processed/training.pt'
TEST_SET = './DataSets/MNIST/processed/test.pt'
SAVE_PATH = './Images/train'
NUM_TRAIN = 5
NUM_TEST = 5

convert.toImages(TRAIN_SET, SAVE_PATH, NUM_TRAIN)