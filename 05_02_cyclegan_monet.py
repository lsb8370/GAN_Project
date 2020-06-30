#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt

from models.cycleGAN import CycleGAN
from utils.loaders import DataLoader


# run params
SECTION = 'paint'
RUN_ID = '0001'
DATA_NAME = 'airplane2line'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode =  'build' # 'build' # 


IMAGE_SIZE = 256

data_loader = DataLoader(dataset_name=DATA_NAME, img_res=(IMAGE_SIZE, IMAGE_SIZE))


# ## 모델 생성

gan = CycleGAN(
        input_dim = (IMAGE_SIZE,IMAGE_SIZE,3)
        , learning_rate = 0.0001
        , lambda_validation = 1
        , lambda_reconstr = 10
        , lambda_id = 5
        , generator_type = 'resnet'
        , gen_n_filters = 64 #원래 32였음
        , disc_n_filters = 64
        )

if mode == 'build':
    gan.save(RUN_FOLDER)
else:
    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))


gan.g_BA.summary()

gan.g_AB.summary()

gan.d_A.summary()

gan.d_B.summary()


# ## 모델 훈련
BATCH_SIZE = 1
EPOCHS = 10
PRINT_EVERY_N_BATCHES = 10

TEST_A_FILE = '0143381.jpg'
TEST_B_FILE = 'n02691156_10391-6.png'

gan.train(data_loader
        , run_folder = RUN_FOLDER
        , epochs=EPOCHS
        , test_A_file = TEST_A_FILE
        , test_B_file = TEST_B_FILE
        , batch_size=BATCH_SIZE
        , sample_interval=PRINT_EVERY_N_BATCHES)
