#  Copyright (c) 2019. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import os
import h5py
from functools import reduce

import torch.utils.data as data
from ..pose import generateSampleBox
from opt import opt


class Cabin(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/cabin/images/'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_cabin = 17
        self.nJoints = 17

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))
        # create train/val split
        with h5py.File('../data/cabin/annot_cabin.h5', 'r') as annot:
            # train
            self.imgname_cabin_train = annot['imgname'][:-50]
            self.bndbox_cabin_train = annot['bndbox'][:-50]
            self.part_cabin_train = annot['part'][:-50]
            # val
            self.imgname_cabin_val = annot['imgname'][-50:]
            self.bndbox_cabin_val = annot['bndbox'][-50:]
            self.part_cabin_val = annot['part'][-50:]

        self.size_train = self.imgname_cabin_train.shape[0]
        self.size_val = self.imgname_cabin_val.shape[0]

    def __getitem__(self, index):
        sf = self.scale_factor

        if self.is_train:
            part = self.part_cabin_train[index]
            bndbox = self.bndbox_cabin_train[index]
            imgname = self.imgname_cabin_train[index]
        else:
            part = self.part_cabin_val[index]
            bndbox = self.bndbox_cabin_val[index]
            imgname = self.imgname_cabin_val[index]

        imgname = reduce(lambda x, y: x + y,
                         map(lambda x: chr(int(x)), imgname))
        img_path = os.path.join(self.img_folder, imgname)

        metaData = generateSampleBox(img_path, bndbox, part, self.nJoints,
                                     'cabin', sf, self, train=self.is_train)

        inp, out, setMask = metaData

        return inp, out, setMask, 'cabin'

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val
