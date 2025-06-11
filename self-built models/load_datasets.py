import os
import cv2 as cv
import numpy
import numpy as np
import pandas as pd

def get_data(annotations,classification_or_dectection, h, w, mode='valid'):
    if classification_or_dectection=='c':
        labels = annotations['category_id'].values
        label = []
        for i in range(len(labels)):
            if labels[i] == 1:
                label.append([1, 0])
            else:
                label.append([0, 1])
        labels = np.array(label)
    else:
        labels = annotations['bbox'].values
        label=[]
        for i in range(len(labels)):
            # 因为图片缩放成0到1
            temparry=np.array([item for item in labels[i]])
            label.append(temparry)

        labels=np.array(label)


    img_path_list = []
    for dirname, _, filenames in os.walk(r'datasets/Penguins_vs_Turtles/%s' % mode):
        for filename in filenames:
            img_path = os.path.join(dirname, filename)
            img_path_list.append(img_path)
    img_path_list.sort()  # 排序
    print(img_path_list)
    img_list = []
    for img_path in img_path_list:
        img = cv.imread(img_path)
        img = cv.resize(img, dsize=(w, h))
        img_list.append(img)
    imgs = np.array(img_list)
    return imgs, labels


def get_all_data(h, w,classification_or_dectection):
    train_annotations = pd.read_json(r'datasets/Penguins_vs_Turtles/train_annotations')
    valid_annotations = pd.read_json(r'datasets/Penguins_vs_Turtles/valid_annotations')
    train_annotations.drop(['id', 'image_id', 'segmentation', 'area'], axis=1)
    valid_annotations.drop(['id', 'image_id', 'segmentation', 'area'], axis=1)

    x_train, y_train = get_data(train_annotations, classification_or_dectection,h, w, mode='train',)
    x_test, y_test = get_data(valid_annotations, classification_or_dectection,h, w, mode='valid')
    return (x_train / 255.).astype('float32'), y_train, \
           (x_test / 255.).astype('float32'), y_test

