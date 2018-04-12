import pickle
import os
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def dopickle(file, ob):
    with open(file, 'wb') as fo:
        pickle.dump(ob, fo)


def convert_data(path, img_size=(32, 32)):
    if os.path.exists(path):
        print('Loading Dataset 2 Images')
        files = os.listdir(path)
        files = [x for x in files if 'batch' in x and 'bw' not in x]
        convertor = [0.299, 0.587, 0.114]

        for f in files:
            print('Converting File:', f)
            bw_data = []
            train_dict = unpickle(os.path.join(path, f))
            train_data = train_dict.get(b'data')
            train_labels = train_dict.get(b'labels')
            for d in train_data:
                x = []
                for i in range(1024):
                    xx = [d[i], d[i + 1024], d[i + 2048]]
                    x.append(np.dot(xx, convertor))
                bw_data.append(x)
                # from skimage.io import imshow
                # print(np.reshape(x, (32, 32), ).astype(int))
                # imshow(np.reshape(x, (32, 32), ).astype(int))
                # import matplotlib.pyplot as plt
                # plt.imshow(np.reshape(x, (32, 32)), cmap='gray')
                # plt.show()
                # break
            dopickle(os.path.join(path, 'bw_' + f), {b'data': bw_data, b'labels': train_labels})
    else:
        print('Dataset doesn\'t exist')


def read_dataset_bw(path, img_size=(32, 32)):
    if os.path.exists(path):
        print('Loading BW Dataset 2 Images')
        files = os.listdir(path)
        files = [x for x in files if 'bw_' in x]
        for f in files:
            train_dict = unpickle(os.path.join(path, f))
            train_data = train_dict.get(b'data')
            print('MAGIC', len(train_data))
    else:
        print('Dataset doesn\'t exist')


# DATA2_PATH = os.path.join(os.path.abspath('..'), 'data\\q4_data\\dataset2')
convert_data("F:\IIITD\Semester_2\Statistical Machine Learning\A4\dataset\cifar-10")
read_dataset_bw(DATA2_PATH)