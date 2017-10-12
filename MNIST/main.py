# encoding:utf-8

import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

train_data_file = './train.csv'
test_data_file  = './test.csv'


train_data = pd.read_csv(train_data_file).as_matrix().astype(np.uint8)
test_data  = pd.read_csv(test_data_file).as_matrix().astype(np.uint8)


def extract_images_and_labels(dataset, validation=False):
	#需要将数据转化为[image_num, x, y, depth]格式
	images = dataset[:, 1:].reshape(-1, 28, 28, 1)

	#由于label为0~9,将其转化为一个向量.如将0 转换为 [1,0,0,0,0,0,0,0,0,0]
	labels_dense = dataset[:, 0]
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels)*10
	labels_one_hot = np.zeros((num_labels, 10))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	if validation:
		num_images = images.shape[0]
		divider = num_images - 200
		return images[:divider], labels_one_hot[:divider], images[divider+1:], labels_one_hot[divider+1:]
	else:
		return images, labels_one_hot

def extract_image(dataset):
	return dataset.reshape(-1, 28*28)


train_images, train_labels, val_images, val_labels = extract_images_and_labels(train_data, validation=True)
test_image = extract_image(test_data)


#这里使用tensorflow库中的DataSet类，将images与labels传入生成DataSet对象，该对象可以直接生成mini-batch便于后续训练数据集。
#train，validation，test分别保存训练、交叉验证、测试数据集

train = DataSet(train_images,
				train_labels,
				dtype=np.float32,
				reshape=True)
validation = DataSet(val_images,
					val_labels,
					dtype=np.float32,
					reshape=True)

test = test_image


