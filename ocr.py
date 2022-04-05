import os
import cv2
import random
import numpy as np

class SVM(object):
    def __init__(self, fn):
        self.fn = fn
        if os.path.exists(self.fn):
            self.model = cv2.ml.SVM_load(self.fn)
        else:
            self.model = cv2.ml.SVM_create()

    def train(self, samples, responses):
        self.model.setKernel(cv2.ml.SVM_INTER)
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
        self.model.save(self.fn)

    def predict(self, samples):
        _, pred = self.model.predict(samples)
        return pred.ravel()

class Reader(object):
    def __init__(self) -> None:
        self.svms2 = SVM('./param/chars2.svm')
        self.svmsChinese = SVM('./param/chars2Chinese.svm')
        self.groups2 = np.load('./param/chars2.npy')
        self.groupsChinese = np.load('./param/charsChinese.npy')


    def recognize_alnum(self, img) -> str:
        ret = self.svms2.predict(img.reshape((1, -1)).astype('float32')).astype('int32')
        return self.groups2[ret]

    def recognize_chinese(self, img) -> str:
        ret = self.svmsChinese.predict(img.reshape((1, -1)).astype('float32')).astype('int32')
        return self.groupsChinese[ret]

def test():
    dataset_root = './dataset'
    # datasets = ['chars2']
    datasets = ['charsChinese']
    data = []
    groups = []
    for dataset in datasets:
        for group in os.listdir(dataset_root + 
                                '/' + dataset):
            for image in os.listdir(dataset_root + 
                                    '/' + dataset + 
                                    '/' + group):
                data.append(np.append(cv2.imread(dataset_root + 
                                                 '/' + dataset + 
                                                 '/' + group + 
                                                 '/' + image, 0).ravel(), len(groups)))
            groups.append(group)

    # np.save('./chars2.npy', np.array(groups))

    random.shuffle(data)
    data = np.array(data).astype('float32')

    len_train = (int)(data.shape[0] * 0.8)
    data_train = data[:len_train]
    data_pred = data[len_train:]

    svm = SVM('./chars2Chinese.svm')
    # svm.train(data_train[:, :-1], data_train[:, -1].ravel().astype('int32'))
    pred = svm.predict(data[:, :-1])
    print('accuracy: ', np.sum(pred == data[:, -1]) / pred.ravel().shape[0])


if __name__ == '__main__':
    reader = Reader()
    # img = cv2.imread('./dataset/chars2/V/gt_215_2.jpg', 0)
    # txt = reader.recognize_alnum(img)
    img = cv2.imread('./dataset/charsChinese/zh_shan/debug_chineseMat477.jpg', 0)
    txt = reader.recognize_chinese(img)

    print(txt)
