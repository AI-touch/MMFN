import csv
import pandas as pd
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import StratifiedShuffleSplit


def Data_less(path, human=[], emotion=[]):
    label_ges = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    gesture_labels = np.array(label_ges)
    emotion_labels = np.array(emotion)

    emotion_name_to_id = dict(zip(emotion_labels, range(len(emotion_labels))))

    sequences_ges = []
    sequences_face = []
    label_emo = []

    for i in range(len(human)):
        ges_person = []
        face_person = []
        label_emo_person = []
        for j in range(len(emotion)):
            path_emo = os.listdir(path + human[i] + '/' + emotion[j])
            for m in range(len(path_emo)):
                if int(path_emo[m].split('_')[-1]) < 3:
                    case_path = path + human[i] + '/' + emotion[j] + '/' + path_emo[m] + '/' + 'ges_data/' + str(path_emo[m]) + '.csv'
                    df = pd.read_csv(case_path, header=0, index_col=0)
                    values = df.values
                    values = values.reshape(30, 10, 10)
                    values = np.true_divide(values, 1024)
                    values = np.expand_dims(values, 0)
                    '''face'''
                    path_face = path + human[i] + '/' + emotion[j] + '/' + path_emo[m] + '/' + 'face_data/'
                    img = []
                    files = os.listdir(path_face)
                    files.sort(key=lambda x: int(x[6:-4]))
                    for p in range(len(files)):
                        faceimg = Image.open(path_face + '/' + files[p])  # .convert('RGB')
                        pil_image = np.array(faceimg)
                        img.append(pil_image)
                    img = np.array(img)
                    img = np.transpose(img, [3, 0, 1, 2])
                    # img = torch.from_numpy(img)
                    face_person.append(img)
                    ges_person.append(values)
                    label_emo_person.append(emotion_name_to_id[emotion[j]])
                else:
                    pass
        sequences_ges.append(ges_person)
        sequences_face.append(face_person)
        label_emo.append(label_emo_person)
    return sequences_ges, sequences_face, label_emo

class CustomDataSet(Dataset):
    def __init__(
            self,
            ges,
            face,
            labels):
        self.ges = ges
        self.face = face
        self.labels = labels

    def __getitem__(self, index):
        gestures = self.ges[index]
        faces = self.face[index]
        label = self.labels[index]
        return gestures, faces, label

    def __len__(self):
        count = len(self.ges)

        assert len(
            self.ges) == len(self.face)
        return count

def Data_set(data_path = '/home/ps/LYK/MyMSDA4/data/',train_human= ['CL'], mode = 'Ges'):
    data_set = []
    if mode == 'Ges':
        for i in range(len(train_human)):
            data_ges, _, target = Data_less(path=data_path,
                                                   human=[train_human[i]],
                                                   emotion=['happiness', 'surprise', 'fear', 'anger', 'disgust',
                                                            'sadness'])
            data = np.squeeze(np.array(data_ges), axis=0)
            target = np.transpose(np.array(target),(1,0))
            np.random.seed(20)
            shuffle_ix = np.random.permutation(len(data))
            shuffle_ix = list(shuffle_ix)
            data = data[shuffle_ix, :]
            target = target[shuffle_ix, :]
            target = np.squeeze(target, axis=1)
            trainX = torch.from_numpy(data)  # 将数组转化为张量，并且二者共享内存，trainX改变，Xtr也会改变
            trainy = torch.from_numpy(target)
            dataset = torch.utils.data.TensorDataset(trainX, trainy)
            print(str(train_human[i])+'数据集合完成', trainX.size(), trainy.size())
            data_set.append(dataset)
        return data_set

    elif mode == 'Face':
        for i in range(len(train_human)):
            _, data_face, target = Data_less(path=data_path,
                                                   human=[train_human[i]],
                                                   emotion=['happiness', 'surprise', 'fear', 'anger', 'disgust',
                                                            'sadness'])
            data = np.squeeze(np.array(data_face), axis=0)
            target = np.transpose(np.array(target),(1,0))
            np.random.seed(10)
            shuffle_ix = np.random.permutation(len(data))
            shuffle_ix = list(shuffle_ix)
            data = data[shuffle_ix, :]
            target = target[shuffle_ix, :]
            target = np.squeeze(target, axis=1)
            trainX = torch.from_numpy(data)  # 将数组转化为张量，并且二者共享内存，trainX改变，Xtr也会改变
            trainy = torch.from_numpy(target)
            dataset = torch.utils.data.TensorDataset(trainX, trainy)
            print(str(train_human[i])+'数据集合完成', trainX.size(), trainy.size())
            data_set.append(dataset)
        return data_set

    elif mode == 'both':
        for i in range(len(train_human)):
            data_ges, data_face, target = Data_less(path=data_path,
                                                   human=[train_human[i]],
                                                   emotion=['happiness', 'surprise', 'fear', 'anger', 'disgust',
                                                            'sadness'])
            data_ges = np.squeeze(np.array(data_ges), axis=0)
            data_face = np.squeeze(np.array(data_face), axis=0)
            target = np.transpose(np.array(target),(1,0))
            np.random.seed(10)
            shuffle_ix = np.random.permutation(len(data_ges))
            shuffle_ix = list(shuffle_ix)
            data_ges = data_ges[shuffle_ix, :]
            data_face = data_face[shuffle_ix, :]
            target = target[shuffle_ix, :]
            target = np.squeeze(target, axis=1)
            trainX_ges = torch.from_numpy(data_ges)  # 将数组转化为张量，并且二者共享内存，trainX改变，Xtr也会改变
            trainX_face = torch.from_numpy(data_face)  # 将数组转化为张量，并且二者共享内存，trainX改变，Xtr也会改变
            trainy = torch.from_numpy(target)
            dataset = CustomDataSet(ges=trainX_ges,face=trainX_face,labels=trainy)
            print(str(train_human[i])+'数据集合完成', trainX_ges.size(), trainX_face.size(), trainy.size())
            data_set.append(dataset)
        return data_set
