import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 获取样本地址
def findfile(path, file_last_name):
    file_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        # 如果是文件夹，则递归
        if os.path.isdir(file_path):
            findfile(file_path, file_last_name)
        elif os.path.splitext(file_path)[1] == file_last_name:
            file_name.append(file_path)
    return file_name


# 读取数据
def read_data(dir):
    folder_nums = 0  # 从第0个文件夹开始，遍历到第1个，共2个文件夹
    file_nums_count = 0  # 文件计数器
    data = []  # 总的训练集集合
    labels = []  # 创建每组数据对应的标签
    for folder in os.listdir(dir):
        folder_path = os.path.join(dir, folder)
        print(folder_nums, folder, folder_path)     # 1 1_非健康 ./data/txt_data_for_breath/spiking_train_val/1_非健康
        file_path = findfile(folder_path, '.txt')
        label = folder.split('_')[0]
        for file in file_path:
            print(file)
            # 补零数不影响训练结果，矩阵的第二维大于单个神经元的最大脉冲数量即可
            data_per_file = np.zeros((2, 16))
            f = open(file, 'r')
            content = f.readlines()
            f.close()
            # print(len(content)):6
            row = 0
            for items in content:
                data_i = items.split()
                # print(data_i)
                col = 0
                for x in data_i:
                    data_per_file[row][col] = x
                    col += 1
                row += 1
            # print(data_per_file)
            data_reshape = data_per_file.reshape(1, -1)[0]
            print(data_reshape)
            data.append(data_reshape)
            labels.append(int(label))
            file_nums_count += 1
            # print(file_nums_count, '\n')
        folder_nums += 1
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


def main():
    train_dir = './data/txt_data_for_breath/spiking_train_val_3/'
    test_dir = ''

    # 读取数据并处理
    train_data, train_labels = read_data(train_dir)
    # print(train_data, train_labels)

    # 随机K折交叉验证
    k = 5
    train_score_sum = 0
    test_score_sum = 0
    for i in range(k):
        # 划分训练集和验证集，划分后的数据样本和标签对齐，并打乱了排序
        (X_train, X_test, Y_train, Y_test) = train_test_split(train_data, train_labels, test_size=0.2)
        # print(X_train, Y_train, X_test, Y_test, Y_test.ravel())

        # 训练SVM
        # kernal='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合。
        # kernal='rbf'时，为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越分散，分类效果越好，但有可能过拟合。
        # ovr指的是one vs rest，一对多，即一个类别与其他类别进行分类。
        # ovo指的是one vs one，一对一，即将类别两两间进行分类，用二分类的方法模拟多分类的结果。
        support_vector_machine = svm.SVC(C=3, kernel='rbf', gamma=0.02, decision_function_shape='ovo')
        support_vector_machine.fit(X_train, Y_train.ravel())
        # 计算SVC的准确率
        train_score = support_vector_machine.score(X_train, Y_train)
        test_score = support_vector_machine.score(X_test, Y_test)
        print(train_score, test_score)
        train_score_sum += train_score
        test_score_sum += test_score
    print('train accuracy: ' + str(train_score_sum / k) + ' test accuracy ' + str(test_score_sum / k))


if __name__ == '__main__':
    main()
