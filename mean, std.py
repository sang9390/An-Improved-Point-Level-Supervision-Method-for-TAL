import math

import numpy as np
import sys

kk = [1, 2, 3, 4, 5]

Test_acc_list = []
average_mAP0107_list =[]
average_mAP0105_list =[]
average_mAP0307_list =[]
mAP01_list = []
mAP02_list = []
mAP03_list = []
mAP04_list = []
mAP05_list = []
mAP06_list = []
mAP07_list = []


for i in range(len(kk)):

    model_seed_list = open("/home/HDD2/LJB/ETRI/Proposed_Method(GTEA)/(Na) + (Ba)/outputs/LACP/best_record_seed_"
                           + str(i) + '.txt', 'r')
    data = model_seed_list.read()
    data_split = data.split()

    Test_acc_list.append(float(data_split[3]))
    average_mAP0107_list.append(float(data_split[5]))
    average_mAP0105_list.append(float(data_split[7]))
    average_mAP0307_list.append(float(data_split[9]))
    mAP01_list.append(float(data_split[11]))
    mAP02_list.append(float(data_split[13]))
    mAP03_list.append(float(data_split[15]))
    mAP04_list.append(float(data_split[17]))
    mAP05_list.append(float(data_split[19]))
    mAP06_list.append(float(data_split[21]))
    mAP07_list.append(float(data_split[23]))

print("Test_acc_mean : ", np.mean(Test_acc_list) * 100)
print("Test_acc_std : ", np.std(Test_acc_list) * 100)
print("=========================================")

print("mAP01_mean : ", np.mean(mAP01_list) * 100)
print("mAP01_std : ", np.std(mAP01_list) * 100)
print("=========================================")

print("mAP02_mean : ", np.mean(mAP02_list) * 100)
print("mAP02_std : ", np.std(mAP02_list) * 100)
print("=========================================")

print("mAP03_mean : ", np.mean(mAP03_list) * 100)
print("mAP03_std : ", np.std(mAP03_list) * 100)
print("=========================================")

print("mAP04_mean : ", np.mean(mAP04_list) * 100)
print("mAP04_std : ", np.std(mAP04_list) * 100)
print("=========================================")

print("mAP05_mean : ", np.mean(mAP05_list) * 100)
print("mAP05_std : ", np.std(mAP05_list) * 100)
print("=========================================")

print("mAP06_mean : ", np.mean(mAP06_list) * 100)
print("mAP06_std : ", np.std(mAP06_list) * 100)
print("=========================================")

print("mAP07_mean : ", np.mean(mAP07_list) * 100)
print("mAP07_std : ", np.std(mAP07_list) * 100)
print("=========================================")

print("average_mAP0107_mean : ", np.mean(average_mAP0107_list) * 100)
print("average_mAP0107_std : ", np.std(average_mAP0107_list) * 100)
print("=========================================")

print("average_mAP0105_mean : ", np.mean(average_mAP0105_list) * 100)
print("average_mAP0105_std : ", np.std(average_mAP0105_list) * 100)
print("=========================================")

print("average_mAP0307_mean : ", np.mean(average_mAP0307_list) * 100)
print("average_mAP0307_std : ", np.std(average_mAP0307_list) * 100)
print("=========================================")

