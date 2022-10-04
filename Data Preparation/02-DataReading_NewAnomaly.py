# In this Experiment we tried to simulate a new occurrence of an anomaly situation
# as a real- world example. For this purpose, we have chosen our train data for
# our classifier from the 3 faulty folders and assign one of the faulty occurrences
# to the test phase and keep the whole folder unseen in the training phase.
# This test helps us to evaluate whether our classifier is overfitted to the
# train data or not.


import numpy as np
import pandas as pd
import os
import glob

path_healthy = "D:/UNI/caseStudy/IR_Images/IR_Images/healthy/healthy_csvData/94689"
path1_fault1_L1K1_94689 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL1K1/FehlerL1K1/94689"
path1_fault2_L1K1K5_94689 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL1K1K5/FehlerL1K1K5/94689"
path1_fault3_L2K3_94689 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL2K3/FehlerL2K3/94689"
path1_fault4_L2K6_7_94689 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL2K6-7/FehlerL2K6-7/94689"

def data_preparation(path_healthy,path_faulty1,path_faulty2,path_faulty3,path_faulty4):

	filenames = glob.glob(os.path.join(path_healthy, "*.xls"))

	health_labels = []
	data = np.empty((32,32))
	for i,filename in enumerate(filenames[:-1]):
	        image = pd.read_csv(filename, engine='python',sep = '\t' , header = None)
	        image = image.replace({',': ''},regex = True).astype(int)
	        data = np.dstack((data,image.values)) #stack in depth
	        health_labels.append(1)
	data = data[:,:,1:]


	data_normalized = (data - data.min())/(data.max()-data.min())
	# print(data_normalized.shape)
	data_normalized = np.transpose(data_normalized,(2,0,1))
	img_No = data_normalized.shape[0]
	indx = np.arange(img_No)
	np.random.shuffle(indx)

	train_No = round(img_No*0.65)
	valid_au_No = round(img_No * 0.1) + train_No
	valid_health_No = round(img_No * 0.1) + valid_au_No
	train_healthy_indx = indx[:train_No]
	valid_auto_indx = indx[train_No:valid_au_No]
	valid_healthy_indx = indx[valid_au_No:valid_health_No]
	test_healthy_indx = indx[valid_health_No:]


	train_auto,valid_auto,valid_health,test_health = data_normalized[train_healthy_indx,:,:],data_normalized[valid_auto_indx,:,:],data_normalized[valid_healthy_indx,:,:],data_normalized[test_healthy_indx,:,:]
	health_labels_train =[]
	auto_labels_valid = []
	healthy_labels_valid = []
	health_labels_test = []
	for i in train_healthy_indx:
		health_labels_train.append(health_labels[i])
	for i in valid_auto_indx:
		auto_labels_valid.append(health_labels[i])
	for i in valid_healthy_indx:
		healthy_labels_valid.append(health_labels[i])
	for i in test_healthy_indx:
		health_labels_test.append(health_labels[i])

	faulty_fileName_1 = glob.glob(os.path.join(path1_fault1_L1K1_94689, "*.xls"))
	faulty_fileName_2 = glob.glob(os.path.join(path1_fault2_L1K1K5_94689, "*.xls"))
	faulty_fileName_3 = glob.glob(os.path.join(path1_fault3_L2K3_94689, "*.xls"))
	faulty_fileName_4 = glob.glob(os.path.join(path1_fault4_L2K6_7_94689, "*.xls"))


	faulty_labels1 = []
	data2 = np.empty((32,32))
	for i,filename in enumerate(faulty_fileName_1[:-1]):
		image = pd.read_csv(filename,engine='python',sep="\t",header=None)
		image = image.replace({",":""}, regex = True).astype(int)
		data2 = np.dstack((data2,image.values))
		faulty_labels1.append(0)

	data2 = data2[:,:,1:]
	data_normalized2 = (data2-data2.min())/(data2.max()-data2.min())
	data_normalized2 = np.transpose(data_normalized2,(2,0,1))
	# print(data_normalized2.shape)
	img_No2 = data_normalized2.shape[0]
	indx2 = np.arange(img_No2)
	np.random.shuffle(indx2)
	train_No2 = round(img_No2 * 0.85)
	train_faulty1_indx = indx2[:train_No2]
	valid_faulty1_indx = indx2[train_No2:]
	train_faulty1 = data_normalized2[train_faulty1_indx,:,:]
	valid_faulty1 = data_normalized2[valid_faulty1_indx,:,:]
	faulty_train_labels1 = []
	faulty_valid_labels1= []
	for i in train_faulty1_indx:
		faulty_train_labels1.append(faulty_labels1[i])
	for i in valid_faulty1_indx:
		faulty_valid_labels1.append(faulty_labels1[i])



#####################################################
	faulty_labels2 = []
	data3 = np.empty((32, 32))
	for i, filename in enumerate(faulty_fileName_2[:-1]):
		image = pd.read_csv(filename, engine='python', sep="\t", header=None)
		image = image.replace({",": ""}, regex=True).astype(int)
		data3 = np.dstack((data3, image.values))
		faulty_labels2.append(0)

	data3 = data3[:, :, 1:]
	data_normalized3 = (data3 - data3.min()) / (data3.max() - data3.min())
	data_normalized3 = np.transpose(data_normalized3, (2, 0, 1))
	# print(data_normalized3.shape)
	img_No3 = data_normalized3.shape[0]
	indx3 = np.arange(img_No3)
	np.random.shuffle(indx3)
	train_No3 = round(img_No2 * 0.85)
	train_faulty2_indx = indx3[:train_No3]
	valid_faulty2_indx = indx3[train_No3:]
	train_faulty2 = data_normalized3[train_faulty2_indx, :, :]
	valid_faulty2 = data_normalized3[valid_faulty2_indx, :, :]
	faulty_train_labels2 = []
	faulty_valid_labels2 = []
	for i in train_faulty2_indx:
		faulty_train_labels2.append(faulty_labels2[i])
	for i in valid_faulty2_indx:
		faulty_valid_labels2.append(faulty_labels2[i])

#########################################################
	faulty_labels3 = []
	data4 = np.empty((32, 32))
	for i, filename in enumerate(faulty_fileName_3[:-1]):
		image = pd.read_csv(filename, engine='python', sep="\t", header=None)
		image = image.replace({",": ""}, regex=True).astype(int)
		data4 = np.dstack((data4, image.values))
		faulty_labels3.append(0)

	data4 = data4[:, :, 1:]
	data_normalized4 = (data4 - data4.min()) / (data4.max() - data4.min())
	data_normalized4 = np.transpose(data_normalized4, (2, 0, 1))
	# print(data_normalized4.shape)
	img_No4 = data_normalized4.shape[0]
	indx4 = np.arange(img_No4)
	np.random.shuffle(indx4)
	train_No4 = round(img_No2 * 0.85)
	train_faulty3_indx = indx4[:train_No4]
	valid_faulty3_indx = indx4[train_No4:]
	train_faulty3 = data_normalized4[train_faulty3_indx, :, :]
	valid_faulty3 = data_normalized4[valid_faulty3_indx, :, :]
	faulty_train_labels3 = []
	faulty_valid_labels3 = []
	for i in train_faulty3_indx:
		faulty_train_labels3.append(faulty_labels3[i])
	for i in valid_faulty3_indx:
		faulty_valid_labels3.append(faulty_labels3[i])
###########################################
	faulty_test_labels= []
	data5 = np.empty((32, 32))
	for i, filename in enumerate(faulty_fileName_4[:-1]):
		image = pd.read_csv(filename, engine='python', sep="\t", header=None)
		image = image.replace({",": ""}, regex=True).astype(int)
		data5 = np.dstack((data5, image.values))
		faulty_test_labels.append(0)

	data5 = data5[:, :, 1:]
	data_normalized5 = (data5 - data5.min()) / (data5.max() - data5.min())
	data_normalized5 = np.transpose(data_normalized5, (2, 0, 1))
	# print(data_normalized5.shape[0])
	img_No5 = data_normalized5.shape[0]
	indx5 = np.arange(img_No5)
	np.random.shuffle(indx5)
	test_faulty = data_normalized5[indx5, :, :]

#################################


	# index2 = np.arange(289)
	# np.random.shuffle(index2)
	# train_faulty_idx = index2[:175]
	# valid_faulty_idx = index2[175:220]
	# test_faulty_idx = index2[220:]
	# train_faulty,valid_faulty,test_faulty = data_normalized2[train_faulty_idx,:,:],data_normalized2[valid_faulty_idx,:,:],data_normalized2[test_faulty_idx,:,:]

	# faulty_labels_train = []
	# faulty_labels_valid = []
	# faulty_labels_test = []
	# for i in train_faulty_idx:
	# 	faulty_labels_train.append(faulty_labels[i])
	# for i in valid_faulty_idx:
	# 	faulty_labels_valid.append(faulty_labels[i])
	# for i in test_faulty_idx:
	# 	faulty_labels_test.append(faulty_labels[i])

	train_classifier = np.concatenate((train_faulty1,train_faulty2,train_faulty3,train_auto),axis = 0)
	train_classifier_labels = faulty_train_labels1+faulty_train_labels2+faulty_train_labels3+health_labels_train

	valid_data = np.concatenate((valid_faulty1,valid_faulty2,valid_faulty3, valid_health), axis=0)
	valid_data_labels = faulty_valid_labels1+faulty_valid_labels2+faulty_valid_labels3 + healthy_labels_valid

	test_data = np.concatenate((test_faulty,test_health),axis = 0)
	test_labels = faulty_test_labels+health_labels_test

	return (train_auto,health_labels_train,valid_auto,auto_labels_valid,train_classifier,train_classifier_labels,valid_data,valid_data_labels,test_data,test_labels)

train_auto,health_labels_train,valid_auto,auto_labels_valid, train_classifier,train_classifier_labels,valid_data,valid_data_labels,test_data,test_labels = data_preparation(path_healthy,path1_fault4_L2K6_7_94689,path1_fault2_L1K1K5_94689,path1_fault3_L2K3_94689,path1_fault4_L2K6_7_94689)

print(f"{train_auto.shape},{len(health_labels_train)},{health_labels_train.count(1)}")
print(f" autoencoder validation data = {len(valid_data_labels)}\n")
print(f"\n train data shape:{train_classifier.shape},{len(train_classifier_labels)},\n zeros label in train data: {train_classifier_labels.count(0)},\n ones in train data :{train_classifier_labels.count(1)}")
print(f"\n validation data shape:{valid_data.shape},{len(valid_data_labels)},\n zeros in valid data: {valid_data_labels.count(0)},\n ones in valid data: {valid_data_labels.count(1)}")
print(f"\n test data shape:{test_data.shape},{len(test_labels)},\n zeros in test data: {test_labels.count(0)},\n ones in test data:{test_labels.count(1)}")

