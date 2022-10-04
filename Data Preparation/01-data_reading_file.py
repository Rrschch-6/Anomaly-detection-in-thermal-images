import numpy as np
import pandas as pd
import os
import glob

path1_healthy_94689 = "D:/UNI/caseStudy/IR_Images/IR_Images/healthy/healthy_csvData/94689"
path1_fault1_L1K1_94689 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL1K1/FehlerL1K1/94689"
path1_fault2_L1K1K5_94689 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL1K1K5/FehlerL1K1K5/94689"
path1_fault3_L2K3_94689 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL2K3/FehlerL2K3/94689"
path1_fault4_L2K6_7_94689 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL2K6-7/FehlerL2K6-7/94689"

path2_healthy_94693 = "D:/UNI/caseStudy/IR_Images/IR_Images/healthy/healthy_csvData/94693"
path2_fault1_L1K1_94693 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL1K1/FehlerL1K1/94693"
path2_fault2_L1K1K5_94693 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL1K1K5/FehlerL1K1K5/94693"
path2_fault3_L2K3_94693 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL2K3/FehlerL2K3/94693"
path2_fault4_L2K6_7_94693 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL2K6-7/FehlerL2K6-7/94693"



path3_healthy_94706 = "D:/UNI/caseStudy/IR_Images/IR_Images/healthy/healthy_csvData/94706"
path3_fault1_L1K1_94706 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL1K1/FehlerL1K1/94706"
path3_fault2_L1K1K5_94706 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL1K1K5/FehlerL1K1K5/94706"
path3_fault3_L2K3_94706 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL2K3/FehlerL2K3/94706"
path3_fault4_L2K6_7_94706 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL2K6-7/FehlerL2K6-7/94706"


path4_healthy_94716 = "D:/UNI/caseStudy/IR_Images/IR_Images/healthy/healthy_csvData/94716"
path4_fault1_L1K1_94716 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL1K1/FehlerL1K1/94716"
path4_fault2_L1K1K5_94716 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL1K1K5/FehlerL1K1K5/94716"
path4_fault3_L2K3_94716 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL2K3/FehlerL2K3/94716"
path4_fault4_L2K6_7_94716 = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL2K6-7/FehlerL2K6-7/94716"



def cam1_94689(path_heal,path1,path2,path3,path4):

	#1. camera 94689 file path
	filenames_healthy_94689 = glob.glob(os.path.join(path1_healthy_94689, "*.xls"))
	filenames_fault1_L1K1_94689 = glob.glob(os.path.join(path1_fault1_L1K1_94689, "*.xls"))
	filenames_fault2_L1K1K5_94689 = glob.glob(os.path.join(path1_fault2_L1K1K5_94689, "*.xls"))
	filenames_fault3_L2K3_94689 = glob.glob(os.path.join(path1_fault3_L2K3_94689, "*.xls"))
	filenames_fault4_L2K6_7_94689 = glob.glob(os.path.join(path1_fault4_L2K6_7_94689, "*.xls"))

	#### 1. camera 94689

	healthy_94689 = np.empty((32, 32))
	for i, filename in enumerate(filenames_healthy_94689[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		healthy_94689 = np.dstack((healthy_94689, image.values))  # stack in depth
	healthy_94689 = healthy_94689[:, :, 1:]
	healthy_94689 = np.transpose(healthy_94689, (2, 0, 1))
	np.save("healthy_94689.npy", healthy_94689)

	fault1_L1K1_94689 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault1_L1K1_94689[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault1_L1K1_94689 = np.dstack((fault1_L1K1_94689, image.values))  # stack in depth
	fault1_L1K1_94689 = fault1_L1K1_94689[:, :, 1:]
	fault1_L1K1_94689 = np.transpose(fault1_L1K1_94689, (2, 0, 1))
	np.save("fault1_L1K1_94689.npy", fault1_L1K1_94689)

	fault2_L1K1K5_94689 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault2_L1K1K5_94689[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault2_L1K1K5_94689 = np.dstack((fault2_L1K1K5_94689, image.values))  # stack in depth
	fault2_L1K1K5_94689 = fault2_L1K1K5_94689[:, :, 1:]
	fault2_L1K1K5_94689 = np.transpose(fault2_L1K1K5_94689, (2, 0, 1))
	np.save("fault2_L1K1K5_94689.npy", fault2_L1K1K5_94689)

	fault3_L2K3_94689 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault3_L2K3_94689[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault3_L2K3_94689 = np.dstack((fault3_L2K3_94689, image.values))  # stack in depth
	fault3_L2K3_94689 = fault3_L2K3_94689[:, :, 1:]
	fault3_L2K3_94689 = np.transpose(fault3_L2K3_94689, (2, 0, 1))
	np.save("fault3_L2K3_94689.npy", fault3_L2K3_94689)

	fault4_L2K6_7_94689 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault4_L2K6_7_94689[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault4_L2K6_7_94689 = np.dstack((fault4_L2K6_7_94689, image.values))  # stack in depth
	fault4_L2K6_7_94689 = fault4_L2K6_7_94689[:, :, 1:]
	fault4_L2K6_7_94689 = np.transpose(fault4_L2K6_7_94689, (2, 0, 1))
	np.save("fault4_L2K6_7_94689.npy", fault4_L2K6_7_94689)

	all_faulty_94689 = np.concatenate(
		(fault1_L1K1_94689, fault2_L1K1K5_94689, fault3_L2K3_94689, fault4_L2K6_7_94689), axis=0)
	np.save("all_faulty_94689.npy", all_faulty_94689)

def cam2_94693(path_heal, path1, path2, path3, path4):
	# camera 94693 file path
	filenames_healthy_94693 = glob.glob(os.path.join(path2_healthy_94693, "*.xls"))
	filenames_fault1_L1K1_94693 = glob.glob(os.path.join(path2_fault1_L1K1_94693, "*.xls"))
	filenames_fault2_L1K1K5_94693 = glob.glob(os.path.join(path2_fault2_L1K1K5_94693, "*.xls"))
	filenames_fault3_L2K3_94693 = glob.glob(os.path.join(path2_fault3_L2K3_94693, "*.xls"))
	filenames_fault4_L2K6_7_94693 = glob.glob(os.path.join(path2_fault4_L2K6_7_94693, "*.xls"))

	#### 2. camera 94693

	healthy_94693 = np.empty((32, 32))
	for i, filename in enumerate(filenames_healthy_94693[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		healthy_94693 = np.dstack((healthy_94693, image.values))  # stack in depth
	healthy_94693 = healthy_94693[:, :, 1:]
	healthy_94693 = np.transpose(healthy_94693, (2, 0, 1))
	np.save("healthy_94693.npy", healthy_94693)

	fault1_L1K1_94693 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault1_L1K1_94693[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault1_L1K1_94693 = np.dstack((fault1_L1K1_94693, image.values))  # stack in depth
	fault1_L1K1_94693 = fault1_L1K1_94693[:, :, 1:]
	fault1_L1K1_94693 = np.transpose(fault1_L1K1_94693, (2, 0, 1))
	np.save("fault1_L1K1_94693.npy", fault1_L1K1_94693)

	fault2_L1K1K5_94693 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault2_L1K1K5_94693[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault2_L1K1K5_94693 = np.dstack((fault2_L1K1K5_94693, image.values))  # stack in depth
	fault2_L1K1K5_94693 = fault2_L1K1K5_94693[:, :, 1:]
	fault2_L1K1K5_94693 = np.transpose(fault2_L1K1K5_94693, (2, 0, 1))
	np.save("fault2_L1K1K5_94693.npy", fault2_L1K1K5_94693)

	fault3_L2K3_94693 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault3_L2K3_94693[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault3_L2K3_94693 = np.dstack((fault3_L2K3_94693, image.values))  # stack in depth
	fault3_L2K3_94693 = fault3_L2K3_94693[:, :, 1:]
	fault3_L2K3_94693 = np.transpose(fault3_L2K3_94693, (2, 0, 1))
	np.save("fault3_L2K3_94693.npy", fault3_L2K3_94693)

	fault4_L2K6_7_94693 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault4_L2K6_7_94693[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault4_L2K6_7_94693 = np.dstack((fault4_L2K6_7_94693, image.values))  # stack in depth
	fault4_L2K6_7_94693 = fault4_L2K6_7_94693[:, :, 1:]
	fault4_L2K6_7_94693 = np.transpose(fault4_L2K6_7_94693, (2, 0, 1))
	np.save("fault4_L2K6_7_94693.npy", fault4_L2K6_7_94693)

	all_faulty_94693 = np.concatenate(
		(fault1_L1K1_94693, fault2_L1K1K5_94693, fault3_L2K3_94693, fault4_L2K6_7_94693), axis=0)
	np.save("all_faulty_94693.npy", all_faulty_94693)


def cam3_94706(path_heal, path1, path2, path3, path4):
	# camera 94706 file path
	filenames_healthy_94706 = glob.glob(os.path.join(path3_healthy_94706, "*.xls"))
	filenames_fault1_L1K1_94706 = glob.glob(os.path.join(path3_fault1_L1K1_94706, "*.xls"))
	filenames_fault2_L1K1K5_94706 = glob.glob(os.path.join(path3_fault2_L1K1K5_94706, "*.xls"))
	filenames_fault3_L2K3_94706 = glob.glob(os.path.join(path3_fault3_L2K3_94706, "*.xls"))
	filenames_fault4_L2K6_7_94706 = glob.glob(os.path.join(path3_fault4_L2K6_7_94706, "*.xls"))

	#### 3. camera 94706

	healthy_94706 = np.empty((32, 32))
	for i, filename in enumerate(filenames_healthy_94706[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		healthy_94706 = np.dstack((healthy_94706, image.values))  # stack in depth
	healthy_94706 = healthy_94706[:, :, 1:]
	healthy_94706 = np.transpose(healthy_94706, (2, 0, 1))
	np.save("healthy_94706.npy", healthy_94706)

	fault1_L1K1_94706 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault1_L1K1_94706[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault1_L1K1_94706 = np.dstack((fault1_L1K1_94706, image.values))  # stack in depth
	fault1_L1K1_94706 = fault1_L1K1_94706[:, :, 1:]
	fault1_L1K1_94706 = np.transpose(fault1_L1K1_94706, (2, 0, 1))
	np.save("fault1_L1K1_94706.npy", fault1_L1K1_94706)

	fault2_L1K1K5_94706 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault2_L1K1K5_94706[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault2_L1K1K5_94706 = np.dstack((fault2_L1K1K5_94706, image.values))  # stack in depth
	fault2_L1K1K5_94706 = fault2_L1K1K5_94706[:, :, 1:]
	fault2_L1K1K5_94706 = np.transpose(fault2_L1K1K5_94706, (2, 0, 1))
	np.save("fault2_L1K1K5_94706.npy", fault2_L1K1K5_94706)

	fault3_L2K3_94706 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault3_L2K3_94706[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault3_L2K3_94706 = np.dstack((fault3_L2K3_94706, image.values))  # stack in depth
	fault3_L2K3_94706 = fault3_L2K3_94706[:, :, 1:]
	fault3_L2K3_94706 = np.transpose(fault3_L2K3_94706, (2, 0, 1))
	np.save("fault3_L2K3_94706.npy", fault3_L2K3_94706)

	fault4_L2K6_7_94706 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault4_L2K6_7_94706[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault4_L2K6_7_94706 = np.dstack((fault4_L2K6_7_94706, image.values))  # stack in depth
	fault4_L2K6_7_94706 = fault4_L2K6_7_94706[:, :, 1:]
	fault4_L2K6_7_94706 = np.transpose(fault4_L2K6_7_94706, (2, 0, 1))
	np.save("fault4_L2K6_7_94706.npy", fault4_L2K6_7_94706)


	all_faulty_94706 = np.concatenate(
		(fault1_L1K1_94706, fault2_L1K1K5_94706, fault3_L2K3_94706, fault4_L2K6_7_94706), axis=0)
	np.save("all_faulty_94706.npy", all_faulty_94706)


def cam4_94716(path_heal, path1, path2, path3, path4):
	# camera 94716 file path
	filenames_healthy_94716 = glob.glob(os.path.join(path4_healthy_94716, "*.xls"))
	filenames_fault1_L1K1_94716 = glob.glob(os.path.join(path4_fault1_L1K1_94716, "*.xls"))
	filenames_fault2_L1K1K5_94716 = glob.glob(os.path.join(path4_fault2_L1K1K5_94716, "*.xls"))
	filenames_fault3_L2K3_94716 = glob.glob(os.path.join(path4_fault3_L2K3_94716, "*.xls"))
	filenames_fault4_L2K6_7_94716 = glob.glob(os.path.join(path4_fault4_L2K6_7_94716, "*.xls"))

	#### 4. camera 94716

	healthy_94716 = np.empty((32, 32))
	for i, filename in enumerate(filenames_healthy_94716[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		healthy_94716 = np.dstack((healthy_94716, image.values))  # stack in depth
	healthy_94716 = healthy_94716[:, :, 1:]
	healthy_94716 = np.transpose(healthy_94716, (2, 0, 1))
	np.save("healthy_94716.npy", healthy_94716)

	fault1_L1K1_94716 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault1_L1K1_94716[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault1_L1K1_94716 = np.dstack((fault1_L1K1_94716, image.values))  # stack in depth
	fault1_L1K1_94716 = fault1_L1K1_94716[:, :, 1:]
	fault1_L1K1_94716 = np.transpose(fault1_L1K1_94716, (2, 0, 1))
	np.save("fault1_L1K1_94716.npy", fault1_L1K1_94716)

	fault2_L1K1K5_94716 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault2_L1K1K5_94716[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault2_L1K1K5_94716 = np.dstack((fault2_L1K1K5_94716, image.values))  # stack in depth
	fault2_L1K1K5_94716 = fault2_L1K1K5_94716[:, :, 1:]
	fault2_L1K1K5_94716 = np.transpose(fault2_L1K1K5_94716, (2, 0, 1))
	np.save("fault2_L1K1K5_94716.npy", fault2_L1K1K5_94716)

	fault3_L2K3_94716 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault3_L2K3_94716[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault3_L2K3_94716 = np.dstack((fault3_L2K3_94716, image.values))  # stack in depth
	fault3_L2K3_94716 = fault3_L2K3_94716[:, :, 1:]
	fault3_L2K3_94716 = np.transpose(fault3_L2K3_94716, (2, 0, 1))
	np.save("fault3_L2K3_94716.npy", fault3_L2K3_94716)

	fault4_L2K6_7_94716 = np.empty((32, 32))
	for i, filename in enumerate(filenames_fault4_L2K6_7_94716[:-1]):
		image = pd.read_csv(filename, engine='python', sep='\t', header=None)
		image = image.replace({',': ''}, regex=True).astype(int)
		fault4_L2K6_7_94716 = np.dstack((fault4_L2K6_7_94716, image.values))  # stack in depth
	fault4_L2K6_7_94716 = fault4_L2K6_7_94716[:, :, 1:]
	fault4_L2K6_7_94716 = np.transpose(fault4_L2K6_7_94716, (2, 0, 1))
	np.save("fault4_L2K6_7_94716.npy", fault4_L2K6_7_94716)

	all_faulty_94716 = np.concatenate(
		(fault1_L1K1_94716, fault2_L1K1K5_94716, fault3_L2K3_94716, fault4_L2K6_7_94716), axis=0)
	np.save("all_faulty_94716.npy", all_faulty_94716)


cam1_94689(path1_healthy_94689,path1_fault1_L1K1_94689,path1_fault2_L1K1K5_94689,path1_fault3_L2K3_94689,path1_fault4_L2K6_7_94689)
cam2_94693(path2_healthy_94693,path2_fault1_L1K1_94693,path2_fault2_L1K1K5_94693,path2_fault3_L2K3_94693,path2_fault4_L2K6_7_94693)
cam3_94706(path3_healthy_94706,path3_fault1_L1K1_94706,path3_fault2_L1K1K5_94706,path3_fault3_L2K3_94706,path3_fault4_L2K6_7_94706)
cam4_94716(path4_healthy_94716,path4_fault1_L1K1_94716,path4_fault2_L1K1K5_94716,path4_fault3_L2K3_94716,path4_fault4_L2K6_7_94716)
# data_preparation(path_healthy_94689)
cam1_np_names = ["healthy_94689.npy", "fault1_L1K1_94689.npy", "fault2_L1K1K5_94689.npy","fault3_L2K3_94689.npy", "fault4_L2K6_7_94689.npy","all_faulty_94689.npy"]
cam2_np_names = ["healthy_94693.npy", "fault1_L1K1_94693.npy", "fault2_L1K1K5_94693.npy","fault3_L2K3_94693.npy", "fault4_L2K6_7_94693.npy","all_faulty_94693.npy"]
cam3_np_names = ["healthy_94706.npy", "fault1_L1K1_94706.npy", "fault2_L1K1K5_94706.npy","fault3_L2K3_94706.npy", "fault4_L2K6_7_94706.npy","all_faulty_94706.npy"]
cam4_np_names = ["healthy_94716.npy", "fault1_L1K1_94716.npy", "fault2_L1K1K5_94716.npy","fault3_L2K3_94716.npy", "fault4_L2K6_7_94716.npy","all_faulty_94716.npy"]
names = cam1_np_names+cam2_np_names+cam3_np_names+cam4_np_names
for name in names:
	x=np.load(name)
	print(x.shape)


