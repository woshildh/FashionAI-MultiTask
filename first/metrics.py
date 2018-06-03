'''
定义损失函数、准确率等评估方法
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import ImageReader
from torchvision import transforms
import config,Network
import numpy as np
import time
def loss(pred_labels,target_labels):
	'''
	使用softmax_corss_entropy求出total_loss
	params:
		pred_labels:预测的标签
		target_labels:目标标签
	return:
		total_loss:多个类别的总的损失值
		loss_list：a list, 各个类别的总的损失值
	'''
	total_loss=0
	loss_list=[]
	for i in range(len(pred_labels)):
		pred_labels[i]=F.log_softmax(pred_labels[i],dim=1)
	for i in range(len(pred_labels)):
		l=-pred_labels[i]*target_labels[i]
		small_l=torch.mean(l)
		total_loss+=small_l
		loss_list.append(small_l.data[0])
	return total_loss,loss_list

def accuracy(pred_labels,target_labels):
	'''
	根据pred_labels和target_labels求出总的准确率和各个类别的准确率
	params:
		pred_labels:预测的标签
		target_labels:目标标签
	return:
		total_acc:多个类别的总的准确率
	'''
	big_class_list=[]
	batch_size=int(pred_labels[0].shape[0])
	true_num=0
	for i in range(batch_size):
		for j in range(len(pred_labels)):			
			if float(torch.max(target_labels[j][i]))==1:
				big_class_list.append(j)
	for i in range(batch_size):
		pred_small_value,pred_small_class=torch.max(pred_labels[big_class_list[i]][i].view([1,-1]),dim=1)
		target_small_value,target_small_class=torch.max(target_labels[big_class_list[i]][i].view([1,-1]),dim=1)
		res=(pred_small_class==target_small_class).data[0]
		if res:
			true_num+=1
	return true_num/batch_size

if __name__=="__main__":
	tran=transforms.Compose([transforms.ToTensor(),
		transforms.Resize((224,224))])
	reader=ImageReader.MultiTaskImageReader(
		root_path="../../ali_data/second_validate_data/Images/",
		csv_path="../../ali_data/second_validate_data/labels.csv",
		transform=tran,name_pos_dict=config.name_pos_dict,
		name_num_dict=config.name_num_dict,num_list=config.num_list)
	trainloader = torch.utils.data.DataLoader(reader, batch_size=1, 
	                                      shuffle=True, num_workers=4)
	for x,y in trainloader:
		break
	model=Network.MyDenseNet(config.name_num_dict)
	
	x=torch.autograd.Variable(x)
	output=model(x)
	for i in range(len(output)):
		y[i]=torch.autograd.Variable(y[i])
	#acc=accuracy(output,y)
	l=loss(output,y)
	print(l[0])

