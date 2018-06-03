"""
定义一些验证准确率和损失相关的操作
"""
import torch
import torchvision
import cv2
import os
import Network,config
import numpy as np
from torch.autograd import Variable
def evaluate(model_path,img_root_path,img_csv,error_csv):
	'''
	用于记录验证集准确率，各个大类的准确率和损失值等，并且要把错误的预测记录下来
	'''
	model=Network.MyDenseNet(config.name_num_dict)
	model.load_state_dict(torch.load(model_path),strict=True)
	model.eval()

	img_list=[]
	correct_num=0
	with open(img_csv,"r",encoding="utf-8") as file:
		lines=file.readlines()
		for line in lines:
			img_list.append(line.split(",")[0:3])
	print("validation set has {} images".format(len(img_list)))

	for i,img in enumerate(img_list):
		img_path=os.path.join(img_root_path,img[0],img[1],img[2])
		img_arr=cv2.imread(img_path)
		img_arr=cv2.resize(img_arr,config.target_size)
		l=torchvision.transforms.ToTensor()
		img_arr=Variable(l(img_arr).view((1,3,224,224)))
		output=model(img_arr)

		big_class=config.name_pos_dict[img[0]]
		small_class=int(img[1])

		small_output=output[big_class]
		del(output)

		value,index=torch.max(small_output,dim=1)
		del(value)
		del(small_output)

		index=index.data[0]
		if small_class==index:
			 correct_num+=1
			 print("{} is correct".format(i))
		else:
			content=",".join([img[2] , str(big_class) , str(small_class) ,str(index)])+"\n"
			with open(error_csv,"a",encoding="utf-8") as file:
				file.write(content)
			print("{} is error".format(i))
	acc=correct_num/len(img_list)
	print("validate finished ... , validate acc is {}".format(acc))
	return acc

if __name__=="__main__":
	evaluate(config.all_weights_path,config.val_img_root_path,config.val_img_csv,
		config.error_log_csv)

