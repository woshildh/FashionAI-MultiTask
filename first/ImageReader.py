'''
定义数据读取相关的操作
'''
import torch,os,torchvision,cv2,time
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as data  
from PIL import Image  
from sklearn.utils import shuffle
import numpy as np
import config

def default_loader(img_path):
	return cv2.imread(img_path)

class MultiTaskImageReader(data.Dataset):
	'''
	一个多任务图像读取类:
	params:
		root_path:Image Folder root path
		csv_path:labels.csv path
		transform:transform method
		name_pos_dict: a dictionary, {"coat":1,"lapel":2,``````}
		name_num_dict: a dictionary, {"coat":8,"lapel":9,``````}
	'''
	def __init__(self,root_path,csv_path,name_pos_dict,name_num_dict,num_list,
		transform=None,loader=default_loader,target_size=(224,224)):
		super(MultiTaskImageReader,self).__init__()
		self.num_list=num_list
		self.root_path=root_path
		self.csv_path=csv_path
		self.name_pos_dict=name_pos_dict
		self.name_num_dict=name_num_dict
		self.loader=default_loader
		self.transform=transform  #不知道为什么无法工作
		self.target_size=target_size
		self.get_img_path_list()
	def __len__(self):
		return len(self.img_path_list)

	def __getitem__(self,index):
		item=self.img_path_list[index]
		img_path=os.path.join(self.root_path,item[0],item[1],item[2])
		labels=self.get_label(item)  #获取labels
		img_arr=self.loader(img_path) #加载图片
		img_arr=cv2.resize(img_arr,self.target_size)
		#print(img_arr.shape)
		l=transforms.ToTensor()
		img_arr=l(img_arr)  #转成tensor
		
		for i in range(len(labels)):
			labels[i]=torch.FloatTensor(labels[i])
		return img_arr,labels

	def get_img_path_list(self):
		'''
		从csv文件中获取图片路径列表
		'''
		self.img_path_list=[]
		with open(self.csv_path,"r",encoding="utf-8") as file:
			lines=file.readlines()
			for line in lines:
				items=line.split(",")[0:3]
				self.img_path_list.append(items)
		for i in range(0,5): #随机打乱5次列表
			self.img_path_list=shuffle(self.img_path_list)
		print("get image list successed...")
	def get_label(self,item):
		big_class_num=int(self.name_pos_dict[item[0]])  #所属的大类别的位置
		small_class_num=int(item[1])  #所属的小类别的位置
		labels=[]
		for x in self.num_list:
			labels.append(np.zeros(shape=(x,),dtype=np.float32))
		labels[big_class_num][small_class_num]=1
		return labels

if __name__=="__main__":
	tran=transforms.Compose([transforms.ToTensor(),
		transforms.Resize((224,224))])
	reader=MultiTaskImageReader(
		root_path="../../ali_data/second_validate_data/Images/",
		csv_path="../../ali_data/second_validate_data/labels.csv",
		transform=tran,name_pos_dict=config.name_pos_dict,
		name_num_dict=config.name_num_dict,num_list=config.num_list)
	trainloader = torch.utils.data.DataLoader(reader, batch_size=32, 
	                                      shuffle=True, num_workers=4)
	t1=time.time()
	for data in trainloader:
		x,y=data
		t2=time.time()
		print(t2-t1,end=" ")
		t1=time.time()
		print(x.shape)
		print(y)
		break
		


