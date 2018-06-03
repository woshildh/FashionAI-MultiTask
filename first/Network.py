'''
定义网络结构
'''
from torchvision import models
import torch
import torch.nn as nn

class MyDenseNet(nn.Module):
	'''
	使用预训练的DenseNet_161,定义自己的多任务DenseNet.
	默认的target_size=(224,224)
	'''
	def __init__(self,name_num_dict,cnn_weights_path=None):
		'''
		初始化函数
		'''
		super(MyDenseNet,self).__init__()
		self.name_num_dict=name_num_dict
		self.cnn_weights_path=cnn_weights_path
		self.base_model=self.__getbase__()

		'''获取8个模型并联层'''
		#coat
		self.fc1_1=nn.Linear(in_features=1024,out_features=512)
		self.fc1_2=nn.Linear(in_features=512,out_features=8)
		#collar
		self.fc2_1=nn.Linear(in_features=1024,out_features=512)
		self.fc2_2=nn.Linear(in_features=512,out_features=5)
		#lapel
		self.fc3_1=nn.Linear(in_features=1024,out_features=512)
		self.fc3_2=nn.Linear(in_features=512,out_features=5)
		#neck
		self.fc4_1=nn.Linear(in_features=1024,out_features=512)
		self.fc4_2=nn.Linear(in_features=512,out_features=5)
		#neckline
		self.fc5_1=nn.Linear(in_features=1024,out_features=512)
		self.fc5_2=nn.Linear(in_features=512,out_features=10)
		#pant
		self.fc6_1=nn.Linear(in_features=1024,out_features=512)
		self.fc6_2=nn.Linear(in_features=512,out_features=6)
		#skirt
		self.fc7_1=nn.Linear(in_features=1024,out_features=512)
		self.fc7_2=nn.Linear(in_features=512,out_features=6)
		#sleeve
		self.fc8_1=nn.Linear(in_features=1024,out_features=512)
		self.fc8_2=nn.Linear(in_features=512,out_features=9)
		
		self.drop_1=nn.Dropout(p=0.5)

	def forward(self,x):
		'''
		前向传播
		'''
		x=nn.functional.relu(self.base_model(x))
		x=self.drop_1(x)
		#coat
		res1=nn.functional.relu(self.fc1_1(x))
		res1=self.fc1_2(res1)
		#collar
		res2=nn.functional.relu(self.fc2_1(x))
		res2=self.fc2_2(res2)
		#lapel
		res3=nn.functional.relu(self.fc3_1(x))
		res3=self.fc3_2(res3)
		#neck
		res4=nn.functional.relu(self.fc4_1(x))
		res4=self.fc4_2(res4)
		#neckline
		res5=nn.functional.relu(self.fc5_1(x))
		res5=self.fc5_2(res5)
		#pant
		res6=nn.functional.relu(self.fc6_1(x))
		res6=self.fc6_2(res6)
		#skirt
		res7=nn.functional.relu(self.fc7_1(x))
		res7=self.fc7_2(res7)
		#sleeve
		res8=nn.functional.relu(self.fc8_1(x))
		res8=self.fc8_2(res8)

		return [res1,res2,res3,res4,res5,res6,res7,res8] 
	def __getbase__(self):
		'''
		获取CNN卷积部分模型
		'''
		if self.cnn_weights_path:  #仅仅加载卷积部分权重
			base_model=models.densenet161(pretrained=False)
			#加载CNN部分的权重
			weights_dict=torch.load(self.cnn_weights_path)
			base_model.load_state_dict(weights_dict,strict=False)
			base_model.classifier=torch.nn.Linear(in_features=2208,
				out_features=1024)
		else:
			base_model=models.densenet161(pretrained=False)
			base_model.classifier=torch.nn.Linear(in_features=2208,
				out_features=1024)
		return base_model 

if __name__=="__main__":
	d=MyDenseNet({"a":5},
		cnn_weights_path="./weights/pretrained/densenet161.pth")
	img=torch.autograd.Variable(torch.rand(1,3,224,224))
	res=d(img)
	for x in res:
		print(x.shape)


