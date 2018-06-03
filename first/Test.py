import torch
from torchvision import transforms
import Network
import config
import os,cv2,time
import numpy as np 

def test(vis=True):
	'''
	定义测试函数
	vis控制测试时是否输出答案
	'''
	model=Network.MyDenseNet(name_num_dict=config.name_num_dict)

	#加载测试部分的权重
	try:
		weights_dict=torch.load(config.test_weights_path)
		model.load_state_dict(state_dict=weights_dict)
		print("load {} weights succecced...".format(config.test_weights_path))
	except:
		print("load weights failed...")
		return 

	#将model设置为验证模式
	model.eval()
	
	#设置转换
	l=transforms.ToTensor()
	
	with open(config.question_csv,"r",encoding="utf-8") as file:
		lines=file.readlines()[13784:]

	#遍历图片进行测试
	for i,line in enumerate(lines):
		img_name=line.split(",")[0].split("/")[-1]
		big_class=line.split(",")[1]
		img_path=os.path.join(config.test_img_path,big_class,img_name)

		pos=config.name_pos_dict[big_class]
		num=config.name_num_dict[big_class]
		img_arr=cv2.imread(img_path)
		img_arr=cv2.resize(img_arr,(224,224))
		img_arr=l(img_arr)
		img_arr=torch.autograd.Variable(img_arr.view(1,3,224,224))
		res=model(img_arr)
		
		res=torch.nn.functional.softmax(res[pos],dim=1)

		if vis:
			value,small_class=torch.max(res,dim=1)
			print("{} ,img is {} ,big class is {} ,small class is {}".format(
				i,img_name,big_class,int(small_class[0])))

		labels=[]
		for ele in res.data[0]:
			labels.append(str(float(ele)))
		labels=";".join(labels)

		content=line.split(",")[0]+","+big_class+","+labels+'\n'
		with open(config.answer_csv,"a",encoding="utf-8") as file:
			file.write(content)

if __name__=="__main__":
	test()

