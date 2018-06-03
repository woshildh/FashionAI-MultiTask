from torchvision import models,transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import config,Evaluate,utils,ImageReader,Network,metrics
from torch import optim
use_gpu=True

def train():
	'''
	定义训练的函数
	'''
	#加载模型
	model=Network.MyDenseNet(name_num_dict=config.name_num_dict,
		cnn_weights_path=config.cnn_weights_path)

	#加载全部的模型权重
	if config.all_weights_path:
		weights_dict=torch.load(config.all_weights_path)
		model.load_state_dict(state_dict=weights_dict)
		print("load {} weights succecced...".format(config.all_weights_path))

	if torch.cuda.is_available() and use_gpu:
		model=model.cuda()
	
	#定义训练数据部分
	tran=transforms.Compose([transforms.ToTensor(),
	transforms.Resize(config.target_size)])
	reader=ImageReader.MultiTaskImageReader(
		root_path=config.train_img_root_path,
		csv_path=config.train_img_csv,
		transform=tran,name_pos_dict=config.name_pos_dict,
		name_num_dict=config.name_num_dict,
		num_list=config.num_list)
	trainloader = torch.utils.data.DataLoader(reader, batch_size=config.batch_size, 
	                                      shuffle=True, num_workers=config.num_thread)
	
	#定义验证数据部分
	val_reader=ImageReader.MultiTaskImageReader(
		root_path=config.val_img_root_path,
		csv_path=config.val_img_csv,
		transform=tran,name_pos_dict=config.name_pos_dict,
		name_num_dict=config.name_num_dict,
		num_list=config.num_list)
	validateloader=torch.utils.data.DataLoader(val_reader,batch_size=config.batch_size,
		shuffle=True,num_workers=config.num_thread)

	#定义优化器
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	
	for epoch in range(config.start_epoch,config.start_epoch+config.epochs):
		print("{} epoch start...".format(epoch))
		#开始训练模式
		model.train()
		#定义一轮中的评估指标 
		step=0  #训练步数
		total_loss=0  #记录总的运行损失
		total_loss_list=[0,0,0,0,0,0,0,0]  #记录各个大类的运行损失
		total_acc=0 #记录总的准确率
		
		val_step=0
		val_total_acc=0
		val_total_loss=0
		val_total_loss_list=[0,0,0,0,0,0,0,0]
		for data in trainloader:
			inputs,labels=data
			inputs = Variable(inputs)
			for i in range(len(labels)):
				labels[i]=Variable(labels[i])

			if torch.cuda.is_available() and use_gpu:
				inputs=inputs.cuda()
				for i in range(len(labels)):
					labels[i]=labels[i].cuda()
			optimizer.zero_grad()  #将梯度缓冲区清0
			outputs=model(inputs)
			step_loss,step_loss_list=metrics.loss(outputs,labels)
			
			total_loss+=step_loss.data[0]  #更新总的损失
			for i in range(len(total_loss_list)):  #更新各个类的损失
				total_loss_list[i]+= step_loss_list[i]
			step_loss.backward()
			optimizer.step()
			if step % 10 ==0 and step!=0:
				step_acc=metrics.accuracy(outputs,labels)
				total_acc+=step_acc
				print("{} epoch, {} step, step loss is {}, step acc is {}".format(
					epoch,step,step_loss.data[0],step_acc))
			else:
				print("{} epoch, {} step, step loss is {}".format(
					epoch,step,step_loss.data[0]))
			#释放显存
			del([inputs,labels,outputs,step_loss,step_loss_list])

			step+=1
		#模型保存参数
		print("start save model weights...")
		torch.save(model.state_dict(),
			config.save_weights_path.format(epoch))
		
		model.eval() #开启验证模式
		print("start validate ...")
		#解下来进行验证
		for data in validateloader:
			inputs,labels=data
			inputs=Variable(inputs)
			for i in range(len(labels)):
				labels[i]=Variable(labels[i])
			if torch.cuda.is_available() and use_gpu:
				inputs=inputs.cuda()
				for i in range(len(labels)):
					labels[i]=labels[i].cuda()
			#获取outputs
			outputs=model(inputs)
			step_loss,step_loss_list=metrics.loss(outputs,labels)
			step_acc=metrics.accuracy(outputs,labels)

			val_total_loss+=step_loss.data[0]
			for i in range(len(val_total_loss_list)):
				val_total_loss_list[i]+=step_loss_list[i]
			val_total_acc+=step_acc
			val_step+=1

			#释放显存
			del([inputs,labels,outputs,step_loss,step_loss_list])

		print("{} epoch , validation loss is {}, validation accuracy is {}".format(
			epoch,val_total_loss/val_step,val_total_acc/val_step))

		#训练完一轮之后将训练结果记录到文件中
		print("start log train info...")
		utils.log_train(epoch,total_loss,total_loss_list,
			step,total_acc,val_total_loss,val_total_loss_list,
			val_total_acc,val_step)
	del(model)	
	print("{} epoch train end ...".format(epoch))


if __name__=="__main__":
	train()

