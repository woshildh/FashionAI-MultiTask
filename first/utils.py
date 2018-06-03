'''
定义一些关于训练、验证中的记录操作
定义模型可视化的操作
'''
#from graphviz import Digraph  
import torch
from Network import MyDenseNet
import config

def make_dot(var, params=None):  
	""" Produces Graphviz representation of PyTorch autograd graph 
	Blue nodes are the Variables that require grad, orange are Tensors 
	saved for backward in torch.autograd.Function 
	Args: 
		var: output Variable 
		params: dict of (name, Variable) to add names to node that 
			require grad (TODO: make optional) 
	"""  
	if params is not None:  
		assert isinstance(params.values()[0], Variable)  
		param_map = {id(v): k for k, v in params.items()}  
  
	node_attr = dict(style='filled',  
					 shape='box',  
					 align='left',  
					 fontsize='12',  
					 ranksep='0.1',  
					 height='0.2')  
	dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))  
	seen = set()  
  
	def size_to_str(size):  
		return '('+(', ').join(['%d' % v for v in size])+')'  
  
	def add_nodes(var):  
		if var not in seen:  
			if torch.is_tensor(var):  
				dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')  
			elif hasattr(var, 'variable'):  
				u = var.variable  
				name = param_map[id(u)] if params is not None else ''  
				node_name = '%s\n %s' % (name, size_to_str(u.size()))  
				dot.node(str(id(var)), node_name, fillcolor='lightblue')  
			else:  
				dot.node(str(id(var)), str(type(var).__name__))  
			seen.add(var)  
			if hasattr(var, 'next_functions'):  
				for u in var.next_functions:  
					if u[0] is not None:  
						dot.edge(str(id(u[0])), str(id(var)))  
						add_nodes(u[0])  
			if hasattr(var, 'saved_tensors'):  
				for t in var.saved_tensors:  
					dot.edge(str(id(t)), str(id(var)))  
					add_nodes(t)  
	for x in var:
		add_nodes(x.grad_fn)  
	return dot

def model_viz():
	'''
	使模型可视化
	'''
	net=MyDenseNet(name_num_dict=config.name_num_dict)
	x=torch.autograd.Variable(torch.rand(1,3,config.target_size[0],
		config.target_size[1]))
	y=net(x)
	g=make_dot(y)
	g.view()

def log_train(epoch,total_loss,total_loss_list,step,total_acc,val_total_loss,
	val_total_loss_list,val_total_acc,val_step):
	'''
	记录每一轮训练的损失
	params:
		epoch:第几轮
		total_loss:总的损失
		total_loss_list:总的损失list
		step:epoch经历了多少step
		total_acc:训练过程中的total_acc

		val_total_loss:验证时的总损失
		val_total_loss_list:验证时各个大类的总损失
		val_total_acc:验证时的准确率
		val_step:验证步数
	'''
	with open(config.train_log_path,"a",encoding="utf-8") as file:
		for i in range(len(total_loss_list)):
			total_loss_list[i]=str(total_loss_list[i])
			val_total_loss_list[i]=str(val_total_loss_list[i])
		content=",".join([str(epoch),str(total_loss/step),str(total_acc*10/step)]
			+total_loss_list+[str(val_total_loss/val_step),
			str(val_total_acc/val_step)]+val_total_loss_list)+"\n"
		file.write(content)



if __name__=="__main__":
	log_train=log_train(0,1563.25,[128,268,237,417,128,32,104,79],2000,300,1500,
		[128,268,237,417,128,32,104,79],500,2000)


