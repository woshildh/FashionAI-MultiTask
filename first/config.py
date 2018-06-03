num_list=[8,5,5,5,10,6,6,9]
#定义大类别的数目
name_num_dict={"coat_length_labels":8,"collar_design_labels":5,
				"lapel_design_labels":5,"neck_design_labels":5,
				"neckline_design_labels":10,"pant_length_labels":6,
				"skirt_length_labels":6,"sleeve_length_labels":9}

#定义大类别所在的位置
name_pos_dict={"coat_length_labels":0,"collar_design_labels":1,
				"lapel_design_labels":2,"neck_design_labels":3,
				"neckline_design_labels":4,"pant_length_labels":5,
				"skirt_length_labels":6,"sleeve_length_labels":7}
#定义数据读取部分的参数
target_size=(224,224)


#定义训练部分的参数

all_weights_path=None
cnn_weights_path="./weights/pretrained/densenet161.pth"
save_weights_path="./weights/trained/multitask_{}.pkl"

train_log_path="./log/train_log.csv"

start_epoch=9
batch_size=16
num_thread=4
epochs=10

train_img_root_path="../../ali_data/second_train_data/train/Images/"
train_img_csv="../../ali_data/second_train_data/labels.csv"

#定义验证部分的参数
val_img_root_path="../../ali_data/second_validate_data/Images/"
val_img_csv="../../ali_data/second_validate_data/labels.csv"
error_log_csv="./log/errors.csv"
val_info_path="./log/val_info.txt"

#定义测试时的参数
question_csv="../../ali_data/second_test_data/week-rank//Tests/question.csv"
answer_csv="./answer.csv"
test_img_path="../../ali_data/second_test_data/week-rank/Images"
test_weights_path="./weights/trained/multitask_21.pkl"

if __name__=="__main__":
	print(name_num_dict)


