from dataloader import read_bci_data
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import matplotlib.pyplot as plt

class EEG(nn.Module):
    def __init__(self, act_func):
        super(EEG, self).__init__()

        self.activationDict = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'ELU': nn.ELU(),
        }

        self.firstConv = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,51), stride=(1,1),padding=(0,25), bias=False), #input=1x2x750 output=16x2x750
			nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False), #input=16x2x750 output=32x1x750
			nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
			self.activationDict[act_func],
			nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0), #input=32x1x750 output=32x1x187
			nn.Dropout(p=0.25),
            
        )

        self.separableConv = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,15),stride=(1,1),padding=(0,7), bias=False), #input=32x1x187 output=32x1x187
			nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
			self.activationDict[act_func],
			nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0), #input=32x1x187 output=32x1x23
			nn.Dropout(p=0.25)
         
        )

        self.classifyConv = nn.Sequential(
        	nn.Flatten(), #input=32x1x23 output=736
			nn.Linear(in_features=736,out_features=2,bias=True)
		
          
        )

    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classifyConv(x)
        
        return x



class DeepConvNet(nn.Module):
    def __init__(self, act_func):
        super(DeepConvNet, self).__init__()

        self.activationDict = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'ELU': nn.ELU(),
        }
		
		
        self.doubleConv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1,5)), #input=1x2x750 output=25x2x746
			nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(2,1)), #input=25x2x746 output=25x1x746
			nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
		    self.activationDict[act_func],
		   	nn.MaxPool2d(kernel_size=(1,2)), #input=25x1x746 output=25x1x373
			nn.Dropout(p=0.5)
        )

        self.secondConv = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1,5)), #input=25x1x378 output=50x1x369
			nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
            self.activationDict[act_func],
			nn.MaxPool2d(kernel_size=(1,2)), #input=50x1x369 output=50x1x184
			nn.Dropout(p=0.5)
            
        )

        self.thirdConv = nn.Sequential(
			nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1,5)), #input=50x1x184 output=100x1x180
			nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
			self.activationDict[act_func],
			nn.MaxPool2d(kernel_size=(1,2)), #input=100x1x180 output=100x1x90
			nn.Dropout(p=0.5)
         
        )

        self.fourthConv = nn.Sequential(
          	nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1,5)), #input=100x1x90 output=200x1x86
			nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
			self.activationDict[act_func],
			nn.MaxPool2d(kernel_size=(1,2)), #input=200x1x86 output=200x1x43
			nn.Dropout(p=0.5)
          
        )

        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(in_features=8600,out_features=2)
        )

    def forward(self, x):
        x = self.doubleConv(x)
        x = self.secondConv(x)
        x = self.thirdConv(x)
        x = self.fourthConv(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


i = 0
def train( model, train_data, train_label, optimizer, batchsize):
	global i
	count = 0
	model.train()
	while count<1080:
		data = torch.cuda.FloatTensor( train_data[i:i+batchsize] )
		target = torch.cuda.LongTensor( train_label[i:i+batchsize] )
		#初始化梯度
		optimizer.zero_grad()
		#訓練後的結果
		output = model(data)
		# 計算損失
		loss = F.cross_entropy(output, target)
		#反向傳播
		loss.backward()
		#參數優化
		optimizer.step()

		i = (i+batchsize)%1080
		count += batchsize

def test(model, test_data, test_label):
	
	model.eval()#將模型設為測試模式

	data = torch.cuda.FloatTensor( test_data )
	target = torch.cuda.LongTensor( test_label )
	#測試數據後的結果
	output = model(data)
	#計算並加總測試損失
	test_loss = F.cross_entropy(output, target)
	#找到概率值的最大的結果
	pred = output.argmax(dim=1)  

	#累計正確的值
	correct=pred.eq(target.view_as(pred)).sum().item()	

	return test_loss.item()/1080.0 , correct/1080.0
	
	
Learning_Rate = 1e-3 #學習率
BATCH_SIZE = 64 #每批處理的數據
EPOCHS = 500 #訓練數據集的輪次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #用CPU或GPU來訓練

if __name__ == '__main__':
	torch.manual_seed(1)
	
	train_data, train_label, test_data, test_label = read_bci_data()

	for task in ['EEG', 'DeepConvNet']:
		plt_array_loss = []
		plt_array_accuracy = []
		for act_func in ['ReLU', 'LeakyReLU', 'ELU']:
			for testset in ['train','test']:
				print(str(task+'_'+act_func+'_'+testset))
				plt_array_loss_tmp = []
				plt_array_accuracy_tmp = []

				if testset == 'train':
					m_data, m_label = train_data, train_label
				elif testset == 'test':
					m_data, m_label = test_data, test_label

				if task == 'EEG':
					model = EEG(act_func=act_func)
				elif task == 'DeepConvNet':
					model = DeepConvNet(act_func=act_func)

				model.to(DEVICE)
				optimizer = optim.Adam(model.parameters(),Learning_Rate)

				for epoch in range(1, EPOCHS +1):
					train(model, train_data, train_label, optimizer, BATCH_SIZE)
					test_loss, correct = test(model, m_data, m_label)

					plt_array_accuracy_tmp.append(correct*100)
					plt_array_loss_tmp.append(test_loss)
					if epoch%250 == 0: print('epoch= ',epoch,' loss= ',test_loss,' correct= ',correct)

				plt_array_accuracy.append(plt_array_accuracy_tmp)
				plt_array_loss.append(plt_array_loss_tmp)

		for arr in plt_array_accuracy: plt.plot(arr)
		plt.title(str("Activation Functions comparision ("+task+')'))
		plt.grid()
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy(%)')
		plt.legend(['relu_train', 'relu_test', 'leaky_relu_train', 'leaky_relu_test', 'elu_train', 'elu_test',])
		plt.savefig(str(task+'_accuracy.png'))
		plt.close()
		plt.show()

		for arr in plt_array_loss: plt.plot(arr)
		plt.grid()
		plt.title(str("Learning curve comparision ("+task+')'))
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend(['relu_train', 'relu_test', 'leaky_relu_train', 'leaky_relu_test', 'elu_train', 'elu_test',])
		plt.savefig(str(task+'_loss.png'))
		plt.close()
		plt.show()