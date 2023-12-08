import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def der_sigmoid(y):
    return y * (1.0 - y)

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def der_tanh(y):
    return 1-tanh(y)*tanh(y)

class GenData:
    
    def __init__(self):
        pass
    
    def generate_linear(n=100):
        pts = np.random.uniform(0,1,(n,2))
        inputs = []
        labels = []
        for pt in pts:
            inputs.append([pt[0],pt[1]])
            if pt[0] > pt[1]:
                labels.append(0)
            else:
                labels.append(1)
        return np.array(inputs), np.array(labels).reshape(n,1)

    def generate_xor_easy(n=100):
        inputs = []
        labels = []
        for i in range(11):
            inputs.append([0.1*i, 0.1*i])
            labels.append(0)
            if 0.1*i == 0.5:
                continue
            inputs.append([0.1*i,1-0.1*i])
            labels.append(1)
        return np.array(inputs),np.array(labels).reshape(21,1)

    def generate_circle(n=100):
        data_x = np.linspace(0, 1, n // 4) * np.pi * 2
        inputs = []
        labels = []
        for x in data_x:
            inputs.append([np.sin(x), np.cos(x)])
            labels.append(0)

            inputs.append([np.sin(x) * 2, np.cos(x) * 2])
            labels.append(0)

            inputs.append([np.sin(x) * 1.5, np.cos(x) * 1.5])
            labels.append(1)

            inputs.append([np.sin(x) * 0.5, np.cos(x) * 0.5])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    def fetch_data(self,mode):    
        assert mode == "Linear" or mode == "XOR" or mode == "Circle"
        data_gen_func = {
            'Linear': GenData.generate_linear,
            'XOR': GenData.generate_xor_easy,
            'Circle': GenData.generate_circle,
        }[mode]
        return data_gen_func()

class SimpleNet:
    def __init__(self, hidden_size, num_step=2000, print_interval=100):
      
        self.num_step = num_step
        self.print_interval = print_interval

        np.random.seed(0)                       # control the random seed for debugging
        self.W1 = np.random.randn(hidden_size[0], 2) 
        self.W2 = np.random.randn(hidden_size[1], hidden_size[0])
        self.W3 = np.random.randn(1, hidden_size[1])
       
        self.b1 = np.ones((hidden_size[0],1))
        self.b2 = np.ones((hidden_size[1],1))
        self.b3 = np.ones((1,1))

        self.loss = 0
        self.grad_W1 = 0
        self.grad_W2 = 0
        self.grad_W3 = 0
        self.grad_b1 = 0
        self.grad_b2 = 0
        self.grad_b3 = 0    

  
    @staticmethod
    def compute_loss(y_pred, labels):
        return np.sum(- labels * np.log(y_pred) - (1-labels) * np.log(1-y_pred))
    
    def forward(self, inputs):
        self.a0 = inputs.T
        self.z1 = np.add( np.dot( self.W1, self.a0 ), self.b1 )
        self.a1 = sigmoid(self.z1)
        self.z2 = np.add( np.dot( self.W2, self.a1 ), self.b2 )
        self.a2 = sigmoid(self.z2)
        self.z3 = np.add( np.dot( self.W3, self.a2 ), self.b3 )
        self.a3 = sigmoid(self.z3)      
        return self.a3.T   
  
    def backward(self):
        self.grad_z3 = self.error.T    
        self.grad_z2 = np.dot (self.W3.T, self.grad_z3) * der_sigmoid(self.a2)
        self.grad_z1 = np.dot (self.W2.T, self.grad_z2) * der_sigmoid(self.a1)

        self.grad_W3 = np.dot(self.grad_z3, self.a2.T)        
        self.grad_W2 = np.dot(self.grad_z2, self.a1.T)                
        self.grad_W1 = np.dot(self.grad_z1, self.a0.T)
        
        self.grad_b3 = np.sum(self.grad_z3)
        self.grad_b2 = np.sum(self.grad_z2)
        self.grad_b1 = np.sum(self.grad_z1)


    def update(self, alpha):       
        self.W1 -= alpha * self.grad_W1
        self.W2 -= alpha * self.grad_W2
        self.W3 -= alpha * self.grad_W3
        self.b1 -= alpha * self.grad_b1
        self.b2 -= alpha * self.grad_b2
        self.b3 -= alpha * self.grad_b3

        self.grad_W1 = 0
        self.grad_W1 = 0
        self.grad_W2 = 0
        self.grad_W3 = 0
        self.grad_b1 = 0
        self.grad_b2 = 0
        self.grad_b3 = 0 
        
    def train(self, inputs, labels, learning_rate = 1, batch_size=500 ):
 
        assert inputs.shape[0] == labels.shape[0]

        record_epoch = []
        record_loss = []
        for epochs in range(self.num_step): #總訓練次數
          
            self.output = self.forward(inputs)
          
            self.error = self.output - labels
            self.backward()                                 
            self.update( alpha=0.001)

            if epochs % self.print_interval == 0:
                print('Epochs {}: '.format(epochs))
                self.test(inputs, labels)

               
                record_epoch.append(epochs)
                record_loss.append(self.loss)

        
        #印出最終的測試結果並繪圖
        print('\n***Training finished***')
        result = self.test(inputs, labels)
        print(result)
        drawPic(data, label, result)

        
        #畫出訓練中的損失函數
        record_epoch = np.array(record_epoch)
        record_loss  = np.array(record_loss)
        plt.figure()
        plt.plot(record_epoch, record_loss, 'b.-')
        plt.xlabel('Epochs')
        plt.ylabel('Total loss')
        plt.show()

    def test(self, inputs, labels):
   
        n = inputs.shape[0]
        error = 0.0
        self.loss = 0
   
        result = self.forward(inputs)

        error = np.sum(abs(result - labels))
        error /= n

        self.loss = np.sum(SimpleNet.compute_loss(result, labels))          
        self.loss /= n

        print(f'accuracy: {(1 - float(error)) :.2%}, loss: {self.loss}')
        return result


def drawPic(data, gt_y, pred_y):
    assert data.shape[0] == pred_y.shape[0]
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize=18)
    for idx in range(data.shape[0]):
        if gt_y[idx] == 0:
            plt.plot(data[idx][0], data[idx][1], 'ro')
        else:
            plt.plot(data[idx][0], data[idx][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Prediction', fontsize=18)

    for idx in range(data.shape[0]):
        if pred_y[idx] < 0.5 :
            plt.plot(data[idx][0], data[idx][1], 'ro')
        else:
            plt.plot(data[idx][0], data[idx][1], 'bo')

    plt.show()

if __name__ == '__main__':
    data, label = GenData().fetch_data('Linear') #Linear XOR Circle
   
    hidden_size = np.array([10,10])
    net = SimpleNet(hidden_size, num_step=100000)
    batch_max = data.shape[0]
    net.train(data, label, learning_rate = 1, batch_size = 1000)

