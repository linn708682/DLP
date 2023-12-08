import os
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from evaluator import evaluation_model

label_dim = 24
img_size = 64

def get_train_data():
    
    # load objects.json
    file_address = "objects.json"
    with open(file_address, 'r') as f:
            object_dict = json.loads(f.read())
    f.close()

    file_address = "train.json"
    with open(file_address, 'r') as f:
        train_dict = json.loads(f.read())
    f.close() 

    N = len(train_dict)
    img_name = np.empty(N, dtype=object)
    label = np.zeros([N, label_dim], dtype=int)

    idx = 0
    for img_id in train_dict:            
        img_name[idx] = img_id
        
        items = train_dict[img_id]
        for item in items:
            label[idx][object_dict[item]] = 1

        idx += 1           

    return img_name, label

def get_test_data():
    
    # load objects.json
    file_address = "objects.json"
    with open(file_address, 'r') as f:
            object_dict = json.loads(f.read())
    f.close()

    file_address = "new_test.json"
    with open(file_address, 'r') as f:
        test_dict = json.loads(f.read())
    f.close() 

    N = len(test_dict)
    label = np.zeros([N, label_dim], dtype=int)

    idx = 0
    for label_content in test_dict:     

        for item in label_content:
            label[idx][object_dict[item]] = 1

        idx += 1           

    return label  


def transform_func():
    return transforms.Compose([
        transforms.Resize([img_size,img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std= [0.5, 0.5, 0.5])
        ])


class iClevr_Loader(data.Dataset):
    def __init__(self, root):
        """
        Args:
            root (string): Root path of the dataset.

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = get_train_data()
        self.transform = transform_func()
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """
           step1. Get the image path from 'self.img_name' and load it.
           step2. Get the ground truth label from self.label
           step3. Transpose the image shape from [H, W, C] to [C, H, W]                         
           step4. Return processed image and label
        """
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.root, self.img_name[index])

        raw_img = Image.open(img_name).convert("RGB")
        img = self.transform(raw_img)        
        label = self.label[index]

        return {'img': img, 'label': label}
    

if __name__ == '__main__':
    test_label = get_test_data()

    root = "./train_data/"
   # root = "..\\train_data"
    batch_size = 100
    dataset = iClevr_Loader(root)
    data_loader = data.DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers = 4)

    tester = evaluation_model()

    # example of data loading
    for i_batch, sampled_batch in enumerate(data_loader):
        inputs = sampled_batch['img']
        labels = sampled_batch['label']

        # *** show example of input images ***
        a = inputs[0].numpy().transpose((1, 2, 0)) + 0.5
        plt.figure()
        plt.imshow(a)
        plt.title('class: ' + str(labels[0].numpy()))
        plt.show()
        print(inputs.shape)
        
        # n = len(sampled_batch["img"])
        acc = tester.eval(inputs.cuda(), labels)
        print('accuracy: '+ str(acc))

        if i_batch >= 5:
            break
