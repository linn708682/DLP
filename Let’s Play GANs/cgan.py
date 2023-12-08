import argparse
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch

from dataloader import iClevr_Loader, get_test_data
from list_all_label import enumerate_all_labels
from evaluator import evaluation_model

class Generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(opt.latent_dim, d*4, 4, 1, 0) #(stride ,padding  ,output_padding ,dilation(controls the spacing between the kernel points) ,groups )
        self.deconv1_1_bn = nn.BatchNorm2d(d*4) #對輸入的批數據進行歸一化，映射到均值為 0 ，方差為 1 的正態分佈。
        
        # nn.ConvTranspose2d(in_channels: int,out_channels: int,kernel_size: _size_2_t,stride: _size_2_t = 1,padding: _size_2_t = 0,)
        self.deconv1_2 = nn.ConvTranspose2d(opt.n_classes, d*4, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*4)
        
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        input = input.view(-1, opt.latent_dim, 1, 1) #參數中的-1就代表這個位置由其他位置的數字來推斷
        label = label.view(-1, opt.n_classes, 1, 1)
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)), 0.2)
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)), 0.2)
       
        x = torch.cat([x, y], 1)

        x = F.relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        x = F.relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = torch.tanh(self.deconv5(x)) #輸出 64 x 64 x 3 的圖像

        return x


class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(3, int(d/2), 4, 2, 1)
        self.conv1_2 = nn.Conv2d(opt.n_classes, int(d/2), 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))        

        return x.view(-1, 1)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def sample_image(n_row, n_col, batches_done, latent_dim):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = FloatTensor(np.random.normal(0, 1, (n_row * n_col, latent_dim)))
    generator.eval()
    with torch.no_grad():
        gen_imgs = generator(z, test_labels)
    save_image(gen_imgs.data, "./output_images/%d.png" % batches_done, nrow=n_col, normalize=True)
    generator.train()
    return gen_imgs


def save_model(epoch):
    tosave = {"generator": generator,
            "discriminator": discriminator,
            "train_hist": train_hist }
    torch.save(tosave, "./model/cDCGAN_" + str(epoch) + "_"+ str(acc) + ".fullmodel")


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=24, help="number of classes for dataset")
opt = parser.parse_args()


if __name__ == '__main__':

    print(opt)

    cuda = True if torch.cuda.is_available() else False
    img_shape = (3, 64, 64)

    # Loss functions 基本上二分法都是用 BCELoss
    adversarial_loss = torch.nn.BCELoss()
    
    generator = Generator()
    discriminator = Discriminator()
    generator.weight_init(mean=0.0, std=0.02)
    discriminator.weight_init(mean=0.0, std=0.02)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Configure data loader
    root = "./train_data/"
    # root = "..\\train_data"
    batch_size = opt.batch_size
    dataset = iClevr_Loader(root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_cpu)
    listed_labels = enumerate_all_labels()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # tester
    tester = evaluation_model()
    test_labels = FloatTensor(get_test_data())

    # label preprocess
    fill = torch.zeros([opt.n_classes, opt.n_classes, img_shape[1], img_shape[2]])
    for i in range(opt.n_classes):
        fill[i, i, :, :] = 1


    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    if_train_D = True    

    for epoch in range(opt.n_epochs):
        D_losses = []
        G_losses = []        

        if (epoch+1) == 30:
            optimizer_G.param_groups[0]['lr'] /= 10
            optimizer_D.param_groups[0]['lr'] /= 10

        for i, sampled_batch in enumerate(dataloader):
            
            imgs = sampled_batch['img']
            labels = sampled_batch['label']
            batch_size = len(imgs)

            # Adversarial ground truths
            valid = FloatTensor(batch_size, 1).fill_(1.0)
            fake = FloatTensor(batch_size, 1).fill_(0.0)

            # Configure input
            real_imgs = imgs.type(FloatTensor)
            labels = labels.type(FloatTensor)

            #real_fill用來做出label  condition為一個24-dim 的 one-hot vector。如: [0,0,0,1,0,0,1,.....0,0,0]
            real_fill = torch.zeros([batch_size, opt.n_classes, img_shape[1], img_shape[2]])
            for idx in range(batch_size):
                real_fill[idx] = torch.sum(fill[labels[idx].type(torch.bool)], 0)
            real_fill = real_fill.type(FloatTensor)

            idx = np.random.randint(0, batch_size, batch_size)
            fake_labels = labels[idx]
            mismatch_fill = real_fill[idx]
            a = torch.sum(fake_labels != labels, 1) == 0
            mismatch_fake = FloatTensor(a.view(-1,1).type(torch.float))
                        

            optimizer_D.zero_grad()

            
            #清晰圖片+正確標籤的loss
            validity_real_correct = discriminator(real_imgs, real_fill)
            d_real_loss_correct = adversarial_loss(validity_real_correct, valid)

            #清晰圖片+錯誤標籤的loss
            validity_real_mismatch = discriminator(real_imgs, mismatch_fill)
            d_real_loss_mismatch = adversarial_loss(validity_real_mismatch, mismatch_fake)
            
            # 生成noise做 generator input            
            z = FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)))
            #不清晰圖片的loss
            gen_imgs = generator(z, labels)
            validity_fake = discriminator(gen_imgs, real_fill)
            d_fake_loss = adversarial_loss(validity_fake, fake)

           
            d_loss = d_real_loss_correct + (d_real_loss_mismatch + d_fake_loss)/2

        

            if if_train_D:
                d_loss.backward()
                optimizer_D.step()

            D_losses.append(d_loss.to(torch.device("cpu")).data.numpy())
            

            if True:


                optimizer_G.zero_grad()

                z = FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)))
              
                gen_imgs = generator(z, labels)

                validity = discriminator(gen_imgs, real_fill)
                g_loss = adversarial_loss(validity, valid)


                g_loss.backward()
                optimizer_G.step()

                G_losses.append(g_loss.to(torch.device("cpu")).data.numpy())

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] D:%d"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), if_train_D)
                )

            batches_done = epoch * len(dataloader) + i

         

            
        gen_img = sample_image(n_row=4, n_col =8, batches_done=batches_done, latent_dim = opt.latent_dim)
        acc = tester.eval(gen_img.cuda(), test_labels)
        print("[Epoch %d/%d] [acc: %f] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, acc, np.mean(D_losses), np.mean(G_losses)))
        train_hist['D_losses'].append(np.mean(D_losses))
        train_hist['G_losses'].append(np.mean(G_losses))
        save_model(epoch)

       