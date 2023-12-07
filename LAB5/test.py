import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

from cgan import Generator, Discriminator, opt
from evaluator import evaluation_model
from dataloader import iClevr_Loader, get_test_data
from list_all_label import enumerate_all_labels

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':  
    
    model_load_path = "./model/cDCGAN_98_0.625.fullmodel"
    
    FloatTensor = torch.cuda.FloatTensor

    toload = torch.load(model_load_path)
    train_hist = toload["train_hist"]
    
    show_train_hist(train_hist, show = False)    
    
    # generator
    generator = toload["generator"]
    generator.eval()

    # tester
    tester = evaluation_model()
    test_labels = FloatTensor(get_test_data())
    # listed_labels = enumerate_all_labels()
    # test_labels = FloatTensor(listed_labels.get(32))

    max_acc = 0    
    for x in range(20):

        z = FloatTensor(np.random.normal(0, 1, (test_labels.size(0), opt.latent_dim)))    
        with torch.no_grad():
            gen_imgs = generator(z, test_labels)    

        acc = tester.eval(gen_imgs.cuda(), test_labels)    
        if acc > max_acc:
            max_acc = acc
            best_gen_imgs = gen_imgs
    
    print("generated image accuracy: %f" % (max_acc))
    save_image(best_gen_imgs.data, "generated_images.png", normalize=True)
    
    



