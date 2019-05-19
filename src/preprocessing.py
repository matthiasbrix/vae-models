import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

'''prepro = Prepro()
        for batch_idx, data in enumerate(data_loader.train_loader):
            #asd = prepro.rand_rotate(data[0][0], angles=[0, 360])
            #asd = prepro.det_rotate(data[0][0], angle=90)
            plt.imshow(asd.view(28, 28).numpy())
            break
        without rotate and only decode(z_t): works ok, but test loss is 2x as train loss.
        without rotate but with decode(xz_t):  works quite well.
        with rotate + decode(xz_t):
'''

class Preprocessing():

    def __init__(self):
        theta_1 = 0 #np.random.randint(-180, 181)
        theta_2 = theta_1 + np.random.randint(-30, 31)

    def rand_rotate(self, img, angles):
        transform_img = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(angles),
                transforms.ToTensor()
            ])
        return transform_img(img)

    def det_rotate(self, img, angle):
        img = TF.rotate(transforms.ToPILImage()(img), angle)
        return transforms.ToTensor()(img)