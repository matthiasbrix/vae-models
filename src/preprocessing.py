import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import torch

'''
for batch_idx, data in enumerate(data_loader.test_loader):
    #asd2 = data[0]
    asd = prepro.scale_rotate_batch(data[0], 2.0, 180) #prepro.rotate_batch(data[0], 180)#prepro.scale_rotate_batch(data[0], 2.0, 180)
    #print(asd.shape, asd[0,0].shape)
#plt.imshow(asd[10,0])
import torchvision
grid_img = torchvision.utils.make_grid(asd)
plt.imshow(grid_img.permute(1, 2, 0))
#grid_img = torchvision.utils.make_grid(asd2)
#plt.imshow(grid_img.permute(1, 2, 0))
    import matplotlib.pyplot as plt
        #print(x.shape)
    #x_rot, _ = prepro.preprocess_batch(x_t, solver.data_loader.input_dim)
    #plt.imshow(x_rot.view(28,28))
    #print(solver.data_loader.input_dim)
    #plt.imshow(x.view(28, 28))
    #print(x_rot.shape)
'''

class Preprocessing():
    def __init__(self, data_loader=None, thetas=None, scales=None):
        self.rotate = thetas is not None
        self.scale = scales is not None
        if self.rotate:
            self.theta_range_1, self.theta_range_2 = [v for _, v in thetas.items()]
            # add 1's because it's exclusive when gen. random
            self.theta_range_1[1] += 1
            self.theta_range_2[1] += 1
        if self.scale:
            self.scale_range_1 = scales["scale_1"]
            self.scale_range_1[1] += 0.1 # TODO: it's not precise...
            scale_1 = self._generate_scale()
            data_loader.img_dims = list([int(scale_1*x) for x in list((data_loader.img_dims[0], data_loader.img_dims[1]))])
            data_loader.input_dim = np.prod(data_loader.img_dims)

    def _generate_angles(self):
        theta_1 = np.random.randint(*self.theta_range_1)
        theta_2 = theta_1 + np.random.randint(*self.theta_range_2)
        return theta_1, theta_2
    
    def _generate_scale(self):
        a, b = self.scale_range_1
        scale_1 = (b-a)*np.random.random_sample() + a
        return scale_1

    def _rotate_batch(self, batch, angle):
        rotated = torch.zeros_like(batch)
        for i in range(batch.size(0)):
            img = TF.rotate(transforms.ToPILImage()(batch[i].cpu()), angle)
            rotated[i] = transforms.ToTensor()(img)
        return rotated

    # TODO: remove the reshaping and stuff., just do the actual resize, or not?
    def _scale_batch(self, batch, scale):
        img_dims = batch.size(2), batch.size(3)
        reshaped_dims = list([int(scale*x) for x in list(img_dims)])
        scaled = torch.zeros((batch.size(0), batch.size(1), *reshaped_dims))
        for i in range(batch.size(0)):
            img = TF.resize(transforms.ToPILImage()(batch[i].cpu()), reshaped_dims)
            scaled[i] = transforms.ToTensor()(img)
        return scaled
        
    def _scale_rotate_batch(self, batch, scale, angle):
        scaled = self._scale_batch(batch, scale)
        return self._rotate_batch(scaled, angle)

    def preprocess_batch(self, x, input_dim):
        if self.rotate:
            theta_1, theta_2 = self._generate_angles()
            print(theta_1, theta_2)
            x_rot = self._rotate_batch(x, theta_1)
            x_next = self._rotate_batch(x, theta_2)
        else:
            raise ValueError("Prepro of batch failed")
        return x_rot.view(-1, input_dim), x_next.view(-1, input_dim)