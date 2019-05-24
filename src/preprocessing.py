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
    def __init__(self, data_loader, thetas=None, scales=None):
        self.rotate = thetas is not None
        self.scale = scales is not None
        self.data_loader = data_loader
        if self.rotate:
            self.theta_range_1, self.theta_range_2 = [v for _, v in thetas.items()]
            # add 1's because it's exclusive when gen. random
            self.theta_range_1[1] += 1
            self.theta_range_2[1] += 1
        if self.scale:
            self.scale_range_1, self.scale_range_2 = scales["scale_1"], scales["scale_2"]
            self.scale_range_1[1] = round(self.scale_range_1[1]+0.1, 1)
            self.scale_range_2[1] = round(self.scale_range_2[1]+0.1, 1)
            # - implement scaling - just give max(scale_1, scale_2) neurons for the input? and then for the min, they're just ignored? Do padding? 
            # TODO: mtake
            # TODO: not sure if this below...
            # change img dims and input dim after the scaling
            #self.data_loader.img_dims = list([int(scale_1*x) for x in list((data_loader.img_dims[0], data_loader.img_dims[1]))])
            #self.data_loader.input_dim = np.prod(data_loader.img_dims)

    def _generate_angles(self):
        theta_1 = np.random.randint(*self.theta_range_1)
        theta_2 = theta_1 + np.random.randint(*self.theta_range_2)
        return theta_1, theta_2

    def _generate_scales(self):
        (a, b), (c, d) = self.scale_range_1, self.scale_range_2
        scale_1 = (b-a)*np.random.random_sample() + a
        scale_2 = scale_1 + ((d-c)*np.random.random_sample() + c)
        return scale_1, scale_2

    def _rotate_batch(self, batch, angle):
        rotated = torch.zeros_like(batch)
        for i in range(batch.size(0)):
            img = TF.rotate(transforms.ToPILImage()(batch[i].cpu()), angle)
            rotated[i] = transforms.ToTensor()(img)
        return rotated

    # TODO:
    def _scale_batch(self, batch):
        scaled = torch.zeros((batch.size(0), batch.size(1), *self.data_loader.input_dim))
        for i in range(batch.size(0)):
            img = TF.resize(transforms.ToPILImage()(batch[i].cpu()), *self.data_loader.input_dim)
            scaled[i] = transforms.ToTensor()(img)
        return scaled

    def _scale_rotate_batch(self, batch, scale, angle):
        scaled_batch = self._scale_batch(batch)
        return self._rotate_batch(scaled_batch, angle)

    def preprocess_batch(self, x):
        res = {}
        if self.rotate:
            theta_1, theta_2 = self._generate_angles()
            x1 = self._rotate_batch(x, theta_1)
            x2 = self._rotate_batch(x, theta_2)
            theta_diff = theta_2 - theta_1
        elif self.scale:
            scale_1, scale_2 = self._generate_scales()
            x1 = self._scale_batch(x, scale_1)
            x2 = self._scale_batch(x, scale_2)
            scale_diff = scale_2 - scale_1
        elif self.rotate and self.scale:
            # TODO:
            pass
        else:
            raise ValueError("Prepro of batch failed")
        if self.rotate:
            res["theta_diff"] = theta_diff
            res["theta_1"] = theta_1
        if self.scale:
            res["scale_diff"] = scale_diff
            res["scale_1"] = scale_1
        return x1.view(-1, self.data_loader.input_dim), x2.view(-1, self.data_loader.input_dim), res