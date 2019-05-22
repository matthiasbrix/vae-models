import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import torch

class Preprocessing():

    def __init__(self, thetas):
        self.theta_range_1, self.theta_range_2 = [v for _, v in thetas.items()]
        self.theta_range_1[1] += 1
        self.theta_range_2[1] += 1

    def generate_angles(self):
        theta_1 = np.random.randint(*self.theta_range_1)
        theta_2 = theta_1 + np.random.randint(*self.theta_range_2)
        return theta_1, theta_2

    def rand_rotate_img(self, img, angles):
        transform_img = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(angles),
                transforms.ToTensor()
            ])
        return transform_img(img)

    def det_rotate_img(self, img, angle):
        img = TF.rotate(transforms.ToPILImage()(img), angle)
        return transforms.ToTensor()(img)

    # copied and modified from: https://github.com/ptrblck/pytorch_misc/blob/master/image_rotation_with_matrix.py
    def rotate_batch(self, batch, angle):
        x_mid = (batch.size(2) + 1) / 2.
        y_mid = (batch.size(3) + 1) / 2.
        angle = torch.FloatTensor([angle])
        # Calculate rotation with inverse rotation matrix
        rot_matrix = torch.tensor([[torch.cos(angle), torch.sin(angle)],
                                [-1.0*torch.sin(angle), torch.cos(angle)]])
        
        # Use meshgrid for pixel coords
        xv, yv = torch.meshgrid(torch.arange(batch.size(2)), torch.arange(batch.size(3)))
        xv = xv.contiguous()
        yv = yv.contiguous()
        src_ind = torch.cat((
            (xv.float() - x_mid).view(-1, 1),
            (yv.float() - y_mid).view(-1, 1)),
            dim=1
        )

        # Calculate indices using rotation matrix
        src_ind = torch.matmul(src_ind, rot_matrix.t())
        src_ind = torch.round(src_ind)
        src_ind += torch.tensor([[x_mid, y_mid]])

        # Set out of bounds indices to limits
        src_ind[src_ind < 0] = 0.
        src_ind[:, 0][src_ind[:, 0] >= batch.size(2)] = float(batch.size(2)) - 1
        src_ind[:, 1][src_ind[:, 1] >= batch.size(3)] = float(batch.size(3)) - 1

        # Create new rotated image
        im_rot2 = torch.zeros_like(batch)
        src_ind = src_ind.long()
        im_rot2[:, :, xv.view(-1), yv.view(-1)] = batch[:, :, src_ind[:, 0], src_ind[:, 1]]
        im_rot2 = im_rot2.view(batch.size(0), 1, batch.size(2), batch.size(3))
        return im_rot2

    # TODO: scale_batch - need also a way to rotate + scale
    def scale_batch(self, batch, scale):
        for i in range(batch.size(0)):
            img = TF.resize(transforms.ToPILImage()(batch[i]), scale)
            return transforms.ToTensor()(img)