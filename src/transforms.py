from preprocessing import preprocess_sample, generate_angles, generate_scales
import torchvision.transforms as transforms

class Rotate(object):
    def __init__(self, batch_size, theta_range_1, theta_range_2):
        self.batch_size = batch_size
        self.count = 0
        self.theta_range_1 = theta_range_1
        self.theta_range_2 = theta_range_2
        self.theta_1 = 0.0
        self.theta_2 = 0.0

    def __call__(self, sample):
        sample = transforms.ToTensor()(sample)
        x_t = sample.clone().detach()
        x_next = sample.clone().detach()
        if self.count % self.batch_size:
            self.theta_1, self.theta_2 = generate_angles(None, None)
        x_t = preprocess_sample(x_t, theta=self.theta_1)
        x_next = preprocess_sample(x_next, theta=self.theta_2)
        self.count += 1
        return x_t, x_next

class Scale(object):
    def __init__(self, batch_size, scale_range_1, scale_range_2):
        self.batch_size = batch_size
        self.count = 0
        self.scale_range_1 = scale_range_1
        self.scale_range_2 = scale_range_2
        self.scale_1 = 0.0
        self.scale_2 = 0.0

    def __call__(self, sample):
        sample = transforms.ToTensor()(sample)
        x_t = sample.clone().detach()
        x_next = sample.clone().detach()
        if self.count % self.batch_size:
            self.scale_1, self.scale_2 = generate_scales(None, None)
        x_t = preprocess_sample(x_t, scale=(self.scale_1, self.scale_1))
        x_next = preprocess_sample(x_next, scale=(self.scale_2, self.scale_2))
        self.count += 1
        return x_t, x_next

class ScaleRotate(object):
    def __init__(self, batch_size, scale_range_1, scale_range_2, theta_range_1, theta_range_2):
        self.batch_size = batch_size
        self.count = 0
        self.scale_range_1 = scale_range_1
        self.scale_range_2 = scale_range_2
        self.scale_1 = 0.0
        self.scale_2 = 0.0
        self.theta_range_1 = theta_range_1
        self.theta_range_2 = theta_range_2
        self.theta_1 = 0.0
        self.theta_2 = 0.0

    def __call__(self, sample):
        sample = transforms.ToTensor()(sample)
        x_t = sample.clone().detach()
        x_next = sample.clone().detach()
        if self.count % self.batch_size:
            self.scale_1, self.scale_2 = generate_scales(None, None)
            self.theta_1, self.theta_2 = generate_angles(None, None)
        #print("test", self.scale_1, self.scale_2, self.theta_1, self.theta_2)
        x_t = preprocess_sample(x_t, theta=self.theta_1, scale=(self.scale_1, self.scale_1))
        x_next = preprocess_sample(x_next, theta=self.theta_2, scale=(self.scale_2, self.scale_2))
        self.count += 1
        return x_t, x_next