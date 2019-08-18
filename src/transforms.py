from preprocessing import preprocess_sample, generate_angles, generate_scales
import torchvision.transforms as transforms

class Rotate(object):
    def __init__(self, theta_range_1=None, theta_range_2=None, time_agnostic=False):
        self.theta_range_1 = theta_range_1
        self.theta_range_2 = theta_range_2
        self.theta_1 = 0.0
        self.theta_2 = 0.0
        self.time_agnostic = time_agnostic

    def __call__(self, sample):
        sample = transforms.ToTensor()(sample)
        x_t = sample.clone().detach()
        x_next = sample.clone().detach()
        self.theta_1, self.theta_2 = generate_angles(self.theta_range_1, self.theta_range_2, self.time_agnostic)
        x_t = preprocess_sample(x_t, theta=self.theta_1)
        x_next = preprocess_sample(x_next, theta=self.theta_2)
        return x_t, x_next

class Scale(object):
    def __init__(self, scale_range_1=None, scale_range_2=None, time_agnostic=False):
        self.scale_range_1 = scale_range_1
        self.scale_range_2 = scale_range_2
        self.scale_1 = 0.0
        self.scale_2 = 0.0
        self.time_agnostic = time_agnostic

    def __call__(self, sample):
        sample = transforms.ToTensor()(sample)
        x_t = sample.clone().detach()
        x_next = sample.clone().detach()
        self.scale_1, self.scale_2 = generate_scales(self.scale_range_1, self.scale_range_2, self.time_agnostic)
        x_t = preprocess_sample(x_t, scale=(self.scale_1, self.scale_1))
        x_next = preprocess_sample(x_next, scale=(self.scale_2, self.scale_2))
        return x_t, x_next

class ScaleRotate(object):
    def __init__(self, scale_range_1=None, scale_range_2=None, theta_range_1=None, theta_range_2=None, time_agnostic=False):
        self.scale_range_1 = scale_range_1
        self.scale_range_2 = scale_range_2
        self.scale_1 = 0.0
        self.scale_2 = 0.0
        self.theta_range_1 = theta_range_1
        self.theta_range_2 = theta_range_2
        self.theta_1 = 0.0
        self.theta_2 = 0.0
        self.time_agnostic = time_agnostic

    def __call__(self, sample):
        sample = transforms.ToTensor()(sample)
        x_t = sample.clone().detach()
        x_next = sample.clone().detach()
        self.scale_1, self.scale_2 = generate_scales(self.scale_range_1, self.scale_range_2, self.time_agnostic)
        self.theta_1, self.theta_2 = generate_angles(self.theta_range_1, self.theta_range_2, self.time_agnostic)
        x_t = preprocess_sample(x_t, theta=self.theta_1, scale=(self.scale_1, self.scale_1))
        x_next = preprocess_sample(x_next, theta=self.theta_2, scale=(self.scale_2, self.scale_2))
        return x_t, x_next