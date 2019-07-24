from preprocessing import preprocess_sample, generate_angles, generate_scales
import torchvision.transforms as transforms

class Rotate(object):
    def __init__(self, theta_range_1, theta_range_2):
        self.theta_range_1 = theta_range_1
        self.theta_range_2 = theta_range_2
        self.theta_1 = 0
        self.theta_2 = 0

    def __call__(self, sample):
        x_t = sample
        x_next = sample
        self.theta_1, self.theta_2 = generate_angles(self.theta_range_1, self.theta_range_2)
        x_t = preprocess_sample(transforms.ToTensor()(x_t), theta=self.theta_1)
        x_next = preprocess_sample(transforms.ToTensor()(x_next), theta=self.theta_2)
        return x_t, x_next

class Scale(object):
    def __init__(self, scale_range_1, scale_range_2):
        self.scale_range_1 = scale_range_1
        self.scale_range_2 = scale_range_2
        self.scale_1 = 0.0
        self.scale_2 = 0.0

    def __call__(self, sample):
        x_t = sample
        x_next = sample
        self.scale_1, self.scale_2 = generate_scales(self.scale_range_1, self.scale_range_2)
        x_t = preprocess_sample(transforms.ToTensor()(x_t), scale=(self.scale_1, self.scale_1))
        x_next = preprocess_sample(transforms.ToTensor()(x_next), scale=(self.scale_2, self.scale_2))
        return x_t, x_next

class ScaleRotate(object):
    def __init__(self, scale_range_1, scale_range_2, theta_range_1, theta_range_2):
        self.scale_range_1 = scale_range_1
        self.scale_range_2 = scale_range_2
        self.scale_1 = 0.0
        self.scale_2 = 0.0
        self.theta_range_1 = theta_range_1
        self.theta_range_2 = theta_range_2
        self.theta_1 = 0.0
        self.theta_2 = 0.0

    def __call__(self, sample):
        x_t = sample
        x_next = sample
        self.scale_1, self.scale_2 = generate_scales(self.scale_range_1, self.scale_range_2)
        self.theta_1, self.theta_2 = generate_angles(self.theta_range_1, self.theta_range_2)
        x_t = preprocess_sample(transforms.ToTensor()(x_t), theta=self.theta_1, scale=(self.scale_1, self.scale_1))
        x_next = preprocess_sample(transforms.ToTensor()(x_next), theta=self.theta_2, scale=(self.scale_2, self.scale_2))
        return x_t, x_next