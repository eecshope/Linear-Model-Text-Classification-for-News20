import numpy as np


class LogLinear:
    def __init__(self, n_label, n_feature, lr):
        self.n_label = n_label
        self.n_feature = n_feature
        self.w = np.random.normal(size=[n_label, n_feature])
        self.grad = 0
        self.lr = lr
        self.masks = np.ones([n_label, n_label, n_feature], dtype=np.bool)
        for i in range(n_label):
            self.masks[i][i] = np.zeros([n_feature], np.bool)

    def get_product(self, x):
        if len(x.shape) == 2:
            x = np.expand_dims(x, 0)  # to ensure the x is a 3-d tensor

        product = np.sum(np.multiply(x, self.w), -1)
        return product  # [batch_size, n_label]

    def set_lr(self, lr):
        self.lr = lr

    def train(self, x, label, if_sgd=False):
        if len(x.shape) == 2:
            x = np.expand_dims(x, 0)  # [batch_size, n_label, n_feature]
        if len(label.shape) == 0:
            label = np.expand_dims(label, 0)  # [batch_size]

        x = x.copy()
        label.dtype = 'int32'
        batch_size = x.shape[0]
        product = self.get_product(x)

        exp_product = np.exp(product)  # [batch_size, n_label]
        norm = np.expand_dims(exp_product, -1) * x / np.expand_dims(np.sum(exp_product, -1, keepdims=True), -1)
        norm_factor = np.sum(norm, 0)  # [n_label, n_feature]

        masks = self.masks[label]  # [batch_size, n_label, n_feature]
        np.putmask(x, masks, 0)
        real_factor = np.sum(x, 0)  # [n_label, n_feature]

        del x
        added_loss = real_factor - norm_factor
        self.grad = self.grad + added_loss
        if if_sgd:
            self.w = self.w + self.lr * self.grad
            self.grad = 0

        return added_loss / batch_size

    def update(self):
        self.w = self.w + self.lr * self.grad
        self.grad = 0

    def predict(self, x):
        if x.shape != (self.n_label, self.n_feature):
            raise ValueError("wrong prediction input")
        prediction = np.sum(self.w*x, -1)
        return np.argmax(prediction)
