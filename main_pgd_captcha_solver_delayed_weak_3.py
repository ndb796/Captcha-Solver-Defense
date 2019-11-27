import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision
import captcha_config as config
import numpy as np
import copy

class RES50(nn.Module):
    def __init__(self):
        super(RES50, self).__init__()
        self.num_cls = config.MAX_CAPTCHA * config.ALL_CHAR_SET_LEN
        self.base = torchvision.models.resnet50(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class LinfPGDAttack(object):
    def __init__(self, model, epsilon=0.1, k=7, a=0.03, random_start=True):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.random_start = random_start
        self.loss_fn = nn.MultiLabelSoftMarginLoss()

    def perturb(self, x_natural, y):
        if self.random_start:
            x = x_natural + np.random.uniform(-self.epsilon, self.epsilon, x_natural.shape).astype('float32')
        else:
            x = np.copy(x_natural)
        for i in range(self.k):
            x_var = Variable(torch.from_numpy(x).to(device), requires_grad=True)
            y_var = Variable(y, requires_grad=True)
            scores = self.model(x_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = x_var.grad.data.cpu().numpy()
            x += self.a * np.sign(grad)
            x = np.clip(x, x_natural - self.epsilon, x_natural + self.epsilon)
            x = np.clip(x, 0, 1)
        return x

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    for parameter in model_copied.parameters():
        parameter.requires_grad = False
    model_copied.eval()
    adversary.model = model_copied
    x_adv = adversary.perturb(x.numpy(), y)
    return torch.from_numpy(x_adv)

use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('Cuda Available')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 80
learning_rate = 0.0005

def test():
    print("[ Test Mode ]")
    cnn = RES50()
    cnn = torch.nn.DataParallel(cnn)
    cnn.to(device)
    cnn.eval()
    cnn.load_state_dict(torch.load("model_pgd_delayed_weak_3.pkl"))

    print("[ Model Loaded ]")
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    adversary = LinfPGDAttack(model=cnn)
    test_data_loader = config.get_test_data_loader()

    correct = 0
    total = 0
    t0 = time.time()
    count = 0
    total_loss = 0.0

    for i, (images, labels) in enumerate(test_data_loader):
        variable = Variable(images).to(device)
        predict_labels = cnn(variable).to(device)

        temp_labels = Variable(labels.float()).to(device)
        loss = criterion(predict_labels, temp_labels)

        total_loss += loss.item()
        count += 1

        current_batch_size = labels.size(0)

        for k in range(current_batch_size):
            predict_label = predict_labels[k]
            c0 = config.ALL_CHAR_SET[np.argmax(predict_label[0:config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
            c1 = config.ALL_CHAR_SET[np.argmax(predict_label[config.ALL_CHAR_SET_LEN:2 * config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
            c2 = config.ALL_CHAR_SET[np.argmax(predict_label[2 * config.ALL_CHAR_SET_LEN:3 * config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
            c3 = config.ALL_CHAR_SET[np.argmax(predict_label[3 * config.ALL_CHAR_SET_LEN:4 * config.ALL_CHAR_SET_LEN].data.cpu().numpy())]

            predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
            true_label = config.decode(labels.numpy()[k])
            if predict_label == true_label:
                correct += 1

        total += current_batch_size

    print("Total Test Accuracy of the model on the %d test images: %f %%" % (total, 100 * correct / total))
    print("Time Spent:", time.time() - t0)
    print("Average Loss:", total_loss / count)

    correct = 0
    total = 0
    t0 = time.time()
    count = 0
    total_loss = 0.0

    for i, (images, labels) in enumerate(test_data_loader):
        temp_labels = Variable(labels.float()).to(device)
        images_adv = attack(images, temp_labels, cnn, adversary)
        images_adv_var = Variable(images_adv).to(device)
        predict_labels = cnn(images_adv_var).to(device)

        temp_labels = Variable(labels.float()).to(device)
        loss = criterion(predict_labels, temp_labels)

        total_loss += loss.item()
        count += 1

        current_batch_size = labels.size(0)

        for k in range(current_batch_size):
            predict_label = predict_labels[k]
            c0 = config.ALL_CHAR_SET[np.argmax(predict_label[0:config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
            c1 = config.ALL_CHAR_SET[np.argmax(predict_label[config.ALL_CHAR_SET_LEN:2 * config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
            c2 = config.ALL_CHAR_SET[np.argmax(predict_label[2 * config.ALL_CHAR_SET_LEN:3 * config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
            c3 = config.ALL_CHAR_SET[np.argmax(predict_label[3 * config.ALL_CHAR_SET_LEN:4 * config.ALL_CHAR_SET_LEN].data.cpu().numpy())]

            predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
            true_label = config.decode(labels.numpy()[k])
            if predict_label == true_label:
                correct += 1

        total += current_batch_size

    print("Total Test Accuracy of the model on the %d test images: %f %%" % (total, 100 * correct / total))
    print("Time Spent:", time.time() - t0)
    print("Average Loss:", total_loss / count)

def main():
    cnn = RES50()
    cnn = torch.nn.DataParallel(cnn)
    cnn.cuda(device)
    cnn.train()

    torch.save(cnn.state_dict(), "./model_pgd_delayed_weak_3.pkl")
    print("[ Model Saved ]")
    test()

    print('[ Model Initialization ]')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    adversary = LinfPGDAttack(model=cnn)

    print('[ Time Checking Start ]')
    start_time = time.time()
    train_dataloader = config.get_train_data_loader()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        for i, (images, labels) in enumerate(train_dataloader):
            labels = Variable(labels.float()).to(device)

            images_natural = Variable(images).to(device)
            predict_labels = cnn(images_natural).to(device)
            loss = criterion(predict_labels, labels)

            if epoch >= 5:
                images_adv = attack(images, labels, cnn, adversary)
                images_adv_var = Variable(images_adv).to(device)
                predict_labels = cnn(images_adv_var).to(device)
                loss = (loss + criterion(predict_labels, labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())

        torch.save(cnn.state_dict(), "./model_pgd_delayed_weak_3.pkl")
        print("[ Model Saved ]")
        print("epoch:", epoch, "step:", i, "loss:", loss.item())
        print("Time Spent:", time.time() - epoch_start_time)
        test()

    torch.save(cnn.state_dict(), "./model_pgd_delayed_weak_3.pkl")
    print("[ Last Model Saved ]")
    print("Total Time", time.time() - start_time)

main()
