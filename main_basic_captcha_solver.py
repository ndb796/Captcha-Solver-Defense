import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision
import captcha_config as config
import numpy as np

class RES18(nn.Module):
    def __init__(self):
        super(RES18, self).__init__()
        self.num_cls = config.MAX_CAPTCHA * config.ALL_CHAR_SET_LEN
        self.base = torchvision.models.resnet18(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('Cuda Available')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 80
learning_rate = 0.0005

def test():
    print("[ Test Mode ]")
    cnn = RES18()
    cnn = torch.nn.DataParallel(cnn)
    cnn.to(device)
    cnn.eval()
    cnn.load_state_dict(torch.load("model.pkl"))

    print("[ Model Loaded ]")
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    correct = 0
    total = 0
    t0 = time.time()
    count = 0
    total_loss = 0.0

    test_data_loader = config.get_test_data_loader()
    for i, (images, labels) in enumerate(test_data_loader):
        image = images
        variable = Variable(image).to(device)
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

def main():
    cnn = RES18()
    cnn = torch.nn.DataParallel(cnn)
    cnn.cuda(device)
    cnn.train()
    
    torch.save(cnn.state_dict(), "./model.pkl")
    print("[ Model Saved ]")
    test()

    print('[ Model Initialization ]')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    print('[ Time Checking Start ]')
    start_time = time.time()
    train_dataloader = config.get_train_data_loader()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images).to(device)
            labels = Variable(labels.float()).to(device)
            predict_labels = cnn(images).to(device)

            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
        torch.save(cnn.state_dict(), "./model.pkl")
        print("[ Model Saved ]")
        print("epoch:", epoch, "step:", i, "loss:", loss.item())
        print("Time Spent:", time.time() - epoch_start_time)
        test()

    torch.save(cnn.state_dict(), "./model.pkl")
    print("[ Last Model Saved ]")
    print("Total Time", time.time() - start_time)

main()
