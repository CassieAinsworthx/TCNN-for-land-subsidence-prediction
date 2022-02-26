import torch.nn as nn
import torch.optim as optim
import numpy
import torch.utils.data as data
from overall_dataset.dataset import deformation_dataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
import torch

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=1,stride=1,padding=0 ), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),

            nn.Conv2d(64, 256, kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1,stride=1,padding=0),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


line_length = 12  # look back window
batch_size = 12

model = AlexNet()

loss_fn = nn.MSELoss(reduction='mean')

optimizer = optim.SGD(model.parameters(), lr=0.0001)
lookback_window = 12
time_train_dataset = data.DataLoader(deformation_dataset("train",16,line_length),batch_size=batch_size,shuffle=True)
time_test_dataset = data.DataLoader(deformation_dataset("test",16,line_length),batch_size=batch_size,shuffle=True)

best_accuracy = 100

for epoch in range(50):
    loss_avg = 0
    accuracy_avg = 0
    steps = 0
    print("epoch {} --------------------------".format(epoch))
    for batch_x, batch_y, batch_z,attr_value in tqdm(time_train_dataset):
        if batch_x.size()[0] != 12:
            continue
        batch_x = Variable(batch_x)
        batch_y = Variable(batch_y)
        attr_value = Variable(attr_value)

        optimizer.zero_grad()
        tag_scores = model(attr_value)
        mse_loss = loss_fn(tag_scores, batch_y)
        RMSE_loss = torch.sqrt(mse_loss)
        RMSE_loss.backward()
        optimizer.step()

        loss_avg += RMSE_loss
        accuracy_avg += numpy.sqrt(mean_squared_error(tag_scores.detach().numpy(), batch_y.numpy()))
        steps +=1
        if steps == 500:
            loss_avg /= steps
            accuracy_avg /= steps
            print("train loss:{}".format(loss_avg*100))
            steps = 0
            loss_avg = 0
            accuracy_avg = 0

    print("\n epoch {} test ------------------------".format(epoch))
    accuracy = 0
    step_test = 0

    with torch.no_grad():

        for batch_x, batch_y,batch_z, attr_value in tqdm(time_test_dataset):
            if batch_x.size()[0] != 12:
                continue
            tag_scores = model(attr_value)
            accuracy += numpy.sqrt(mean_squared_error(tag_scores.detach().numpy(), batch_y.numpy()))
            step_test += 1
    with open("alexnet_1d_3.txt", 'a', encoding='UTF-8') as f:
        print("Test accuracy :{}".format(accuracy / step_test), file=f)
    print("Test accuracy :{}".format(accuracy / step_test))

    if accuracy / step_test < best_accuracy:
        torch.save(model.state_dict(),"best_1d_alexnet_2_state_dict.pth")

torch.save(model.state_dict(),"AlexNet_1d_final_state_dict.pth")
