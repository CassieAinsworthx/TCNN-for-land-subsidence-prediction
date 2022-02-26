from overall_dataset.dataset import deformation_dataset_matrix_attr
from Nets_for_matrix.AlexNet_for_dual import AlexNet_4x4
import torch.nn as nn
import torch.optim as optim
import numpy
import torch.utils.data as data

from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
import torch

# from data.data_cluster.load_matrix import setr_load_matrix_dual
from Dual_net_cat_exp import *
line_length = 12  # look back window
batch_size = 12

time_train_dataset = data.DataLoader(deformation_dataset_matrix_attr("train",16,line_length),batch_size=batch_size,shuffle=True)
time_test_dataset = data.DataLoader(deformation_dataset_matrix_attr("test",16,line_length),batch_size=batch_size,shuffle=True)


def train(model,num_flag, line_length,batch_size,input_dim,):
    loss_fn = nn.MSELoss(reduction='mean')

    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.8)

    best_accuracy = 100
    epoch_flag = 100
    for epoch in range(100):
        loss_avg = 0
        accuracy_avg = 0
        steps = 0
        print("epoch {} --------------------------".format(epoch))
        for batch_x, batch_y, batch_z, attr_value in tqdm(time_train_dataset):  # x lines y label z origins attrvalue
            if batch_x.size()[0] != 12:
                continue
            batch_x = Variable(batch_x)
            batch_y = Variable(batch_y)
            batch_z = Variable(batch_z)
            attr_value = Variable(attr_value.transpose(1,3))

            optimizer.zero_grad()


            tag_scores = model(batch_x, attr_value)
            mse_loss = loss_fn(tag_scores, batch_y)
            RMSE_loss = torch.sqrt(mse_loss)
            RMSE_loss.backward()
            optimizer.step()

            loss_avg += RMSE_loss
            accuracy_avg += numpy.sqrt(mean_squared_error(tag_scores.detach().numpy(), batch_y.numpy()))
            steps += 1
            if steps == 500:
                loss_avg /= steps
                accuracy_avg /= steps
                print("train loss:{}".format(loss_avg * 100))
                steps = 0

        print("\n epoch {} test ------------------------".format(epoch))
        accuracy = 0
        step_test = 0
        for batch_x, batch_y, batch_z, attr_value in tqdm(time_test_dataset):
            if batch_x.size()[0] != 12:
                continue
            tag_scores = model(batch_x, attr_value.transpose(1,3))
            accuracy += numpy.sqrt(mean_squared_error(tag_scores.detach().numpy(), batch_y.numpy()))
            step_test += 1

        print("Test accuracy :{}".format(accuracy / step_test))

        with open("setr_dual_whole"+num_flag+".txt", 'a', encoding='UTF-8') as f:
            print("Test accuracy :{}".format(accuracy / step_test), file=f)

        if accuracy / step_test < best_accuracy:
            best_accuracy = accuracy / step_test
            epoch_flag = epoch

    torch.save(model, "SETR_dual"+num_flag+".pth")


if __name__ == "__main__":
    line_length = 12
    batch_size = 12
    input_dim = 12

    model = Dual_net(input_dim=16, embedding_dim=24, num_heads=4, num_layers=2, hidden_dim=64,
                                            # embedding 32时 到0.249
                                            batch_size=12,
                                            line_length=12)

    train(model,"fin",line_length,batch_size,input_dim)

