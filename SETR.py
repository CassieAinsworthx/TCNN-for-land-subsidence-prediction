import torch
import torch.nn as nn
from setr.Transformer import TransformerModel
from setr.PositionalEncoding import FixedPositionalEncoding
from setr.IntmdSequential import IntermediateSequential

import torch.optim as optim
import numpy
import torch.utils.data as data
from overall_dataset.dataset import deformation_dataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
torch.manual_seed(1)
import warnings
from setr.SETR_lines import Transformer_lines_without_intmd
warnings.filterwarnings("ignore")
from setr.SETRs import *
from LSTM import RegLSTM

batch_size = 12

def train_transformer_SETR(model,line_length,path):
    loss_fn = nn.MSELoss(reduction='mean')

    optimizer = optim.SGD(model.parameters(), lr=0.0008)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.8)

    time_train_dataset = data.DataLoader(deformation_dataset("train", line_length, line_length), batch_size=batch_size,
                                         shuffle=True)
    time_test_dataset = data.DataLoader(deformation_dataset("test", line_length, line_length), batch_size=batch_size,
                                        shuffle=True)

    best_accuracy = -1
    epoch_flag = -1
    for epoch in range(50):
        loss_avg = 0
        accuracy_avg = 0
        steps = 0
        print("epoch {} --------------------------".format(epoch))
        for batch_x, batch_y,batch_z, attr_value in tqdm(time_train_dataset):
            if batch_x.size()[0] != 12:
                continue
            batch_x = Variable(batch_x)
            batch_y = Variable(batch_y)
            # attr_value = Variable(attr_value)

            optimizer.zero_grad()
            tag_scores = model(batch_z)
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
        for batch_x, batch_y,batch_z, attr_value in tqdm(time_test_dataset):
            if batch_x.size()[0] != 12:
                continue
            tag_scores = model(batch_z)
            accuracy += numpy.sqrt(mean_squared_error(tag_scores.detach().numpy(), batch_y.numpy()))
            step_test += 1

        with open(path+".txt", 'a', encoding='UTF-8') as f:
            print("epoch{},Test accuracy :{}".format(epoch,accuracy / step_test),file=f)
        print("Test accuracy :{}".format(accuracy / step_test))
        if accuracy/step_test < best_accuracy:
            best_accuracy = accuracy/step_test
            epoch_flag = epoch

    torch.save(model,path+".pth")
    return best_accuracy,epoch_flag

def train_LSTM(line_length,hidden_size,num_layers,data_path):
    model= RegLSTM(input_size = 1,hidden_size = hidden_size,hidden_num_layers = num_layers,line_length=line_length)

    loss_fn = nn.MSELoss(reduction='mean')
    # RMSE_loss = torch.sqrt(loss_fn(prediction, target))
    # RMSE_loss.backward()
    optimizer = optim.SGD(model.parameters(), lr=0.0008)
    # lookback_window = 12
    time_train_dataset = data.DataLoader(
        deformation_dataset("train", line_length,line_length),
        batch_size=12, shuffle=True)
    time_test_dataset = data.DataLoader(
        deformation_dataset("test", line_length,line_length),
        batch_size=12, shuffle=True)

    for epoch in range(50):
        loss_avg = 0
        accuracy_avg = 0
        steps = 0
        print("epoch {} --------------------------".format(epoch))
        for batch_x, batch_y,batch_z, attr_value in tqdm(time_train_dataset):
            if batch_x.size()[0] != 12:
                continue
            batch_x = Variable(batch_x)
            batch_y = Variable(batch_y)
            attr_value = Variable(attr_value)

            optimizer.zero_grad()
            tag_scores = model(batch_z)
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
        for batch_x, batch_y,batch_z, attr_value in tqdm(time_test_dataset):
            if batch_x.size()[0] != 12:
                continue
            tag_scores = model(batch_z)
            accuracy += numpy.sqrt(mean_squared_error(tag_scores.detach().numpy(), batch_y.numpy()))
            step_test += 1

        with open(data_path+".txt", 'a', encoding='UTF-8') as f:
            print("epoch{},Test accuracy :{}".format(epoch, accuracy / step_test), file=f)
        print("Test accuracy :{}".format(accuracy / step_test))

    torch.save(model,data_path+".pth",)

def train_transformer(model,lookback_window,path,line_length,epochs):
    loss_fn = nn.MSELoss(reduction='mean')

    optimizer = optim.SGD(model.parameters(), lr=0.0008)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.8)

    time_train_dataset = data.DataLoader(
        deformation_dataset("train", lookback_window,line_length),
        batch_size=12, shuffle=True)
    time_test_dataset = data.DataLoader(
        deformation_dataset("test", lookback_window,line_length),
        batch_size=12, shuffle=True)

    best_accuracy = -1
    for epoch in range(epochs):
        loss_avg = 0
        accuracy_avg = 0
        steps = 0
        print("epoch {} --------------------------".format(epoch))
        for batch_x, batch_y,batch_z, attr_value in tqdm(time_train_dataset):
            if batch_x.size()[0] != 12:
                continue
            batch_x = Variable(batch_x)
            batch_y = Variable(batch_y)
            attr_value = Variable(attr_value)

            optimizer.zero_grad()
            tag_scores = model(batch_x)
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
                print("train loss:{}, accuracy:{}".format(loss_avg * 100, accuracy_avg * 100))
                loss_avg = 0
                accuracy_avg = 0
                steps = 0

        print("\n epoch {} test ------------------------".format(epoch))
        accuracy = 0
        step_test = 0
        with torch.no_grad():
            step_max = time_test_dataset.__len__() - 1
            for batch_x, batch_y,batch_z, attr_value in tqdm(time_test_dataset):
                if batch_x.size()[0] != 12:
                    continue
                else:
                    tag_scores = model(batch_x)
                    accuracy += numpy.sqrt(mean_squared_error(tag_scores.detach().numpy(), batch_y.numpy()))
                    step_test += 1

        with open(path+"dual.txt", 'a', encoding='UTF-8') as f:
            print("epoch{},Test accuracy :{}".format(epoch,accuracy / step_test),file=f)
        print("Test accuracy :{}".format(accuracy / step_test))
        if accuracy / step_test < best_accuracy:
            best_accuracy = accuracy / step_test

    torch.save(model,path+"dual.pth")
    return best_accuracy

if __name__ == "__main__":

    # train_LSTM(line_length=32,  hidden_size=200, num_layers=2, data_path=r"LSTM_32_100_2")
    # train_LSTM(line_length=16,  hidden_size=200, num_layers=2, data_path=r"LSTM_16_100_2")
    # train_LSTM(line_length=64, hidden_size=200, num_layers=2, data_path=r"LSTM_64_100_2")
    #
    # model = Transformer_without_intmd(embedding_dim=64, num_heads=4, num_layers=2, hidden_dim=128, batch_size=12,line_length=32)
    # best_accuracy,epoch_flag = train_transformer_SETR(model,32,"exps/setr_64_4_2_128_32.txt")
    #
    # model = Transformer_without_intmd(embedding_dim=24, num_heads=4, num_layers=2, hidden_dim=64, batch_size=12,line_length=16)
    # best_accuracy,epoch_flag = train_transformer_SETR(model,16,"exps/setr_24_4_2_64_16.txt")
    #
    # model = Transformer_without_intmd(embedding_dim=128, num_heads=4, num_layers=2, hidden_dim=256, batch_size=12,line_length=64)
    # best_accuracy,epoch_flag = train_transformer_SETR(model,64,"exps/setr_128_4_2_256_64.txt")
    #
    model = Transformer_lines_without_intmd(input_dim=16, embedding_dim=24, num_heads=4, num_layers=2, hidden_dim=64,
                                            # embedding 32时 到0.249
                                            batch_size=12,
                                            line_length=12)
    best_accuracy = train_transformer(model, 16, "setr_lines16_24_4_2_64_12_fordual", 12,50)  # 4x4

    # model = Transformer_lines_without_intmd(input_dim=32, embedding_dim=64, num_heads=4, num_layers=2, hidden_dim=128,
    #                                         # embedding 32时 到0.249
    #                                         batch_size=12,
    #                                         line_length=12)  # line length 为输入line组长度  input_dim 为每个encoding的长度    # 最高到0.2460
    # best_accuracy = train_transformer(model, 32, "setr_lines_fordual", 12,50)  # 4x4

    # model = Transformer_lines_without_intmd(input_dim=64, embedding_dim=128, num_heads=4, num_layers=2, hidden_dim=256,
    #                                         # embedding 32时 到0.249
    #                                         batch_size=12,
    #                                         line_length=12)  # line length 为输入line组长度  input_dim 为每个encoding的长度    # 最高到0.2460
    # best_accuracy = train_transformer(model, 64, "setr_lines64_128_4_2_256_12", 12,50)  # 4x4





