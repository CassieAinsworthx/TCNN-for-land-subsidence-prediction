import torch
import torch.nn as nn
import torch.optim as optim
import numpy
import torch.utils.data as data
from overall_dataset.dataset import deformation_dataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
torch.manual_seed(1)
import warnings

warnings.filterwarnings("ignore")


"""
class torch.nn.LSTM(*args, **kwargs)
参数有：
    input_size：x的特征维度
    hidden_size：隐藏层的特征维度
    num_layers：lstm隐层的层数，默认为1
    bias：False则bihbih=0和bhhbhh=0. 默认为True
    batch_first：True则输入输出的数据格式为 (batch, seq, feature)
    dropout：除最后一层，每一层的输出都进行dropout，默认为: 0
    bidirectional：True则为双向lstm默认为False

输入：
input(seq_len, batch, input_size)
参数有：
    seq_len：序列长度，在NLP中就是句子长度，一般都会用pad_sequence补齐长度
    batch：每次喂给网络的数据条数，在NLP中就是一次喂给网络多少个句子
    input_size：特征维度，和前面定义网络结构的input_size一致。

输出：
output,(ht, ct) = net(input)
    output: 最后一个状态的隐藏层的神经元输出
    ht：最后一个状态的隐含层的状态值
    ct：最后一个状态的隐含层的遗忘门值

output(seq_len, batch, hidden_size * num_directions)
ht(num_layers * num_directions, batch, hidden_size)
ct(num_layers * num_directions, batch, hidden_size)
"""


random_dataset_length = 5000

class RegLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,hidden_num_layers,line_length):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, hidden_num_layers,batch_first=True) # 这里不需要定义句子长度

        self.reg = nn.Sequential(
            # nn.Linear(12800, 6400),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(6400, 3200),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden_size*line_length, 1200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1200, 600),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(600, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)

        )

    def forward(self, x):
        x, (ht,ct) = self.rnn(x)
        batch_size,seq_len,hidden_size= x.shape
        # x = x.reshape(-1, hidden_size)
        # x = self.reg(x)
        # x = x.view(batch_size,seq_len,  -1)
        x = x.reshape(batch_size, -1)
        x = self.reg(x)
        # x = x.view(batch_size,seq_len,  -1)

        return x

def train_LSTM(line_length,batch_size,hidden_size,num_layers,data_path):
    model = RegLSTM(input_size = 1,hidden_size = hidden_size,hidden_num_layers = num_layers)

    loss_fn = nn.MSELoss(reduction='mean')
    # RMSE_loss = torch.sqrt(loss_fn(prediction, target))
    # RMSE_loss.backward()
    optimizer = optim.SGD(model.parameters(), lr=0.0008)

    data_time = numpy.load(r"data/datas_new/data_time.npy")[0:4000]
    data_attribute = numpy.load(r"data/datas_new/data_attr.npy")[0:4000]

    # use_id = numpy.load("data/datas_new/matrix_flag_4×4.npy")
    # data_time = data_time[use_id]
    # data_attribute = data_attribute[use_id]

    time_train_dataset = data.DataLoader(
        time_series_dataset("train", random_dataset_length, line_length, data_time, data_attribute),
        batch_size=batch_size, shuffle=True)
    time_test_dataset = data.DataLoader(
        time_series_dataset("test", random_dataset_length, line_length, data_time, data_attribute),
        batch_size=batch_size, shuffle=True)

    for epoch in range(50):
        loss_avg = 0
        accuracy_avg = 0
        steps = 0
        print("epoch {} --------------------------".format(epoch))
        for batch_x, batch_y, attr_value in tqdm(time_train_dataset):
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
                print("train loss:{}".format(loss_avg * 100))
                steps = 0

        print("\n epoch {} test ------------------------".format(epoch))
        accuracy = 0
        step_test = 0
        for batch_x, batch_y, attr_value in tqdm(time_test_dataset):
            if batch_x.size()[0] != 12:
                continue
            tag_scores = model(batch_x)
            accuracy += numpy.sqrt(mean_squared_error(tag_scores.detach().numpy(), batch_y.numpy()))
            step_test += 1

        with open(data_path, 'a', encoding='UTF-8') as f:
            print("epoch{},Test accuracy :{}".format(epoch, accuracy / step_test), file=f)
        print("Test accuracy :{}".format(accuracy / step_test))


if __name__ == "__main__":

    train_LSTM(line_length=32, batch_size=12, hidden_size=200, num_layers=2,data_path=r"LSTM_32_100_2.txt")
    train_LSTM(line_length=16, batch_size=12, hidden_size=200, num_layers=2,data_path=r"LSTM_12_100_2.txt")
    train_LSTM(line_length=64, batch_size=12, hidden_size=200, num_layers=2,data_path=r"LSTM_64_100_2.txt")


