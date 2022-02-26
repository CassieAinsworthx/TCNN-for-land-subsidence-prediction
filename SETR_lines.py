import torch
import torch.nn as nn
from setr.Transformer import TransformerModel
from setr.PositionalEncoding import FixedPositionalEncoding
from setr.IntmdSequential import IntermediateSequential
from overall_dataset.dataset import deformation_dataset
import torch.optim as optim
import numpy
import torch.utils.data as data

from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
torch.manual_seed(1)
import warnings


warnings.filterwarnings("ignore")

class Transformer_lines(nn.Module):
    def __init__(self,input_dim,embedding_dim,num_heads,num_layers,hidden_dim,batch_size,line_length):
        super(Transformer_lines, self).__init__()
        self.batch_size = batch_size
        # encoding
        self.linear_encoding = nn.Linear(input_dim, embedding_dim)

        self.pe_dropout = nn.Dropout(0.8)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            dropout_rate=0.8,
            attn_dropout_rate=0.8
        )

        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        self.position_encoding = FixedPositionalEncoding(embedding_dim,line_length)

        # decoding

        self.linear_intmd_1 = nn.Conv1d(line_length, line_length // 2,kernel_size=1)
        self.linear_intmd_2 = nn.Conv1d(line_length, line_length // 2,kernel_size=1)
        self.linear_intmd_3 = nn.Conv1d(line_length, line_length // 2,kernel_size=1)
        self.linear_intmd_4 = nn.Conv1d(line_length, line_length // 2,kernel_size=1)

        self.linear_cat1nd2 = nn.Conv1d(line_length, line_length // 2,kernel_size=1)
        self.linear_cat2nd3 = nn.Conv1d(line_length, line_length // 2,kernel_size=1)
        self.linear_cat3nd4 = nn.Conv1d(line_length, line_length // 2,kernel_size=1)

        self.relu = nn.ReLU()

        self.linear_intmd = nn.Sequential(
            nn.Linear(line_length//2 * embedding_dim, line_length//2 * embedding_dim//2),
            nn.Dropout(0.5),
            nn.Linear(line_length//2 * embedding_dim//2,line_length//2 * embedding_dim//4),
            nn.Dropout(0.5)
        )

        self.linear_x = nn.Sequential(
            nn.Linear(line_length * embedding_dim, line_length * embedding_dim//2),
            nn.Dropout(0.5),
            nn.Linear(line_length * embedding_dim//2, line_length * embedding_dim//4),
            nn.Dropout(0.5),
            # nn.Linear(512,256),
            # nn.Dropout(0.5)
        )

        self.output = nn.Sequential(
            nn.Linear(line_length*3//2 * embedding_dim//4, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.Linear(64,1),
        )


    def encode(self, x): # x shape

        x = self.linear_encoding(x)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x) # 输入为 1 256 1024 /  输出为 output, intermediate_outputs  # 1 256 1024  / 48个 1 256 1024
        x = self.pre_head_ln(x) # layer norm

        return x, intmd_x

    def decode(self, x, intmd_x):
        intmd_x_1 = self.relu(self.linear_intmd_1(intmd_x['0']))
        intmd_x_2 = self.relu(self.linear_intmd_2(intmd_x['1']))
        intmd_x_3 = self.relu(self.linear_intmd_3(intmd_x['2']))
        intmd_x_4 = self.relu(self.linear_intmd_4(intmd_x['3']))

        intmd_x = self.relu(self.linear_cat1nd2(torch.cat([intmd_x_1,intmd_x_2],dim=1)))
        intmd_x = self.relu(self.linear_cat2nd3(torch.cat([intmd_x, intmd_x_3], dim=1)))
        intmd_x = self.relu(self.linear_cat3nd4(torch.cat([intmd_x, intmd_x_4], dim=1)))

        intmd_x = self.linear_intmd(intmd_x.view(self.batch_size,-1))
        x = self.linear_x(x.view(self.batch_size,-1))

        output = self.output(torch.cat([intmd_x,x],dim=1))

        return output


    def forward(self, x):
        encoder_output, intmd_encoder_outputs = self.encode(x)  # 使用 transformer encode 得到 1 256 embedding num(1024) / 中间层输出 48个输出
        decoder_output = self.decode(encoder_output, intmd_encoder_outputs)

        return decoder_output

class Transformer_lines_without_intmd(nn.Module):
    def __init__(self,input_dim,embedding_dim,num_heads,num_layers,hidden_dim,batch_size,line_length):
        super(Transformer_lines_without_intmd, self).__init__()
        self.batch_size = batch_size

        self.linear_encoding = nn.Linear(input_dim, embedding_dim)

        self.pe_dropout = nn.Dropout(0.8)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            dropout_rate=0.8,
            attn_dropout_rate=0.8
        )

        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        self.position_encoding = FixedPositionalEncoding(embedding_dim,line_length)

        self.relu = nn.ReLU()


        self.linear_x = nn.Sequential(
            nn.Linear(line_length * embedding_dim, line_length * embedding_dim//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(line_length * embedding_dim//2, line_length * embedding_dim//4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(line_length * embedding_dim//4,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )


    def encode(self, x): # x shape

        x = self.linear_encoding(x)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x) # 输入为 1 256 1024 /  输出为 output, intermediate_outputs  # 1 256 1024  / 48个 1 256 1024
        x = self.pre_head_ln(x) # layer norm

        return x, intmd_x

    def decode(self, x, intmd_x):
        x = self.linear_x(x.view(self.batch_size,-1))
        return x


    def forward(self, x):
        encoder_output, intmd_encoder_outputs = self.encode(x)  # 使用 transformer encode 得到 1 256 embedding num(1024) / 中间层输出 48个输出
        decoder_output = self.decode(encoder_output, intmd_encoder_outputs)

        return decoder_output



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

        with open(path+".txt", 'a', encoding='UTF-8') as f:
            print("epoch{},Test accuracy :{}".format(epoch,accuracy / step_test),file=f)
        print("Test accuracy :{}".format(accuracy / step_test))
        if accuracy / step_test < best_accuracy:
            best_accuracy = accuracy / step_test

    torch.save(model,path+".pth")
    return best_accuracy

if __name__ == "__main__":
    model = Transformer_lines_without_intmd(input_dim=16, embedding_dim=24, num_heads=4, num_layers=2, hidden_dim=64,
                                            # embedding 32时 到0.249
                                            batch_size=12,
                                            line_length=12)  # line length 为输入line组长度  input_dim 为每个encoding的长度    # 最高到0.2460
    best_accuracy = train_transformer(model, 12, "setr_lines12_24_4_2_64_12", 12,50)  # 4x4

    model = Transformer_lines_without_intmd(input_dim=32, embedding_dim=64, num_heads=4, num_layers=2, hidden_dim=128,
                                            # embedding 32时 到0.249
                                            batch_size=12,
                                            line_length=12)  # line length 为输入line组长度  input_dim 为每个encoding的长度    # 最高到0.2460
    best_accuracy = train_transformer(model, 32, "setr_lines32_64_4_2_128_12", 12,50)  # 4x4

    model = Transformer_lines_without_intmd(input_dim=64, embedding_dim=128, num_heads=4, num_layers=2, hidden_dim=256,
                                            # embedding 32时 到0.249
                                            batch_size=12,
                                            line_length=12)  # line length 为输入line组长度  input_dim 为每个encoding的长度    # 最高到0.2460
    best_accuracy = train_transformer(model, 64, "setr_lines64_128_4_2_256_12", 12,50)  # 4x4





