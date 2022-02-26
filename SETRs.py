import torch
import torch.nn as nn
from setr.Transformer import TransformerModel
from setr.PositionalEncoding import FixedPositionalEncoding
from setr.IntmdSequential import IntermediateSequential

line_length = 32

class Transformer(nn.Module):
    def __init__(self,embedding_dim,num_heads,num_layers,hidden_dim,batch_size,line_length):
        super(Transformer, self).__init__()
        self.batch_size = batch_size
        # encoding
        self.linear_encoding = nn.Linear(1, embedding_dim)

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


class Transformer_without_intmd(nn.Module):
    def __init__(self,embedding_dim,num_heads,num_layers,hidden_dim,batch_size,line_length):
        super(Transformer_without_intmd, self).__init__()
        self.batch_size = batch_size
        # encoding
        self.linear_encoding = nn.Linear(1, embedding_dim)

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

class Transformer_for_dual(nn.Module):
    def __init__(self,embedding_dim,num_heads,num_layers,hidden_dim,batch_size,line_length):
        super(Transformer_for_dual, self).__init__()
        self.batch_size = batch_size
        # encoding
        self.linear_encoding = nn.Linear(1, embedding_dim)

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

    def encode(self, x): # x shape

        x = self.linear_encoding(x)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x) # 输入为 1 256 1024 /  输出为 output, intermediate_outputs  # 1 256 1024  / 48个 1 256 1024
        x = self.pre_head_ln(x) # layer norm

        return x, intmd_x


    def forward(self, x):
        encoder_output, intmd_encoder_outputs = self.encode(x)  # 使用 transformer encode 得到 1 256 embedding num(1024) / 中间层输出 48个输出
        return encoder_output

class Transformer_lines_for_dual(nn.Module):
    def __init__(self,input_dim,embedding_dim,num_heads,num_layers,hidden_dim,batch_size,line_length):
        super(Transformer_lines_for_dual, self).__init__()
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


    def encode(self, x): # x shape

        x = self.linear_encoding(x)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x) # 输入为 1 256 1024 /  输出为 output, intermediate_outputs  # 1 256 1024  / 48个 1 256 1024
        x = self.pre_head_ln(x) # layer norm

        return x, intmd_x


    def forward(self, x):
        encoder_output, intmd_encoder_outputs = self.encode(x)  # 使用 transformer encode 得到 1 256 embedding num(1024) / 中间层输出 48个输出

        return encoder_output
