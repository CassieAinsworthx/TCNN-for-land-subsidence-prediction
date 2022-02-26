
from setr.SETRs import Transformer_lines_for_dual
from Nets_for_matrix.AlexNet_for_dual import AlexNet_4x4
import torch.nn as nn
import torch

class Dual_net(nn.Module):
    def __init__(self,input_dim,embedding_dim,num_heads,num_layers,hidden_dim,batch_size,line_length):
        super(Dual_net,self).__init__()

        self.AlexNet = AlexNet_4x4()
        self.batch_size = batch_size
        # self.Transformer = Transformer_for_dual(embedding_dim,num_heads,num_layers,hidden_dim,batch_size,input_dim)
        self.Transformer_line = Transformer_lines_for_dual(input_dim,embedding_dim,num_heads,num_layers,hidden_dim,batch_size,line_length)

        pretrained_dict=torch.load(r'E:\research_data\subsidence_exps\setr\setr_lines16_24_4_2_64_12.pth').state_dict()

        model_dict=self.Transformer_line.state_dict()

        checkpoint = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        checkpoint.update(model_dict)
        self.Transformer_line.load_state_dict(checkpoint)

        pretrained_dict=torch.load(r'E:\research_data\subsidence_exps\Nets_for_matrix\best_alexnet_fordual1.pth')#,map_location='cpu'  hrnet_w48_cityscapes_cls19_1024x2048_ohem_trainvalset.pth

        model_dict=self.AlexNet.state_dict()

        checkpoint = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        checkpoint.update(model_dict)
        self.AlexNet.load_state_dict(checkpoint)


        self.relu = nn.ReLU()

        self.linear = nn.Sequential(
            nn.Linear(1440,2048), #nn.Linear(embedding_dim*(input_dim+line_length)+128*2*2,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),  # nn.Linear(embedding_dim*(input_dim+line_length)+128*2*2,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),  # nn.Linear(embedding_dim*(input_dim+line_length)+128*2*2,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),  # nn.Linear(embedding_dim*(input_dim+line_length)+128*2*2,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self,x,y):
        # output_1 = self.relu(self.Transformer(z))
        output_2 = self.relu(self.Transformer_line(x))
        output_3 = self.relu(self.AlexNet(y))
        output_temp = torch.cat([output_2.view(self.batch_size,-1),output_3],dim=1)#output_1.view(self.batch_size,-1),
        output = self.linear(output_temp)

        return output
