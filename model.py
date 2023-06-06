import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class SimpleLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SimpleLSTM, self).__init__()
    
        self.model = nn.LSTM(input_dim,
                           hidden_dim,
                           num_layers=2,
                           bidirectional=True,
                           dropout=dropout_rate)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.leakrelu = nn.LeakyReLU()
        

    def forward(self, x):
        x, _  = self.model(x)
        output = self.fc(self.leakrelu(x))
        return output

class SimpleDNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(SimpleDNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        

    def forward(self, x):
        x = x.squeeze(1)
        x = self.fc2(self.fc1(x))
        return x

class BertModelFT(nn.Module):
    def __init__(self, hidden_dim, output_dim) -> None:
        super(BertModelFT, self).__init__()
        self.bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
        self.fc = nn.Linear(hidden_dim, 64)
        self.fc1 = nn.Linear(64, output_dim)
        
    def forward(self, x_ids, x_mask, finetune=True):
        # with torch.no_grad():
        bert_output = self.bert(x_ids, attention_mask=x_mask)
        bert_cls_hidden_state = bert_output[0][:,0,:]       #提取[CLS]对应的隐藏状态
        x = self.fc1(F.dropout(self.fc(F.dropout(bert_cls_hidden_state))))
        return x
    
if __name__ == '__main__':
    a = torch.randn(12, 1, 11)
    model = SimpleLSTM(11, 64, 2, 0.2)
    x = model(a)
    print(x.shape)
    
    
    
    
    
    
    