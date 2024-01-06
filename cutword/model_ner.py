import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
try:    
    from .pytorchcrf import CRF
except:
    from pytorchcrf import CRF



class LstmNerModel(nn.Module):
    def __init__(
            self, 
            embedding_size=256, 
            num_tags=41,
            vocab_size=3675, 
            hidden_size=128,
            batch_first=True, 
            dropout=0.1
            ):
        super(LstmNerModel, self).__init__()
        self.batch_first = batch_first
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_size, 
            dtype=torch.float32
        )

        self.lstm = nn.LSTM(
            embedding_size, 
            hidden_size // 2,
            num_layers=2, 
            batch_first=True,
            bidirectional=True, 
            dropout=dropout
        )
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

        self.fc = nn.Linear(hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
  

    def forward(self, input_tensor,seq_lens):
        # print(input_tensor.shape)
        input_tensor_1 = self.embedding(input_tensor)
        # print(input_tensor_1.shape)
        total_length = input_tensor_1.size(1) if self.batch_first else input_tensor_1.size(0)
        seq_lens = seq_lens.cpu()
        input_packed = pack_padded_sequence(input_tensor_1, seq_lens, batch_first=self.batch_first, enforce_sorted=False)
        output_lstm, hidden = self.lstm(input_packed)
        output_lstm, length = pad_packed_sequence(output_lstm, batch_first=self.batch_first, total_length=total_length)
        output_fc = self.fc(output_lstm)
        
        mask = torch.zeros(input_tensor.shape[:2]).to(input_tensor.device)
        # print(input_tensor.shape, mask.shape)
        mask = torch.greater(input_tensor, mask).type(torch.ByteTensor).to(input_tensor.device)

        return output_fc, mask


