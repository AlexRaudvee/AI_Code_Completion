import torch
import random

import torch.nn as nn 
from data_loading import lang_in, lang_out
from config import system

if 'mac' in system.lower():
    system = 'mps' 
elif 'win' in system.lower():
    system = 'cpu'
else:
    raise TypeError('Unknown system, type: "Mac" or "Windows"')

device = torch.device('cuda' if torch.cuda.is_available() else system)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class EncDec(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncDec, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = target.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(lang_out.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        hidden, cell = self.encoder(source)

        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else top1
        
