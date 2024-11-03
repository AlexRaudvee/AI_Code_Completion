import torch 
import random
import string
import torch.nn as nn 
from config import device, file, first_run

from torch.utils.tensorboard import SummaryWriter


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x, hidden, cell):
        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[1], -1))
        return out, (hidden, cell)
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell
    
class Generator:
    def __init__(self, 
                chunk_len = 250,
                num_epochs = 5000,
                batch_size = 1,
                print_every = 50,
                hidden_size = 256,
                num_layers = 2,
                lr = 0.003):
        
        self.chunk_len = chunk_len
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.print_every = print_every
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.all_characters = string.printable
        self.n_characters = len(self.all_characters)


    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = self.all_characters.index(string[c])

        return tensor
    
    def get_random_batch(self):
        start_idx = random.randint(0, len(file) - self.chunk_len)
        end_idx = start_idx + self.chunk_len + 1
        text_str = file[start_idx: end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            text_input[i, :] = self.char_tensor(text_str[:-1])
            text_target[i, :] = self.char_tensor(text_str[1:])

        return text_input.long(), text_target.long()
    
    def generate(self, initial_str = 'A', predict_len = 200, temperature = 0.85):

        initial_str = initial_str.lower()

        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str)-1):
            _, (hidden, cell) = self.rnn(
                initial_input[p].view(1).to(device), hidden, cell
            )

        last_char = initial_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.rnn(
                last_char.view(1).to(device), hidden, cell
            )
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = self.all_characters[top_char]
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)

        return predicted
    
    
    def train(self):
        if first_run:
            self.rnn = RNN(
                self.n_characters, 
                self.hidden_size, 
                self.num_layers,
                self.n_characters
            ).to(device)

            optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()
            writer = SummaryWriter(f"runs/names0")  # for tensorboard

            print("=> Starting training")

            for epoch in range(1, self.num_epochs + 1):
                inp, target = self.get_random_batch()
                hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)

                self.rnn.zero_grad()
                loss = 0
                inp = inp.to(device)
                target = target.to(device)

                for c in range(self.chunk_len):
                    output, (hidden, cell) = self.rnn(inp[:, c], hidden, cell)
                    loss += criterion(output, target[:, c])

                loss.backward()
                optimizer.step()
                loss = loss.item() / self.chunk_len

                if epoch % self.print_every == 0:
                    print(f"Loss: {loss}, Epoch: {epoch}")
                    print(self.generate())

                writer.add_scalar("Training loss", loss, global_step=epoch)

            # save the model as a state dictionary in pth format
            torch.save({
                'model_state_dict': self.rnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },  'model_weights/model.pth')

        elif not first_run:
            # define the structure and the parameters
            self.rnn = RNN(self.n_characters, 
                            self.hidden_size, 
                            self.num_layers,
                            self.n_characters)
            
            # define the optimizer
            optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)

            # load the checkpoint data
            checkpoint = torch.load('model_weights/model.pth')
            # load the checkpoint into the model structure
            self.rnn.load_state_dict(checkpoint['model_state_dict'])
            # load the checkpoint to the optimizer structure
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # pass model to the device
            self.rnn.to(device)

        else: 
            raise TypeError(f"Input value must be boolean instead of {first_run} in config.py")