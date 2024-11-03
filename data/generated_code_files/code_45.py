import torch

import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import BucketIterator

from func import translate_sentence, bleu, save_checkpoint, load_checkpoint
from data_loading import lang_in, lang_out, train_data, valid_data, test_data
from Net import Encoder, Decoder, EncDec, device

# Training hyperparameters
num_epochs = 100
learning_rate = 0.001
batch_size = 64

# Model hyperparameters
load_model = False
input_size_encoder = len(lang_in.vocab)
input_size_decoder = len(lang_out.vocab)
output_size = len(lang_out.vocab)#<PREDICTION>

#</PREDICTION>
decoder_embedding_size = 300
hidden_size = 1024  # Needs to be the same for both RNN's
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard to get nice loss plot
writer = SummaryWriter(f"runs/loss_plot")
step = 0


train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    device=device,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    repeat=False
)

encoder_net = Encoder(
    input_size_encoder,
    encoder_embedding_size,
    hidden_size,
    num_layers,
    dec_dropout,
    output_size
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    dec_dropout).to(device)

model = EncDec(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters, lr=learning_rate)

pad_idx = lang_out.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("model.pth"), model, optimizer)

sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(
        model, sentence, lang_in, lang_out, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N).
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1


score = bleu(test_data[1:100], model, lang_in, lang_out, device)
print(f"Bleu score {score*100:.2f}")