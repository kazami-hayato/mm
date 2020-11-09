import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time
import coloredlogs, logging


logger = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG')

# If you don't want to see log messages from libraries, you can pass a
# specific logger object to the install() function. In this case only log
# messages originating from that logger will show up on the terminal.
coloredlogs.install(level='DEBUG', logger=logger)

'''
init random seed
'''
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

'''
load data
'''
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    """
    tokenize a text from string to list string and reverse it
    :param text:
    :return:
    """
    return [token.text for token in spacy_de.tokenizer(text)][::-1]


def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)][::-1]


SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

'''
build vocabulary
'''
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

print(f'Unique tokens in source(de) vocabulary: {len(SRC.vocab)}')
print(f'Unique tokens in target(en) vocabulary: {len(TRG.vocab)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
iterator init
'''

batch_size = 128

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=batch_size, device=device
)

'''
encoder & decoder
'''


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        inp = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(inp))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "en&de hidden_dim must be same size"
        assert encoder.n_layers == decoder.n_layers, "en&de n_layers must be same size"

    def forward(self, src, trg, teacher_f_r=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab).to(self.device)
        hidden, cell = self.encoder(src)
        inp = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(inp, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_f_r
            top1 = output.argmax(1)
            inp = trg[t] if teacher_force else top1
        return outputs


INP_DIM = len(SRC.vocab)
OUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 4
ENC_DP = 0.5
DEC_DP = 0.5
enc = Encoder(INP_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DP)
dec = Decoder(OUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DP)

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'the model trainable parameters size is {count_parameters(model)}')
'''
optimizer
'''
optim = optim.Adam(model.parameters())

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train(model, iterator, optimizer, crt, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = crt(output, trg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, crt):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = crt(output, trg)

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(st, et):
    elp = et - st
    elp_min = int(elp / 60)
    elp_sec = int(elp - elp_min * 60)
    return elp_min, elp_sec


N_EPOCHS = 24

CLIP = 1
best_valid_loss = float('inf')

# Create a logger object.
for epoch in range(N_EPOCHS):
    start_t = time.time()
    train_loss = train(model, train_iter, optim, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)
    end_t = time.time()
    epoch_m, epoch_s = epoch_time(start_t, end_t)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
        logger.critical(f'Epoch: {epoch + 1:02} | Time: {epoch_m}m {epoch_s}s')
    logger.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    logger.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
