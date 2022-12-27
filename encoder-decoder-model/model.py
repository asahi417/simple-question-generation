import torch
import torch.nn.functional as F
import torch.nn as nn

# Config
SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 256
HIDDEN_SIZE = 256
DROPOUT = 0.1
NUM_LAYERS_ENCODER = 4
NUM_LAYERS_DECODER = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WordIndexer:

    def __init__(self):
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentences):
        for sentence in sentences:
            for word in sentence.split(' '):
                if word not in self.word2index:
                    self.word2index[word] = self.n_words
                    self.index2word[self.n_words] = word
                    self.n_words += 1


class EncoderRNN(nn.Module):

    def __init__(self, input_size):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, HIDDEN_SIZE)
        self.gru = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE, num_layers=NUM_LAYERS_ENCODER)

    def forward(self, _input, hidden):
        return self.gru(self.embedding(_input).view(1, 1, -1), hidden)

    @staticmethod
    def init_hidden():
        return torch.zeros(NUM_LAYERS_ENCODER, 1, HIDDEN_SIZE, device=DEVICE)


class AttnDecoderRNN(nn.Module):

    def __init__(self, output_size):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, HIDDEN_SIZE)
        self.attn = nn.Linear(HIDDEN_SIZE * 2, MAX_LENGTH)
        self.attn_combine = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)
        self.dropout = nn.Dropout(DROPOUT)
        self.gru = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE, num_layers=NUM_LAYERS_DECODER)
        self.out = nn.Linear(HIDDEN_SIZE, output_size)

    def forward(self, _input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(_input).view(1, 1, -1))
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = F.relu(self.attn_combine(torch.cat((embedded[0], attn_applied[0]), 1)).unsqueeze(0))
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    @staticmethod
    def init_hidden():
        return torch.zeros(NUM_LAYERS_DECODER, 1, HIDDEN_SIZE, device=DEVICE)

