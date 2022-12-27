import torch
import torch.nn.functional as F
import torch.nn as nn
import spacy


class WordIndexer:

    def __init__(self, sentences, min_word_count: int = 5):
        self.eos = 'eos'
        self.unk = 'unk'
        self.sos = 'sos'
        self.nlp = spacy.load("en_core_web_sm")
        self.word_freq = {}
        # collection
        for sentence in sentences:
            for word in self.nlp.tokenizer(sentence):
                word = str(word).lower()
                if word in self.word_freq:
                    self.word_freq[word] += 1
                else:
                    self.word_freq[word] = 1
        # filter
        self.vocab = sorted([k for k, v in self.word_freq.items() if v > min_word_count])
        self.word2index = {self.sos: 0, self.eos: 1, self.unk: 2}
        word2index = {w: n + len(self.word2index) for n, w in enumerate(self.vocab)}
        self.word2index.update(word2index)
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.n_words = len(self.index2word)

    def encode(self, sentence):
        tokens = [str(word).lower() for word in self.nlp.tokenizer(sentence)]
        return [self.word2index[t] if t in self.word2index else self.word2index[self.unk] for t in tokens]

    def decode(self, id_list):
        return ' '.join([self.index2word[i] for i in id_list])


class EncoderRNN(nn.Module):

    def __init__(self, input_size, num_layers, hidden_size):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.num_layers)

    def forward(self, _input, hidden):
        return self.gru(self.embedding(_input).view(1, 1, -1), hidden)

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


class AttnDecoderRNN(nn.Module):

    def __init__(self, output_size, num_layers, hidden_size, dropout_p, max_length):
        super(AttnDecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, _input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(_input).view(1, 1, -1))
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = F.relu(self.attn_combine(torch.cat((embedded[0], attn_applied[0]), 1)).unsqueeze(0))
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

