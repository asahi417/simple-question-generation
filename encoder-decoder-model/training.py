import os
import time
import math
import random
import json
from itertools import chain

import torch
import torch.nn as nn
from torch import optim
from datasets import load_dataset

from model import EncoderRNN, AttnDecoderRNN, WordIndexer, DEVICE, MAX_LENGTH, EOS_TOKEN, SOS_TOKEN, HIDDEN_SIZE

# Config
EPOCH = 3
TEACHER_FORCING_RATIO = 0.5
LEARNING_RATE = 0.01
PRINT_EVERY = 1000
DATASET = load_dataset("lmqg/qg_squad", split='train')
DATASET_TEST = load_dataset("lmqg/qg_squad", split='test')

PAIRS = list(zip(DATASET['paragraph_answer'], DATASET['question']))
PAIRS = [i for i in PAIRS if len(i[0].split(" ")) < MAX_LENGTH]
PAIRS_TEST = list(zip(DATASET_TEST['paragraph_answer'], DATASET_TEST['question']))
WORD_INDEXER = WordIndexer()
WORD_INDEXER.add_sentence(list(chain(*PAIRS)) + list(chain(*PAIRS_TEST)))
ENCODER = EncoderRNN(WORD_INDEXER.n_words).to(DEVICE)
DECODER = AttnDecoderRNN(WORD_INDEXER.n_words).to(DEVICE)


def sentence_to_tensor(sentence):
    indexes = [WORD_INDEXER.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def get_prediction(sentence):
    with torch.no_grad():
        input_tensor = sentence_to_tensor(sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = ENCODER.init_hidden()
        encoder_outputs = torch.zeros(MAX_LENGTH, HIDDEN_SIZE, device=DEVICE)
        for ei in range(input_length):
            encoder_output, encoder_hidden = ENCODER(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        decoder_input = torch.tensor([[SOS_TOKEN]], device=DEVICE)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = DECODER(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(WORD_INDEXER.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()
    return decoded_words


def train_single_epoch(input_tensor, target_tensor, encoder_optimizer, decoder_optimizer):

    criterion = nn.NLLLoss()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = ENCODER.init_hidden()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(MAX_LENGTH, HIDDEN_SIZE, device=DEVICE)
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = ENCODER(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    decoder_input = torch.tensor([[SOS_TOKEN]], device=DEVICE)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = DECODER(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = DECODER(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_TOKEN:
                break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train():

    def time_since(since, percent):

        def as_min(_s):
            m = math.floor(_s / 60)
            _s -= m * 60
            return '%dm %ds' % (m, _s)

        s = time.time() - since
        return '%s (- %s)' % (as_min(s), as_min(s / percent - s))

    def pair_to_tensor(pair):
        return sentence_to_tensor(pair[0]), sentence_to_tensor(pair[1])

    start = time.time()
    print_loss_total = 0  # Reset every print_every
    n_iters = len(PAIRS) * EPOCH
    encoder_optimizer = optim.SGD(ENCODER.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.SGD(DECODER.parameters(), lr=LEARNING_RATE)
    training_pairs = [pair_to_tensor(random.choice(PAIRS)) for i in range(n_iters)]

    for i in range(1, n_iters + 1):
        training_pair = training_pairs[i - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train_single_epoch(input_tensor, target_tensor, encoder_optimizer, decoder_optimizer)
        print_loss_total += loss

        if i % PRINT_EVERY == 0:
            print('%s (%d %d%%) %.4f' % (time_since(start, i / n_iters), i, i / n_iters * 100, print_loss_total / PRINT_EVERY))
            print_loss_total = 0
            for test_pair in PAIRS_TEST[:10]:
                print(f"PREDICTION: {test_pair[0]}\n\t*gene: {get_prediction(test_pair[0])}\n\t*gold: {test_pair[1]}\n")

    os.makedirs('model', exist_ok=True)
    torch.save(ENCODER.state_dict(), "model/encoder.pt")
    torch.save(DECODER.state_dict(), "model/decoder.pt")
    with open("model/id2token.json", "w") as f:
        json.dump(WORD_INDEXER.index2word, f)


if __name__ == '__main__':
    train()

