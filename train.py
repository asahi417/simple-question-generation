import os
import time
import math
import random
import json
import logging
from itertools import chain
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from datasets import load_dataset

from model import EncoderRNN, AttnDecoderRNN, WordIndexer

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Config
EPOCH = 3
TEACHER_FORCING_RATIO = 0.5
LEARNING_RATE = 0.01
PRINT_EVERY = 100
MAX_LENGTH = 128
NUM_LAYERS = 4
HIDDEN_SIZE = 256
DROPOUT_P = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
random.seed(0)
DATASET = load_dataset("lmqg/qg_squad", split='train')
DATASET_TEST = load_dataset("lmqg/qg_squad", split='test')
PAIRS = list(zip(DATASET['sentence_answer'], DATASET['question']))
PAIRS_TEST = list(zip(DATASET_TEST['sentence_answer'], DATASET_TEST['question']))
random.shuffle(PAIRS_TEST)
WORD_INDEXER = WordIndexer(list(chain(*PAIRS)) + list(chain(*PAIRS_TEST)))
ENCODER = EncoderRNN(WORD_INDEXER.n_words, NUM_LAYERS, HIDDEN_SIZE).to(DEVICE)
DECODER = AttnDecoderRNN(WORD_INDEXER.n_words, NUM_LAYERS, HIDDEN_SIZE, DROPOUT_P, MAX_LENGTH).to(DEVICE)


def get_prediction(sentence):
    with torch.no_grad():
        input_tensor = WORD_INDEXER.sentence_to_tensor(sentence).to(DEVICE)
        input_length = input_tensor.size()[0]
        encoder_hidden = ENCODER.init_hidden().to(DEVICE)
        encoder_outputs = torch.zeros(MAX_LENGTH, HIDDEN_SIZE, device=DEVICE)
        for ei in range(input_length):
            encoder_output, encoder_hidden = ENCODER(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        decoder_input = torch.tensor([[WORD_INDEXER.word2index[WORD_INDEXER.sos]]], device=DEVICE)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = DECODER(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.data.topk(1)
            predicted_token = WORD_INDEXER.index2word[topi.item()]
            decoded_words.append(WORD_INDEXER.eos)
            if predicted_token == WORD_INDEXER.eos:
                break
            decoder_input = topi.squeeze().detach()
    return ' '.join(decoded_words)


def train_single_epoch(input_tensor, target_tensor, encoder_optimizer, decoder_optimizer):

    criterion = nn.NLLLoss()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = ENCODER.init_hidden().to(DEVICE)
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(MAX_LENGTH, HIDDEN_SIZE, device=DEVICE)
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = ENCODER(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    decoder_input = torch.tensor([[WORD_INDEXER.word2index[WORD_INDEXER.sos]]], device=DEVICE)
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
            print(decoder_input.item(), WORD_INDEXER.word2index[WORD_INDEXER.eos], decoder_input.item() == WORD_INDEXER.word2index[WORD_INDEXER.eos])
            input()
            if decoder_input.item() == WORD_INDEXER.word2index[WORD_INDEXER.eos]:
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

    start = time.time()
    print_loss_total = 0  # Reset every print_every
    n_iters = len(PAIRS) * EPOCH
    encoder_optimizer = optim.SGD(ENCODER.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.SGD(DECODER.parameters(), lr=LEARNING_RATE)

    for i in tqdm(list(range(1, n_iters + 1))):
        q, a = random.choice(PAIRS)
        input_tensor = WORD_INDEXER.sentence_to_tensor(q).to(DEVICE)
        target_tensor = WORD_INDEXER.sentence_to_tensor(a).to(DEVICE)
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        if input_length >= MAX_LENGTH or target_length >= MAX_LENGTH:
            continue

        loss = train_single_epoch(input_tensor, target_tensor, encoder_optimizer, decoder_optimizer)
        print_loss_total += loss

        if i % PRINT_EVERY == 0:
            logging.info('%s (%d %d%%) %.4f' % (time_since(start, i / n_iters), i, i / n_iters * 100, print_loss_total / PRINT_EVERY))
            print_loss_total = 0
            for test_pair in PAIRS_TEST[:5]:
                logging.info(f"PREDICTION: {test_pair[0]}\n\t*gene: {get_prediction(test_pair[0])}\n\t*gold: {test_pair[1]}\n")

    os.makedirs('model', exist_ok=True)
    torch.save(ENCODER.state_dict(), "model/encoder.pt")
    torch.save(DECODER.state_dict(), "model/decoder.pt")
    with open("model/id2token.json", "w") as f:
        json.dump(WORD_INDEXER.index2word, f)


if __name__ == '__main__':
    train()

