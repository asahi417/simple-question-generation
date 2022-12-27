import json
import torch
from model import EncoderRNN, AttnDecoderRNN, DEVICE, EOS_TOKEN, MAX_LENGTH, HIDDEN_SIZE, SOS_TOKEN

with open("model/id2token.json") as f:
    id2token = json.load(f)
    token2id = {v: k for k, v in id2token.items()}

ENCODER = EncoderRNN(len(token2id)).to(DEVICE)
ENCODER.load_state_dict(torch.load("model/encoder.pt"))
DECODER = AttnDecoderRNN(len(token2id)).to(DEVICE)
DECODER.load_state_dict(torch.load("model/decoder.pt"))


def sentence_to_tensor(sentence):
    indexes = [token2id[word] for word in sentence.split(' ')]
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
                decoded_words.append(id2token[topi.item()])
            decoder_input = topi.squeeze().detach()
    return decoded_words

