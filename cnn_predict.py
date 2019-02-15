from collections import defaultdict
import torch
import os
from cnn_model import CNNclass

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

def read_dataset(filename):
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])

# Read in the data
data_path = "topicclass"
data_prefix = "topicclass_"
train = list(read_dataset(os.path.join(data_path, data_prefix + "train.txt")))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset(os.path.join(data_path, data_prefix + "valid.txt")))
test = list(read_dataset(os.path.join(data_path, data_prefix + "test.txt")))
nwords = len(w2i)
ntags = len(t2i)
ntrain = len(train)
i2t = {i: t for t, i in t2i.items()}

# Define the model
MODEL = "CNNCLASS"
EMB_SIZE = 64
WIN_SIZE = 3
FILTER_SIZE = 64
MODEL_PATH = os.path.join("best", MODEL)

model = CNNclass(nwords, EMB_SIZE, FILTER_SIZE, WIN_SIZE, ntags)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()

dev_correct = 0.0
dev_predicts = []
for words, tag in dev:
    # Padding (can be done in the conv layer as well)
    if len(words) < WIN_SIZE:
        words += [0] * (WIN_SIZE - len(words))
    words_tensor = torch.tensor(words).type(type)
    scores = model(words_tensor)[0]
    predict = scores.argmax().item()
    if predict == tag:
        dev_correct += 1
    dev_predicts.append(predict)

print("dev acc=%.4f" % (dev_correct / len(dev)))

test_predicts = []
for words, tag in test:
    # Padding (can be done in the conv layer as well)
    if len(words) < WIN_SIZE:
        words += [0] * (WIN_SIZE - len(words))
    words_tensor = torch.tensor(words).type(type)
    scores = model(words_tensor)[0]
    predict = scores.argmax().item()
    test_predicts.append(predict)

with open(MODEL + "_dev_res.txt", 'w', encoding="utf8") as f:
    for predict in dev_predicts:
        f.write("%s\n" % i2t[predict])

with open(MODEL + "_test_res.txt", 'w', encoding="utf8") as f:
    for predict in test_predicts:
        f.write("%s\n" % i2t[predict])
