from collections import defaultdict
import time
import random
import torch
import numpy as np
import os
from cnn_model import CNNclass, CNN2d

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
MODEL = "CNN2D"
EMB_SIZE = 64
WIN_SIZE = 3
FILTER_SIZE = 64

# Training Params
BATCH_SIZE = 32
MAX_ITER = 10
LR = 0.0005
WEIGHT_DECAY = 0.0001

# initialize the model
model = CNN2d(nwords, EMB_SIZE, FILTER_SIZE, ntags)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()

best_dev = 0.0

for ITER in range(MAX_ITER):
    # Perform training
    model.train()
    random.shuffle(train)
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()

    for st in range(0, ntrain, BATCH_SIZE):
        ed = min(ntrain, st + BATCH_SIZE)

        tags = [train[i][1] for i in range(st, ed)]
        sents = [train[i][0] for i in range(st, ed)]
        nsents = len(sents)

        padlen = max(WIN_SIZE, len(max(sents, key=len)))
        for i in range(nsents):
            sents[i] += [0] * (padlen - len(sents[i]))

        sents_tensor = torch.tensor(sents).type(type)
        tags_tensor = torch.tensor(tags).type(type)

        scores = model(sents_tensor)
        predict = [score.argmax().item() for score in scores]
        train_correct += sum(1 for x, y in zip(predict, tags) if x == y)

        my_loss = criterion(scores, tags_tensor)
        train_loss += my_loss.item() * nsents
        # Do back-prop
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()

    print("iter %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" % (
        ITER, train_loss / len(train), train_correct / len(train), time.time() - start))

    # Cross validation
    conf_matrix = np.zeros((ntags, ntags), dtype=int)
    model.eval()
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
        conf_matrix[tag, predict] += 1
        dev_predicts.append(predict)

    print("iter %r: dev acc=%.4f" % (ITER, dev_correct / len(dev)))

    if dev_correct / len(dev) > best_dev:
        print("best so far found, predicting test dataset...")
        best_dev = dev_correct / len(dev)
        torch.save(model.state_dict(), MODEL)
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

        np.savetxt(MODEL + "dev_conf.txt", conf_matrix.astype(int), fmt='%i', delimiter=",")
        import json
        with open(MODEL + "dev_i2t.txt", 'w', encoding="utf8") as f:
            f.write(json.dumps(i2t))

        with open(MODEL + "_test_res.txt", 'w', encoding="utf8") as f:
            for predict in test_predicts:
                f.write("%s\n" % i2t[predict])

