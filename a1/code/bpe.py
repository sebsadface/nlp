import re
import time
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

PATH = './dataset/bpe/bpe_data.txt'
S1 = 'Analysts were expecting the opposite, a deepening of the deficit.'
S2 = 'Five minutes later, a second person arrived, aged around thirty, with knife wounds.'

def bpe_train(filename):
    vocab = Counter()
    best_freq = 99999
    step = 0
    vsize = []
    clen = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num >= 4000:
                break
            wrds = line.strip().split()
            for wrd in wrds:
                vocab[' '.join(wrd) + '</end>'] += 1

    while best_freq > 2:
        p = count(vocab)

        if not p:
            break
        best = max(p, key=p.get)
        print(f'Step {step},' + " Best pair: " + str(best) + ", Freq: " + str(best_freq))

        vocab = merge(best, vocab)
        best_freq = p.get(best)

        current_vsize = len(set(' '.join(vocab.keys()).split()))
        current_clen = sum(len(wrd.split()) * freq for wrd, freq in vocab.items())
        vsize.append(current_vsize)
        clen.append(current_clen)
        step += 1

    bpe_vocab = set()
    for wrd in vocab.keys():
        bpe_vocab.update(wrd.split())

    return bpe_vocab, vsize, clen


def bpe_apply(lines, bpe_vocab):
    res = []
    num_tokens = 0

    for line in lines:
        line = line.strip()
        wrds = line.split()
        tokenized_line = []

        for wrd in wrds:
            tokenized_wrd = tokenize(wrd + '</end>', bpe_vocab)
            tokenized_line.extend(tokenized_wrd)
            num_tokens += len(tokenized_wrd)

        res.append(tokenized_line)

    return res, num_tokens

# ====================== helper functions ======================
def plot (vsize, clen):
    print()
    print("Final vocab type size: " + str(vsize[-1]))
    print("Final training corpus length: " + str(clen[-1]))
    print()
    plt.scatter(vsize, clen)
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Token Length of Training Corpus')
    plt.title('BPE Vocabulary Size vs Token Length')
    plt.show()

def count(vocab):
    p = defaultdict(int)
    for wrd, freq in vocab.items():
        t = wrd.split()
        for i in range(len(t) - 1):
            p[t[i], t[i + 1]] += freq
    return p

def merge(pair, vocab):
    merged = {}
    bigram = re.escape(' '.join(pair))

    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    for wrd in vocab:
        merged[p.sub(''.join(pair), wrd)] = vocab[wrd]

    return merged

def tokenize(string, bpe_vocab):
        if string == '':
            return []
        if len(string) == 1:
            return [string]

        for i in range(len(string), -1, -1):
            t = string[:i]
            if t in bpe_vocab:
                return [t] + tokenize(string[i:], bpe_vocab)

        return [string]

# ======================= Experiments ========================
time_start = time.time()
with open(PATH, 'r', encoding='utf-8') as f:
        test = f.readlines()[-1000:]

bpe_vocab, vsize, clen = bpe_train(PATH)

_, num_tokens = bpe_apply(test, bpe_vocab)
tokenized_s1, _ = bpe_apply([S1], bpe_vocab)
tokenized_s2, _ = bpe_apply([S2], bpe_vocab)

#------------------------Print Results------------------------
print()
print("total tokens in last 1000 lines:", num_tokens)
print("Encoded sentence 1:", tokenized_s1)
print()
print("Encoded sentence 2:", tokenized_s2)
print()
time_end = time.time()
print("Running time: " + str(time_end - time_start))
print()
plot(vsize, clen)

