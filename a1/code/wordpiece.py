import re
import time
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

PATH = './dataset/bpe/bpe_data.txt'
S1 = 'Analysts were expecting the opposite, a deepening of the deficit.'
S2 = 'Five minutes later, a second person arrived, aged around thirty, with knife wounds.'

def wordpiece_train(filename):
    vocab = Counter()
    step = 0
    new_vsize = 0
    vsize = []
    clen = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num >= 4000:
                break
            wrds = line.strip().split()
            for wrd in wrds:
                vocab[' '.join(wrd) + '</end>'] += 1

    base_vsize = len(set(' '.join(vocab.keys()).split()))

    while new_vsize < 4000:
        pairs = count(vocab)

        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        best_freq = pairs[best]
        vocab = merge(best, vocab)

        current_vsize = len(set(' '.join(vocab.keys()).split()))
        current_clen = sum(len(wrd.split()) * freq for wrd, freq in vocab.items())
        vsize.append(current_vsize)
        clen.append(current_clen)
        new_vsize = current_vsize - base_vsize
        step += 1
        print(f'Step {step},' + " Best pair: " + str(best), "Freq: " + str(best_freq), "New Vocab Size: ", new_vsize)

    wp_vocab = set()
    for wrd in vocab.keys():
        wp_vocab.update(wrd.split())

    return wp_vocab, vsize, clen

def wordpiece_apply(lines, wp_vocab):
    res = []
    num_tokens = 0

    for line in lines:
        tokens = []
        wrds = line.strip().split()

        for wrd in wrds:
            wrd += '</end>'
            wrd_len = len(wrd)

            while wrd_len > 0:
                found = False
                for i in range(wrd_len, 0, -1):
                    subwrd = wrd[:i]
                    if subwrd in wp_vocab:
                        tokens.append(subwrd)
                        wrd = wrd[i:]
                        wrd_len = len(wrd)
                        found = True
                        break

                if not found:
                    # If no subword is found, treat it as <unk>
                    tokens.append('<unk>')
                    break

        res.append(tokens)
        num_tokens += len(tokens)

    return res, num_tokens

# ====================== helper functions ======================
def plot(vsize, clen):
    print("Final vocab type size:", vsize[-1])
    print("Final training corpus length:", clen[-1])
    plt.scatter(vsize, clen)
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Token Length of Training Corpus')
    plt.title('WordPiece Vocabulary Size vs Token Length')
    plt.show()

def count(vocab):
    freqs = Counter()
    p = defaultdict(int)

    for wrd, freq in vocab.items():
        tokens = wrd.split()
        for token in tokens:
            freqs[token] += freq

    for wrd, freq in vocab.items():
        t = wrd.split()
        for i in range(len(t) - 1):
            pair = (t[i], t[i + 1])
            p[pair] += freq / (freqs[t[i]] * freqs[t[i + 1]])

    return p

def merge(pair, vocab):
    merged = {}
    bigram = re.escape(' '.join(pair))

    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    for wrd in vocab:
        merged[p.sub(''.join(pair), wrd)] = vocab[wrd]

    return merged

# ======================= Experiments ========================
time_start = time.time()
with open(PATH, 'r', encoding='utf-8') as f:
        test = f.readlines()[-1000:]

wp_vocab, vsize, clen = wordpiece_train(PATH)

_, num_tokens = wordpiece_apply(test, wp_vocab)
tokenized_s1, _ = wordpiece_apply([S1], wp_vocab)
tokenized_s2, _ = wordpiece_apply([S2], wp_vocab)

#------------------------Print Results------------------------
print("Total tokens in last 1000 lines:", num_tokens)
print()
print("Encoded sentence 1:", tokenized_s1)
print()
print("Encoded sentence 2:", tokenized_s2)
print()
time_end = time.time()
print("Running time:", time_end - time_start)
print()
plot(vsize, clen)