import math
import numpy as np
from collections import Counter

DEV = 'dataset/n_gram/1b_benchmark.dev.tokens'
TRAIN = 'dataset/n_gram/1b_benchmark.train.tokens'
TEST = 'dataset/n_gram/1b_benchmark.test.tokens'
START = '<START>'
STOP = '<STOP>'
OOV = '<UNK>'

def unigram(data, laplace=False):
    unigram_counter = Counter(token for seq in data for token in seq if token != START)
    tot = sum(unigram_counter.values())
    vsize = len(set(token for seq in data for token in seq))

    if laplace:
        unigram = {token: (count + 1) / (tot + vsize) for token, count in unigram_counter.items()}
    else:
        unigram = {token: count / tot for token, count in unigram_counter.items()}

    return unigram


def bigram(data, laplace=False):
    bigram_counter = Counter()
    first_token_counter = Counter()
    vsize = len(set(token for seq in data for token in seq))

    for seq in data:
        bigrams = list(zip(seq, seq[1:]))
        bigram_counter.update(bigrams)
        first_tokens = seq[:-1]
        first_token_counter.update(first_tokens)

    if laplace:
        bigram = {bigram: (count + 1) / (first_token_counter[bigram[0]] + vsize) for bigram, count in bigram_counter.items()}
    else:
        bigram = {bigram: count / first_token_counter[bigram[0]] for bigram, count in bigram_counter.items()}

    return bigram


def trigram(data, laplace=False):
    trigram_counter = Counter()
    first_two_counter = Counter()
    vsize = len(set(token for seq in data for token in seq))

    for seq in data:
        seq = [START, START] + seq + [STOP]

        for i in range(2, len(seq)):
            trigram = (seq[i-2], seq[i-1], seq[i])
            bigram = (seq[i-2], seq[i-1])

            if trigram != (START, START, START):
                trigram_counter[trigram] += 1
                first_two_counter[bigram] += 1

    if laplace:
        trigram = {trigram: (count + 1) / (first_two_counter[(trigram[0], trigram[1])] + vsize)
                         for trigram, count in trigram_counter.items()}
    else:
        trigram = {trigram: count / first_two_counter[(trigram[0], trigram[1])]
                         for trigram, count in trigram_counter.items()}

    return trigram



def perplexity(model, data, n):
    pps = []
    lens = []

    for seq in data:
        length = len(seq)

        if n == 1:
            # Unigram model
            log_probs = [math.log(model[token]) if token in model else float('inf')
                                 for token in seq if token != START]
        elif n == 2:
            # Bigram model
            bigrams = list(zip(seq, seq[1:]))
            log_probs = [math.log(model[bigram]) if bigram in model else float('inf')
                                 for bigram in bigrams]
        else:
            # Trigram model
            trigrams = list(zip(seq, seq[1:], seq[2:]))
            log_probs = [math.log(model[trigram]) if trigram in model else float('inf')
                                 for trigram in trigrams]

        # Calculate perplexity for the sequence
        if float('inf') in log_probs:
            # Infinite perplexity if any token/n-gram is not in the model
            seq_pp = float('inf')
        else:
            log_prob = sum(log_probs)
            seq_pp = math.exp(-log_prob / length)

        pps.append(seq_pp)
        lens.append(length)

    # Weighted average of perplexities across all sequences
    pp = np.average(pps, weights=lens)

    return pp


def perplexity_laplace(model, data, n, vsize):
    pps = []
    lens = []

    for seq in data:
        length = len(seq)

        if n == 1:
            # Unigram
            log_probs = [math.log(model.get(token, 1 / vsize)) for token in seq if token != START]
        elif n == 2:
            # Bigram
            bigrams = list(zip(seq, seq[1:]))
            log_probs = [math.log(model.get(bigram, 1 / vsize)) for bigram in bigrams]
        else:
            # Trigram
            trigrams = list(zip(seq, seq[1:], seq[2:]))
            log_probs = [math.log(model.get(trigram, 1 / vsize)) for trigram in trigrams]

        log_prob = sum(log_probs)
        seq_pp = math.exp(-log_prob / length)
        pps.append(seq_pp)
        lens.append(length)

    # Weighted average of perplexities across all sequences
    pp = np.average(pps, weights=lens)
    return pp

# ====================== helper functions ======================
def build_vocabulary(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    counter = Counter()
    for line in lines:
        tokens = line.strip().split()
        counter.update(tokens)

    unks = [token for token, count in counter.items() if count < 3]

    for token in unks:
        counter[OOV] += counter[token]
        del counter[token]

    vocab = set(counter.keys())
    vocab.update([STOP])

    data = []
    for line in lines:
        tokens = [START] + [token if token in vocab else OOV for token in line.strip().split()] + [STOP]
        data.append(tokens)

    return data, vocab

def vocab_for_non_train_data(path, training_vocab):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        # Tokenize each line and replace tokens not in training vocabulary with OOV
        tokens = [START] + [token if token in training_vocab else OOV for token in line.strip().split()] + [STOP]
        data.append(tokens)

    return data

def interpolation(token, prev_token, prev_prev_token, unigram, bigram, trigram, lambdas):
    lambda1, lambda2, lambda3 = lambdas

    unigram_prob = unigram.get(token, 0)

    bigram_prob = bigram.get((prev_token, token), 0) if prev_token is not None else 0

    trigram_prob = trigram.get((prev_prev_token, prev_token, token), 0) if prev_prev_token is not None and prev_token is not None else 0

    interp_prob = lambda1 * unigram_prob + lambda2 * bigram_prob + lambda3 * trigram_prob
    return interp_prob

def perplexity_interpolated(data, unigram, bigram, trigram, lambdas):
    pps = []
    lens = []

    for seq in data:
        length = len(seq) - 2  # Adjust for START tokens
        log_probs = []

        for i in range(2, len(seq)):
            token = seq[i]
            prev_token = seq[i - 1]
            prev_prev_token = seq[i - 2]

            prob = interpolation(token, prev_token, prev_prev_token, unigram, bigram, trigram, lambdas)
            if prob > 0:
                log_probs.append(math.log(prob))
            else:
                log_probs.append(float('inf'))

        if float('inf') in log_probs:
            seq_pp = float('inf')
        else:
            log_prob = sum(log_probs)
            seq_pp = math.exp(-log_prob / length)

        pps.append(seq_pp)
        lens.append(length)

    pp = np.average(pps, weights=lens)
    return pp

# ======================= Experiments ========================
lambda1 = (0.33, 0.33, 0.34)
lambda2 = (0.2, 0.3, 0.5)
lambda3 = (0.2, 0.2, 0.6)
lambda4 = (0.1, 0.15, 0.75)
lambda5 = (0.05, 0.08, 0.87)
lambdas6 = (0.1, 0.3, 0.6)


processed_train_data, train_vocabulary = build_vocabulary(TRAIN)
processed_dev_data = vocab_for_non_train_data(DEV, train_vocabulary)
processed_test_data = vocab_for_non_train_data(TEST, train_vocabulary)

unigram_model = unigram(processed_train_data)
bigram_model = bigram(processed_train_data)
trigram_model = trigram(processed_train_data)

unigram_model_laplace = unigram(processed_train_data, laplace=True)
bigram_model_laplace = bigram(processed_train_data, laplace=True)
trigram_model_laplace = trigram(processed_train_data, laplace=True)

training_perplexity_unigram = perplexity(unigram_model, processed_train_data, 1)
training_perplexity_bigram = perplexity(bigram_model, processed_train_data, 2)
training_perplexity_trigram = perplexity(trigram_model, processed_train_data, 3)

dev_perplexity_unigram = perplexity(unigram_model, processed_dev_data, 1)
dev_perplexity_bigram = perplexity(bigram_model, processed_dev_data, 2)
dev_perplexity_trigram = perplexity(trigram_model, processed_dev_data, 3)

test_perplexity_unigram = perplexity(unigram_model, processed_test_data, 1)
test_perplexity_bigram = perplexity(bigram_model, processed_test_data, 2)
test_perplexity_trigram = perplexity(trigram_model, processed_test_data, 3)

training_perplexity_unigram_laplace = perplexity_laplace(unigram_model_laplace, processed_train_data, 1, len(train_vocabulary))
training_perplexity_bigram_laplace = perplexity_laplace(bigram_model_laplace, processed_train_data, 2, len(train_vocabulary))
training_perplexity_trigram_laplace = perplexity_laplace(trigram_model_laplace, processed_train_data, 3, len(train_vocabulary))

dev_perplexity_unigram_laplace = perplexity_laplace(unigram_model_laplace, processed_dev_data, 1, len(train_vocabulary))
dev_perplexity_bigram_laplace = perplexity_laplace(bigram_model_laplace, processed_dev_data, 2, len(train_vocabulary))
dev_perplexity_trigram_laplace = perplexity_laplace(trigram_model_laplace, processed_dev_data, 3, len(train_vocabulary))

test_perplexity_unigram_laplace = perplexity_laplace(unigram_model_laplace, processed_test_data, 1, len(train_vocabulary))
test_perplexity_bigram_laplace = perplexity_laplace(bigram_model_laplace, processed_test_data, 2, len(train_vocabulary))
test_perplexity_trigram_laplace = perplexity_laplace(trigram_model_laplace, processed_test_data, 3, len(train_vocabulary))

training_perplexity_interpolated1 = perplexity_interpolated(processed_train_data, unigram_model, bigram_model, trigram_model, lambda1)
dev_perplexity_interpolated1 = perplexity_interpolated(processed_dev_data, unigram_model, bigram_model, trigram_model, lambda1)
test_perplexity_interpolated1 = perplexity_interpolated(processed_test_data, unigram_model, bigram_model, trigram_model, lambda1)

training_perplexity_interpolated2 = perplexity_interpolated(processed_train_data, unigram_model, bigram_model, trigram_model, lambda2)
dev_perplexity_interpolated2 = perplexity_interpolated(processed_dev_data, unigram_model, bigram_model, trigram_model, lambda2)
test_perplexity_interpolated2 = perplexity_interpolated(processed_test_data, unigram_model, bigram_model, trigram_model, lambda2)

training_perplexity_interpolated3 = perplexity_interpolated(processed_train_data, unigram_model, bigram_model, trigram_model, lambda3)
dev_perplexity_interpolated3 = perplexity_interpolated(processed_dev_data, unigram_model, bigram_model, trigram_model, lambda3)
test_perplexity_interpolated3 = perplexity_interpolated(processed_test_data, unigram_model, bigram_model, trigram_model, lambda3)

training_perplexity_interpolated4 = perplexity_interpolated(processed_train_data, unigram_model, bigram_model, trigram_model, lambda4)
dev_perplexity_interpolated4 = perplexity_interpolated(processed_dev_data, unigram_model, bigram_model, trigram_model, lambda4)
test_perplexity_interpolated4 = perplexity_interpolated(processed_test_data, unigram_model, bigram_model, trigram_model, lambda4)

training_perplexity_interpolated5 = perplexity_interpolated(processed_train_data, unigram_model, bigram_model, trigram_model, lambda5)
dev_perplexity_interpolated5 = perplexity_interpolated(processed_dev_data, unigram_model, bigram_model, trigram_model, lambda5)
test_perplexity_interpolated5 = perplexity_interpolated(processed_test_data, unigram_model, bigram_model, trigram_model, lambda5)

training_perplexity_interpolated6 = perplexity_interpolated(processed_train_data, unigram_model, bigram_model, trigram_model, lambdas6)
dev_perplexity_interpolated6 = perplexity_interpolated(processed_dev_data, unigram_model, bigram_model, trigram_model, lambdas6)
test_perplexity_interpolated6 = perplexity_interpolated(processed_test_data, unigram_model, bigram_model, trigram_model, lambdas6)

#------------------------Print Results------------------------

print('Train vocabulary size:', len(train_vocabulary))
print()
print('Unigram model size:', len(unigram_model))
print('Bigram model size:', len(bigram_model))
print('Trigram model size:', len(trigram_model))
print()
print("Training set perplexity:" + "\n" +
      "Unigram: " + str(training_perplexity_unigram) + "\n" +
      "Bigram: " + str(training_perplexity_bigram) + "\n" +
      "Trigram: " + str(training_perplexity_trigram) + "\n")
print()
print("Dev set perplexity:" + "\n" +
      "Unigram: " + str(dev_perplexity_unigram) + "\n" +
      "Bigram: " + str(dev_perplexity_bigram) + "\n" +
      "Trigram: " + str(dev_perplexity_trigram) + "\n")
print()
print("Test set perplexity:" + "\n" +
      "Unigram: " + str(test_perplexity_unigram) + "\n" +
      "Bigram: " + str(test_perplexity_bigram) + "\n" +
      "Trigram: " + str(test_perplexity_trigram) + "\n")
print("Training set perplexity with Laplace smoothing:" + "\n" +
      "Unigram: " + str(training_perplexity_unigram_laplace) + "\n" +
      "Bigram: " + str(training_perplexity_bigram_laplace) + "\n" +
      "Trigram: " + str(training_perplexity_trigram_laplace) + "\n")
print("Dev set perplexity with Laplace smoothing:" + "\n" +
        "Unigram: " + str(dev_perplexity_unigram_laplace) + "\n" +
        "Bigram: " + str(dev_perplexity_bigram_laplace) + "\n" +
        "Trigram: " + str(dev_perplexity_trigram_laplace) + "\n")
print("Test set perplexity with Laplace smoothing:" + "\n" +
        "Unigram: " + str(test_perplexity_unigram_laplace) + "\n" +
        "Bigram: " + str(test_perplexity_bigram_laplace) + "\n" +
        "Trigram: " + str(test_perplexity_trigram_laplace) + "\n")
print()
print("Training set perplexity with interpolation(0.33, 0.33, 0.34): " + str(training_perplexity_interpolated1))
print("Dev set perplexity with interpolation(0.33, 0.33, 0.34): " + str(dev_perplexity_interpolated1))
print("Test set perplexity with interpolation(0.33, 0.33, 0.34): " + str(test_perplexity_interpolated1))
print()
print("Training set perplexity with interpolation(0.2, 0.3, 0.5): " + str(training_perplexity_interpolated2) )
print("Dev set perplexity with interpolation(0.2, 0.3, 0.5): " + str(dev_perplexity_interpolated2) )
print("Test set perplexity with interpolation(0.2, 0.3, 0.5): " + str(test_perplexity_interpolated2) )
print()
print("Training set perplexity with interpolation(0.2, 0.2, 0.6): " + str(training_perplexity_interpolated3) )
print("Dev set perplexity with interpolation(0.2, 0.2, 0.6): " + str(dev_perplexity_interpolated3) )
print("Test set perplexity with interpolation(0.2, 0.2, 0.6): " + str(test_perplexity_interpolated3) )
print()
print("Training set perplexity with interpolation(0.1, 0.15, 0.75): " + str(training_perplexity_interpolated4) )
print("Dev set perplexity with interpolation(0.1, 0.15, 0.75): " + str(dev_perplexity_interpolated4) )
print("Test set perplexity with interpolation(0.1, 0.15, 0.75): " + str(test_perplexity_interpolated4) )
print()
print("Training set perplexity with interpolation(0.05, 0.08, 0.87): " + str(training_perplexity_interpolated5) )
print("Dev set perplexity with interpolation(0.05, 0.08, 0.87): " + str(dev_perplexity_interpolated5) )
print("Test set perplexity with interpolation(0.05, 0.08, 0.87): " + str(test_perplexity_interpolated5) )
print()
print("Training set perplexity with interpolation(0.1, 0.3, 0.6): " + str(training_perplexity_interpolated6) )
print("Dev set perplexity with interpolation(0.1, 0.3, 0.6): " + str(dev_perplexity_interpolated6) )
print("Test set perplexity with interpolation(0.1, 0.3, 0.6): " + str(test_perplexity_interpolated6) )