import math
import numpy as np
from collections import Counter

DEV_PATH = 'dataset/n_gram/1b_benchmark.dev.tokens'
TRAIN_PATH = 'dataset/n_gram/1b_benchmark.train.tokens'
TEST_PATH = 'dataset/n_gram/1b_benchmark.test.tokens'
START_TOKEN = '<START>'
STOP_TOKEN = '<STOP>'
OOV = '<UNK>'


def build_vocabulary(file_path):
    # Read data from file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Tokenize and count token frequency
    token_counter = Counter()
    for line in lines:
        tokens = line.strip().split()
        token_counter.update(tokens)

    # Identify rare tokens
    rare_tokens = [token for token, count in token_counter.items() if count < 3]

    # Replace rare tokens with OOV
    for token in rare_tokens:
        token_counter[OOV] += token_counter[token]
        del token_counter[token]

    # Add special tokens to the vocabulary
    vocabulary = set(token_counter.keys())
    vocabulary.update([STOP_TOKEN])

    # Process data for n-gram models
    processed_data = []
    for line in lines:
        tokens = [START_TOKEN] + [token if token in vocabulary else OOV for token in line.strip().split()] + [STOP_TOKEN]
        processed_data.append(tokens)

    return processed_data, vocabulary

def build_vocabulary_for_nontraining_data(file_path, training_vocabulary):
    # Read data from file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Process data for n-gram models
    processed_data = []
    for line in lines:
        # Tokenize each line and replace tokens not in training vocabulary with OOV
        tokens = [START_TOKEN] + [token if token in training_vocabulary else OOV for token in line.strip().split()] + [STOP_TOKEN]
        processed_data.append(tokens)

    return processed_data


def unigram(processed_data, laplace=False):
    unigram_counter = Counter(token for sentence in processed_data for token in sentence if token != START_TOKEN)
    total_tokens = sum(unigram_counter.values())
    vocab_size = len(set(token for sentence in processed_data for token in sentence))

    if laplace:
        unigram_model = {token: (count + 1) / (total_tokens + vocab_size) for token, count in unigram_counter.items()}
    else:
        unigram_model = {token: count / total_tokens for token, count in unigram_counter.items()}

    return unigram_model


def bigram(processed_data, laplace=False):
    bigram_counter = Counter()
    first_token_counter = Counter()
    vocab_size = len(set(token for sentence in processed_data for token in sentence))

    for sentence in processed_data:
        bigrams = list(zip(sentence, sentence[1:]))
        bigram_counter.update(bigrams)
        first_tokens = sentence[:-1]
        first_token_counter.update(first_tokens)

    if laplace:
        bigram_model = {bigram: (count + 1) / (first_token_counter[bigram[0]] + vocab_size) for bigram, count in bigram_counter.items()}
    else:
        bigram_model = {bigram: count / first_token_counter[bigram[0]] for bigram, count in bigram_counter.items()}

    return bigram_model


def trigram(processed_data, laplace=False):
    trigram_counter = Counter()
    first_two_token_counter = Counter()
    vocab_size = len(set(token for sentence in processed_data for token in sentence))

    for sentence in processed_data:
        sentence = [START_TOKEN, START_TOKEN] + sentence + [STOP_TOKEN]

        for i in range(2, len(sentence)):
            trigram = (sentence[i-2], sentence[i-1], sentence[i])
            bigram = (sentence[i-2], sentence[i-1])

            if trigram != (START_TOKEN, START_TOKEN, START_TOKEN):
                trigram_counter[trigram] += 1
                first_two_token_counter[bigram] += 1

    if laplace:
        trigram_model = {trigram: (count + 1) / (first_two_token_counter[(trigram[0], trigram[1])] + vocab_size)
                         for trigram, count in trigram_counter.items()}
    else:
        trigram_model = {trigram: count / first_two_token_counter[(trigram[0], trigram[1])]
                         for trigram, count in trigram_counter.items()}

    return trigram_model



def perplexity(ngram_model, processed_data, n):
    perplexities = []
    lengths = []

    for sequence in processed_data:
        length = len(sequence)

        if n == 1:
            # Unigram model
            log_probabilities = [math.log(ngram_model[token]) if token in ngram_model else float('inf')
                                 for token in sequence if token != START_TOKEN]
        elif n == 2:
            # Bigram model
            bigrams = list(zip(sequence, sequence[1:]))
            log_probabilities = [math.log(ngram_model[bigram]) if bigram in ngram_model else float('inf')
                                 for bigram in bigrams]
        else:
            # Trigram model
            trigrams = list(zip(sequence, sequence[1:], sequence[2:]))
            log_probabilities = [math.log(ngram_model[trigram]) if trigram in ngram_model else float('inf')
                                 for trigram in trigrams]

        # Calculate perplexity for the sequence
        if float('inf') in log_probabilities:
            # Infinite perplexity if any token/n-gram is not in the model
            seq_perplexity = float('inf')
        else:
            log_probability = sum(log_probabilities)
            seq_perplexity = math.exp(-log_probability / length)

        perplexities.append(seq_perplexity)
        lengths.append(length)

    # Weighted average of perplexities across all sequences
    perplexity = np.average(perplexities, weights=lengths)

    return perplexity


def perplexity_laplace(ngram_model, processed_data, n, vocab_size):
    perplexities = []
    lengths = []

    for sequence in processed_data:
        length = len(sequence)

        if n == 1:
            # Unigram model
            log_probabilities = [math.log(ngram_model.get(token, 1 / vocab_size)) for token in sequence if token != START_TOKEN]
        elif n == 2:
            # Bigram model
            bigrams = list(zip(sequence, sequence[1:]))
            log_probabilities = [math.log(ngram_model.get(bigram, 1 / vocab_size)) for bigram in bigrams]
        else:
            # Trigram model
            trigrams = list(zip(sequence, sequence[1:], sequence[2:]))
            log_probabilities = [math.log(ngram_model.get(trigram, 1 / vocab_size)) for trigram in trigrams]

        log_probability = sum(log_probabilities)
        seq_perplexity = math.exp(-log_probability / length)
        perplexities.append(seq_perplexity)
        lengths.append(length)

    # Weighted average of perplexities across all sequences
    perplexity = np.average(perplexities, weights=lengths)
    return perplexity


processed_train_data, train_vocabulary = build_vocabulary(TRAIN_PATH)
processed_dev_data = build_vocabulary_for_nontraining_data(DEV_PATH, train_vocabulary)
processed_test_data = build_vocabulary_for_nontraining_data(TEST_PATH, train_vocabulary)


print('Train vocabulary size:', len(train_vocabulary))
print()

unigram_model = unigram(processed_train_data)
bigram_model = bigram(processed_train_data)
trigram_model = trigram(processed_train_data)

print('Unigram model size:', len(unigram_model))
print('Bigram model size:', len(bigram_model))
print('Trigram model size:', len(trigram_model))
print()

unigram_model_laplace = unigram(processed_train_data, laplace=True)
bigram_model_laplace = bigram(processed_train_data, laplace=True)
trigram_model_laplace = trigram(processed_train_data, laplace=True)

training_perplexity_unigram = perplexity(unigram_model, processed_train_data, 1)
training_perplexity_bigram = perplexity(bigram_model, processed_train_data, 2)
training_perplexity_trigram = perplexity(trigram_model, processed_train_data, 3)
print("Training set perplexity:" + "\n" +
      "Unigram: " + str(training_perplexity_unigram) + "\n" +
      "Bigram: " + str(training_perplexity_bigram) + "\n" +
      "Trigram: " + str(training_perplexity_trigram) + "\n")
print()

dev_perplexity_unigram = perplexity(unigram_model, processed_dev_data, 1)
dev_perplexity_bigram = perplexity(bigram_model, processed_dev_data, 2)
dev_perplexity_trigram = perplexity(trigram_model, processed_dev_data, 3)
print("Dev set perplexity:" + "\n" +
      "Unigram: " + str(dev_perplexity_unigram) + "\n" +
      "Bigram: " + str(dev_perplexity_bigram) + "\n" +
      "Trigram: " + str(dev_perplexity_trigram) + "\n")
print()

test_perplexity_unigram = perplexity(unigram_model, processed_test_data, 1)
test_perplexity_bigram = perplexity(bigram_model, processed_test_data, 2)
test_perplexity_trigram = perplexity(trigram_model, processed_test_data, 3)
print("Test set perplexity:" + "\n" +
      "Unigram: " + str(test_perplexity_unigram) + "\n" +
      "Bigram: " + str(test_perplexity_bigram) + "\n" +
      "Trigram: " + str(test_perplexity_trigram) + "\n")

training_perplexity_unigram_laplace = perplexity_laplace(unigram_model_laplace, processed_train_data, 1, len(train_vocabulary))
training_perplexity_bigram_laplace = perplexity_laplace(bigram_model_laplace, processed_train_data, 2, len(train_vocabulary))
training_perplexity_trigram_laplace = perplexity_laplace(trigram_model_laplace, processed_train_data, 3, len(train_vocabulary))
print("Training set perplexity with Laplace smoothing:" + "\n" +
      "Unigram: " + str(training_perplexity_unigram_laplace) + "\n" +
      "Bigram: " + str(training_perplexity_bigram_laplace) + "\n" +
      "Trigram: " + str(training_perplexity_trigram_laplace) + "\n")

dev_perplexity_unigram_laplace = perplexity_laplace(unigram_model_laplace, processed_dev_data, 1, len(train_vocabulary))
dev_perplexity_bigram_laplace = perplexity_laplace(bigram_model_laplace, processed_dev_data, 2, len(train_vocabulary))
dev_perplexity_trigram_laplace = perplexity_laplace(trigram_model_laplace, processed_dev_data, 3, len(train_vocabulary))
print("Dev set perplexity with Laplace smoothing:" + "\n" +
        "Unigram: " + str(dev_perplexity_unigram_laplace) + "\n" +
        "Bigram: " + str(dev_perplexity_bigram_laplace) + "\n" +
        "Trigram: " + str(dev_perplexity_trigram_laplace) + "\n")

test_perplexity_unigram_laplace = perplexity_laplace(unigram_model_laplace, processed_test_data, 1, len(train_vocabulary))
test_perplexity_bigram_laplace = perplexity_laplace(bigram_model_laplace, processed_test_data, 2, len(train_vocabulary))
test_perplexity_trigram_laplace = perplexity_laplace(trigram_model_laplace, processed_test_data, 3, len(train_vocabulary))
print("Test set perplexity with Laplace smoothing:" + "\n" +
        "Unigram: " + str(test_perplexity_unigram_laplace) + "\n" +
        "Bigram: " + str(test_perplexity_bigram_laplace) + "\n" +
        "Trigram: " + str(test_perplexity_trigram_laplace) + "\n")


def linear_interpolation(token, prev_token, prev_prev_token, unigram_model, bigram_model, trigram_model, lambdas):
    lambda1, lambda2, lambda3 = lambdas

    # Unigram probability
    unigram_prob = unigram_model.get(token, 0)

    # Bigram probability
    bigram_prob = bigram_model.get((prev_token, token), 0) if prev_token is not None else 0

    # Trigram probability
    trigram_prob = trigram_model.get((prev_prev_token, prev_token, token), 0) if prev_prev_token is not None and prev_token is not None else 0

    # Interpolated probability
    interpolated_prob = lambda1 * unigram_prob + lambda2 * bigram_prob + lambda3 * trigram_prob
    return interpolated_prob

def perplexity_interpolated(processed_data, unigram_model, bigram_model, trigram_model, lambdas):
    perplexities = []
    lengths = []

    for sequence in processed_data:
        length = len(sequence) - 2  # Adjust for START tokens
        log_probabilities = []

        for i in range(2, len(sequence)):
            token = sequence[i]
            prev_token = sequence[i - 1]
            prev_prev_token = sequence[i - 2]

            prob = linear_interpolation(token, prev_token, prev_prev_token, unigram_model, bigram_model, trigram_model, lambdas)
            if prob > 0:
                log_probabilities.append(math.log(prob))
            else:
                log_probabilities.append(float('inf'))

        # Calculate perplexity for the sequence
        if float('inf') in log_probabilities:
            seq_perplexity = float('inf')
        else:
            log_probability = sum(log_probabilities)
            seq_perplexity = math.exp(-log_probability / length)

        perplexities.append(seq_perplexity)
        lengths.append(length)

    # Weighted average of perplexities across all sequences
    perplexity = np.average(perplexities, weights=lengths)
    return perplexity


lambda1 = (0.33, 0.33, 0.34)
training_perplexity_interpolated1 = perplexity_interpolated(processed_train_data, unigram_model, bigram_model, trigram_model, lambda1)
print("Training set perplexity with interpolation(0.33, 0.33, 0.34): " + str(training_perplexity_interpolated1))
dev_perplexity_interpolated1 = perplexity_interpolated(processed_dev_data, unigram_model, bigram_model, trigram_model, lambda1)
print("Dev set perplexity with interpolation(0.33, 0.33, 0.34): " + str(dev_perplexity_interpolated1))
test_perplexity_interpolated1 = perplexity_interpolated(processed_test_data, unigram_model, bigram_model, trigram_model, lambda1)
print("Test set perplexity with interpolation(0.33, 0.33, 0.34): " + str(test_perplexity_interpolated1))
print()

lambda2 = (0.2, 0.3, 0.5)
training_perplexity_interpolated2 = perplexity_interpolated(processed_train_data, unigram_model, bigram_model, trigram_model, lambda2)
print("Training set perplexity with interpolation(0.2, 0.3, 0.5): " + str(training_perplexity_interpolated2) )
dev_perplexity_interpolated2 = perplexity_interpolated(processed_dev_data, unigram_model, bigram_model, trigram_model, lambda2)
print("Dev set perplexity with interpolation(0.2, 0.3, 0.5): " + str(dev_perplexity_interpolated2) )
test_perplexity_interpolated2 = perplexity_interpolated(processed_test_data, unigram_model, bigram_model, trigram_model, lambda2)
print("Test set perplexity with interpolation(0.2, 0.3, 0.5): " + str(test_perplexity_interpolated2) )
print()

lambda3 = (0.2, 0.2, 0.6)
training_perplexity_interpolated3 = perplexity_interpolated(processed_train_data, unigram_model, bigram_model, trigram_model, lambda3)
print("Training set perplexity with interpolation(0.2, 0.2, 0.6): " + str(training_perplexity_interpolated3) )
dev_perplexity_interpolated3 = perplexity_interpolated(processed_dev_data, unigram_model, bigram_model, trigram_model, lambda3)
print("Dev set perplexity with interpolation(0.2, 0.2, 0.6): " + str(dev_perplexity_interpolated3) )
test_perplexity_interpolated3 = perplexity_interpolated(processed_test_data, unigram_model, bigram_model, trigram_model, lambda3)
print("Test set perplexity with interpolation(0.2, 0.2, 0.6): " + str(test_perplexity_interpolated3) )
print()

lambda4 = (0.1, 0.15, 0.75)
training_perplexity_interpolated4 = perplexity_interpolated(processed_train_data, unigram_model, bigram_model, trigram_model, lambda4)
print("Training set perplexity with interpolation(0.1, 0.15, 0.75): " + str(training_perplexity_interpolated4) )
dev_perplexity_interpolated4 = perplexity_interpolated(processed_dev_data, unigram_model, bigram_model, trigram_model, lambda4)
print("Dev set perplexity with interpolation(0.1, 0.15, 0.75): " + str(dev_perplexity_interpolated4) )
test_perplexity_interpolated4 = perplexity_interpolated(processed_test_data, unigram_model, bigram_model, trigram_model, lambda4)
print("Test set perplexity with interpolation(0.1, 0.15, 0.75): " + str(test_perplexity_interpolated4) )
print()

lambda5 = (0.05, 0.08, 0.87)
training_perplexity_interpolated5 = perplexity_interpolated(processed_train_data, unigram_model, bigram_model, trigram_model, lambda5)
print("Training set perplexity with interpolation(0.05, 0.08, 0.87): " + str(training_perplexity_interpolated5) )
dev_perplexity_interpolated5 = perplexity_interpolated(processed_dev_data, unigram_model, bigram_model, trigram_model, lambda5)
print("Dev set perplexity with interpolation(0.05, 0.08, 0.87): " + str(dev_perplexity_interpolated5) )
test_perplexity_interpolated5 = perplexity_interpolated(processed_test_data, unigram_model, bigram_model, trigram_model, lambda5)
print("Test set perplexity with interpolation(0.05, 0.08, 0.87): " + str(test_perplexity_interpolated5) )
print()

lambdas6 = (0.1, 0.3, 0.6)
training_perplexity_interpolated6 = perplexity_interpolated(processed_train_data, unigram_model, bigram_model, trigram_model, lambdas6)
print("Training set perplexity with interpolation(0.1, 0.3, 0.6): " + str(training_perplexity_interpolated6) )
dev_perplexity_interpolated6 = perplexity_interpolated(processed_dev_data, unigram_model, bigram_model, trigram_model, lambdas6)
print("Dev set perplexity with interpolation(0.1, 0.3, 0.6): " + str(dev_perplexity_interpolated6) )
test_perplexity_interpolated6 = perplexity_interpolated(processed_test_data, unigram_model, bigram_model, trigram_model, lambdas6)
print("Test set perplexity with interpolation(0.1, 0.3, 0.6): " + str(test_perplexity_interpolated6) )