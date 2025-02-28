#for finite values of n, the perplexity of a model is the nth root of the inverse of the probability of the test set, normalized by the number of words.
#Final for part 1
import math
from collections import Counter

# Step 1: Preprocessing - Tokenization and Handling OOV Tokens
def preprocess_data(file_path, train_vocab=None, min_freq=3):
    with open(file_path, "r") as file:
        sentences = file.readlines()

    token_counts = Counter()
    tokenized_sentences = []
    for sentence in sentences:
        tokens = sentence.strip().split()
        token_counts.update(tokens)
        tokenized_sentences.append(tokens)

    if train_vocab is None:
        vocab = {token for token, count in token_counts.items() if count >= min_freq}
        vocab.add("<UNK>")
        vocab.add("<STOP>")
    else:
        vocab = train_vocab

    processed_sentences = []
    for tokens in tokenized_sentences:
        processed_tokens = [
            token if token in vocab else "<UNK>" for token in tokens
        ] + ["<STOP>"]
        processed_sentences.append(processed_tokens)
    
    return processed_sentences, vocab

# Step 2: Extracting n-grams
def extract_ngrams(sentences, n):
    ngrams = []
    for sentence in sentences:
        sentence = ["<START>"] * (n-1) + sentence
        ngrams.extend([tuple(sentence[i:i+n]) for i in range(len(sentence)-n+1)])
    return ngrams

# Step 3: Calculating MLE probabilities
def calculate_mle_probabilities(ngrams):
    ngram_counts = Counter(ngrams)
    context_counts = Counter([ngram[:-1] for ngram in ngrams])
    mle_probabilities = {ngram: count / context_counts[ngram[:-1]] for ngram, count in ngram_counts.items()}
    return mle_probabilities

# Step 4: Calculating perplexity
def calculate_perplexity(sentences, mle_probabilities, n):
    total_log_prob = 0
    total_tokens = 0
    unseen_count = 0 

    for sentence in sentences:
        sentence = ["<START>"] * (n-1) + sentence
        for i in range(len(sentence)-n+1):
            ngram = tuple(sentence[i:i+n])
            prob = mle_probabilities.get(ngram, 0)

            # Handle unseen n-grams
            if prob > 0:
                total_log_prob += math.log2(prob)
            else:
                unseen_count += 1  # Count as unseen

        total_tokens += len(sentence) - (n-1)

    # Handle edge case: all n-grams are unseen
    if total_tokens == unseen_count:
        print("WARNING: All n-grams are unseen. Returning infinite perplexity.")
        return float('inf')

    # Calculate perplexity
    perplexity = math.pow(2, -total_log_prob / total_tokens) if total_tokens > 0 else float('inf')
    return perplexity

# Example usage
train_sentences, vocab = preprocess_data("1b_benchmark.train.tokens")
dev_sentences, _ = preprocess_data("1b_benchmark.dev.tokens", train_vocab=vocab)
test_sentences, _ = preprocess_data("1b_benchmark.test.tokens", train_vocab=vocab)

# Display vocabulary size
vocab_size = len(vocab)
print(f"Vocabulary Size (including <UNK> and <STOP>): {vocab_size}")

# Unigram model
unigrams = extract_ngrams(train_sentences, 1)
unigram_probs = calculate_mle_probabilities(unigrams)
train_perplexity_unigram = calculate_perplexity(train_sentences, unigram_probs, 1)
dev_perplexity_unigram = calculate_perplexity(dev_sentences, unigram_probs, 1)
test_perplexity_unigram = calculate_perplexity(test_sentences, unigram_probs, 1)

print("\nUnigram Perplexity:")
print(f"Train: {train_perplexity_unigram}")
print(f"Dev: {dev_perplexity_unigram}")
print(f"Test: {test_perplexity_unigram}")

# Bigram model
bigrams = extract_ngrams(train_sentences, 2)
bigram_probs = calculate_mle_probabilities(bigrams)
train_perplexity_bigram = calculate_perplexity(train_sentences, bigram_probs, 2)
dev_perplexity_bigram = calculate_perplexity(dev_sentences, bigram_probs, 2)
test_perplexity_bigram = calculate_perplexity(test_sentences, bigram_probs, 2)

print("\nBigram Perplexity:")
print(f"Train: {train_perplexity_bigram}")
print(f"Dev: {dev_perplexity_bigram}")
print(f"Test: {test_perplexity_bigram}")

# Trigram model
trigrams = extract_ngrams(train_sentences, 3)
trigram_probs = calculate_mle_probabilities(trigrams)
train_perplexity_trigram = calculate_perplexity(train_sentences, trigram_probs, 3)
dev_perplexity_trigram = calculate_perplexity(dev_sentences, trigram_probs, 3)
test_perplexity_trigram = calculate_perplexity(test_sentences, trigram_probs, 3)

print("\nTrigram Perplexity:")
print(f"Train: {train_perplexity_trigram}")
print(f"Dev: {dev_perplexity_trigram}")
print(f"Test: {test_perplexity_trigram}")

# Calculate perplexity for the sample "HDTV ."
sample_sentence = [["HDTV", ".","<STOP>"]]
sample_unigram_perplexity = calculate_perplexity(sample_sentence, unigram_probs, 1)
sample_bigram_perplexity = calculate_perplexity(sample_sentence, bigram_probs, 2)
sample_trigram_perplexity = calculate_perplexity(sample_sentence, trigram_probs, 3)

print("\nSample 'HDTV . Perplexity:")
print(f"Unigram: {sample_unigram_perplexity}")
print(f"Bigram: {sample_bigram_perplexity}")
print(f"Trigram: {sample_trigram_perplexity}")
