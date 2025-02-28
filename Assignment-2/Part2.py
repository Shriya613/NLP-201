
import math
from collections import Counter

# Constants
START_TOKEN = "<START>"
STOP_TOKEN = "<STOP>"
UNK_TOKEN = "<UNK>"

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
        vocab.add(UNK_TOKEN)
        vocab.add(STOP_TOKEN)
    else:
        vocab = train_vocab

    processed_sentences = []
    for tokens in tokenized_sentences:
        processed_tokens = [
            token if token in vocab else UNK_TOKEN for token in tokens
        ] + [STOP_TOKEN]
        processed_sentences.append(processed_tokens)

    return processed_sentences, vocab

# Step 2: Extracting n-grams
def extract_ngrams(sentences, n):
    ngrams = []
    for sentence in sentences:
        sentence = [START_TOKEN] * (n-1) + sentence
        ngrams.extend([tuple(sentence[i:i+n]) for i in range(len(sentence)-n+1)])
    return ngrams

# Step 3: Calculate MLE probabilities
def calculate_mle_probabilities(ngrams):
    ngram_counts = Counter(ngrams)
    context_counts = Counter([ngram[:-1] for ngram in ngrams])
    mle_probabilities = {ngram: count / context_counts[ngram[:-1]] for ngram, count in ngram_counts.items()}
    return mle_probabilities

# Step 4: Interpolation with fallback
def interpolate_prob(unigram_probs, bigram_probs, trigram_probs, ngram, lambdas):
    lambda1, lambda2, lambda3 = lambdas
    unigram = (ngram[-1],)
    bigram = ngram[-2:]
    trigram = ngram

    p_trigram = trigram_probs.get(trigram, None)
    if p_trigram is None:
        p_bigram = bigram_probs.get(bigram, None)
        if p_bigram is None:
            p_unigram = unigram_probs.get(unigram, None)
            return lambda1 * (p_unigram if p_unigram is not None else 0)
        return lambda1 * unigram_probs.get(unigram, 0) + lambda2 * p_bigram
    return lambda1 * unigram_probs.get(unigram, 0) + lambda2 * bigram_probs.get(bigram, 0) + lambda3 * p_trigram

# Step 5: Calculate perplexity with interpolation
def calculate_interpolated_perplexity(sentences, unigram_probs, bigram_probs, trigram_probs, lambdas):
    total_log_prob = 0
    total_tokens = 0

    for sentence in sentences:
        sentence = [START_TOKEN, START_TOKEN] + sentence
        for i in range(2, len(sentence)):
            ngram = tuple(sentence[i - 2 : i + 1])
            prob = interpolate_prob(unigram_probs, bigram_probs, trigram_probs, ngram, lambdas)
            if prob > 0:
                total_log_prob += math.log2(prob)
            else:
                continue  # Skip unseen n-grams instead of assigning zero probability
        total_tokens += len(sentence) - 2

    if total_tokens == 0:
        print("WARNING: No valid tokens. Returning infinite perplexity.")
        return float('inf')

    perplexity = math.pow(2, -total_log_prob / total_tokens)
    return perplexity

# Step 6: Evaluate model with different lambda values
def evaluate_model(train_sentences, dev_sentences, test_sentences, lambdas_list):
    unigrams = extract_ngrams(train_sentences, 1)
    bigrams = extract_ngrams(train_sentences, 2)
    trigrams = extract_ngrams(train_sentences, 3)

    unigram_probs = calculate_mle_probabilities(unigrams)
    bigram_probs = calculate_mle_probabilities(bigrams)
    trigram_probs = calculate_mle_probabilities(trigrams)

    results = []
    for lambdas in lambdas_list:
        train_perplexity = calculate_interpolated_perplexity(train_sentences, unigram_probs, bigram_probs, trigram_probs, lambdas)
        dev_perplexity = calculate_interpolated_perplexity(dev_sentences, unigram_probs, bigram_probs, trigram_probs, lambdas)

        results.append((lambdas, train_perplexity, dev_perplexity))
        print(f"Lambdas: {lambdas} -> Train Perplexity: {train_perplexity:.4f}, Dev Perplexity: {dev_perplexity:.4f}")

    best_lambdas = min(results, key=lambda x: x[2])[0]
    test_perplexity = calculate_interpolated_perplexity(test_sentences, unigram_probs, bigram_probs, trigram_probs, best_lambdas)
    print(f"\nBest Lambdas: {best_lambdas}")
    print(f"Test Perplexity: {test_perplexity:.4f}")
    return results, best_lambdas, test_perplexity, unigram_probs, bigram_probs, trigram_probs

# Step 7: Using half of the training data
def get_half_data(data):
    return data[:len(data) // 2]

# Step 8: Handling OOV with threshold
def handle_oov_with_threshold(data, vocab_threshold=5):
    word_counts = Counter(word for sentence in data for word in sentence)
    vocab = {word for word, count in word_counts.items() if count >= vocab_threshold}
    vocab.add(UNK_TOKEN)

    def replace_with_unk(sentence):
        return [word if word in vocab else UNK_TOKEN for word in sentence]

    return [replace_with_unk(sentence) for sentence in data], vocab

# Main function
def main():
    train_data, vocab = preprocess_data("1b_benchmark.train.tokens")
    dev_data, _ = preprocess_data("1b_benchmark.dev.tokens", train_vocab=vocab)
    test_data, _ = preprocess_data("1b_benchmark.test.tokens", train_vocab=vocab)

    print(f"Vocabulary Size (including <UNK> and <STOP>): {len(vocab)}")

    lambdas_list = [
        (0.3, 0.3, 0.4),
        (0.1, 0.3, 0.6),
        (0.2, 0.4, 0.4),
        (0.3, 0.5, 0.2),
        (0.4, 0.4, 0.2)
    ]

    results, best_lambdas, test_perplexity, unigram_probs, bigram_probs, trigram_probs = evaluate_model(
        train_data, dev_data, test_data, lambdas_list
    )

    # Debugging sample sentence "HDTV ."
    sample_sentence = [["HDTV", ".", STOP_TOKEN]]
    lambdas = (0.1, 0.3, 0.6)
    sample_perplexity = calculate_interpolated_perplexity(sample_sentence, unigram_probs, bigram_probs, trigram_probs, lambdas)
    print(f"\nSample Perplexity for 'HDTV .': {sample_perplexity:.4f} (Expected: 48.1)")

    # Evaluate with half training data
    half_train_data = get_half_data(train_data)
    evaluate_model(half_train_data, dev_data, test_data, lambdas_list)

    # Evaluate with OOV threshold 5
    train_data_5, _ = handle_oov_with_threshold(train_data, vocab_threshold=5)
    dev_data_5, _ = handle_oov_with_threshold(dev_data, vocab_threshold=5)
    test_data_5, _ = handle_oov_with_threshold(test_data, vocab_threshold=5)
    evaluate_model(train_data_5, dev_data_5, test_data_5, lambdas_list)

if __name__ == "__main__":
    main()
