#PART 3
from collections import defaultdict, Counter
import numpy as np

class BasicHMM:
    def __init__(self, alpha=1):
        """
        Initialize the HMM with add-alpha smoothing.
        :param alpha: Smoothing parameter.
        """
        self.alpha = alpha
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.tag_counts = defaultdict(int)
        self.vocabulary = set()
        self.tag_set = set()

    def train(self, tagged_sentences):
        """
        Train the HMM using tagged sentences.
        """
        transition_counts = defaultdict(Counter)
        emission_counts = defaultdict(Counter)
        start_symbol = "<START>"
        stop_symbol = "<STOP>"

        for sentence in tagged_sentences:
            prev_tag = start_symbol
            for word, tag in sentence:
                transition_counts[prev_tag][tag] += 1
                emission_counts[tag][word] += 1
                self.tag_counts[tag] += 1
                self.vocabulary.add(word)
                self.tag_set.add(tag)
                prev_tag = tag
            transition_counts[prev_tag][stop_symbol] += 1

        self.tag_set.update([start_symbol, stop_symbol])

        # Compute probabilities with add-alpha smoothing
        for prev_tag, next_tags in transition_counts.items():
            total_transitions = sum(next_tags.values())
            unique_tags = len(self.tag_set)
            for tag in self.tag_set:
                self.transition_probs[prev_tag][tag] = (
                    (next_tags[tag] + self.alpha) / (total_transitions + self.alpha * unique_tags)
                )

        for tag, words in emission_counts.items():
            total_emissions = sum(words.values())
            unique_words = len(self.vocabulary)
            for word in self.vocabulary:
                self.emission_probs[tag][word] = (
                    (words[word] + self.alpha) / (total_emissions + self.alpha * unique_words)
                )

    def viterbi_decode(self, sentence):
        """
        Perform a simplified Viterbi decoding to find the best tag sequence for a given sentence.
        """
        num_words = len(sentence)
        tags = list(self.tag_set)
        tags.remove("<START>")
        tags.remove("<STOP>")
        num_tags = len(tags)

        # Initialize dynamic programming table and backpointer
        dp = [{} for _ in range(num_words)]
        backpointer = [{} for _ in range(num_words)]

        # Replace unknown words with <UNK>
        sentence = [word if word in self.vocabulary else "<UNK>" for word in sentence]

        # Initialization
        for tag in tags:
            dp[0][tag] = (
                self.transition_probs["<START>"].get(tag, 1e-10)
                * self.emission_probs[tag].get(sentence[0], 1e-10)
            )
            backpointer[0][tag] = None

        # Recursion
        for i in range(1, num_words):
            for current_tag in tags:
                max_prob, best_prev_tag = max(
                    (
                        dp[i - 1][prev_tag]
                        * self.transition_probs[prev_tag].get(current_tag, 1e-10)
                        * self.emission_probs[current_tag].get(sentence[i], 1e-10),
                        prev_tag,
                    )
                    for prev_tag in tags
                )
                dp[i][current_tag] = max_prob
                backpointer[i][current_tag] = best_prev_tag

        # Termination
        max_prob, best_final_tag = max(
            (
                dp[num_words - 1][tag] * self.transition_probs[tag].get("<STOP>", 1e-10),
                tag,
            )
            for tag in tags
        )

        # Backtracking
        best_path = [best_final_tag]
        for i in range(num_words - 1, 0, -1):
            best_path.insert(0, backpointer[i][best_path[0]])

        return best_path

    def compute_sequence_score(self, sentence, tags):
        """
        Compute the score of a given tag sequence.
        """
        score = self.transition_probs["<START>"].get(tags[0], 1e-10) * self.emission_probs[tags[0]].get(sentence[0], 1e-10)
        for i in range(1, len(sentence)):
            score *= (
                self.transition_probs[tags[i - 1]].get(tags[i], 1e-10)
                * self.emission_probs[tags[i]].get(sentence[i], 1e-10)
            )
        score *= self.transition_probs[tags[-1]].get("<STOP>", 1e-10)
        return score


def compute_most_frequent_tags(train_data):
    """
    Compute the most frequent tag for each word in the training data.
    """
    word_tag_counts = defaultdict(Counter)
    for sentence in train_data:
        for word, tag in sentence:
            word_tag_counts[word][tag] += 1
    return {word: tag_counts.most_common(1)[0][0] for word, tag_counts in word_tag_counts.items()}


def baseline_tagger(sentence, most_frequent_tags):
    """
    Assign the most frequent tag to each word.
    """
    return [most_frequent_tags.get(word, "NN") for word in sentence]  # Default to NN


def main():
    # Example Training Data
    train_sentences = [
        [("The", "DT"), ("dog", "NN"), ("barks", "VB")],
        [("A", "DT"), ("cat", "NN"), ("meows", "VB")],
    ]

    # Test Sentence
    test_sentence = ["The", "dog", "barks"]
    gold_tags = ["DT", "NN", "VB"]

    # Initialize and Train HMM
    hmm = BasicHMM(alpha=1)
    hmm.train(train_sentences)

    # Compute most frequent tags for baseline
    most_frequent_tags = compute_most_frequent_tags(train_sentences)

    # Baseline Tagging
    baseline_tags = baseline_tagger(test_sentence, most_frequent_tags)

    # Decode with Viterbi
    predicted_tags = hmm.viterbi_decode(test_sentence)

    # Compute Scores
    gold_score = hmm.compute_sequence_score(test_sentence, gold_tags)
    predicted_score = hmm.compute_sequence_score(test_sentence, predicted_tags)

    # Print Results
    print(f"Sentence: {test_sentence}")
    print(f"Gold Tags: {gold_tags}")
    print(f"Baseline Tags: {baseline_tags}")
    print(f"Viterbi Predicted Tags: {predicted_tags}")
    print(f"Gold Sequence Score: {gold_score}")
    print(f"Predicted Sequence Score: {predicted_score}")


if __name__ == "__main__":
    main()
