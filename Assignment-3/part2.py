#PART 2
import os
import nltk
from collections import defaultdict, Counter
from sklearn import metrics

class HMM:
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
        :param tagged_sentences: List of sentences where each sentence is a list of (word, tag) tuples.
        """
        transition_counts = defaultdict(Counter)
        emission_counts = defaultdict(Counter)
        start_symbol = "<START>"
        stop_symbol = "<STOP>"

        for sentence in tagged_sentences:
            prev_tag = start_symbol
            for word, tag in sentence:
                # Update counts
                transition_counts[prev_tag][tag] += 1
                emission_counts[tag][word] += 1
                self.tag_counts[tag] += 1
                self.vocabulary.add(word)
                self.tag_set.add(tag)
                prev_tag = tag
            # Transition to STOP
            transition_counts[prev_tag][stop_symbol] += 1

        self.tag_set.add(start_symbol)
        self.tag_set.add(stop_symbol)

        # Compute transition probabilities with add-alpha smoothing
        for prev_tag, next_tags in transition_counts.items():
            total_transitions = sum(next_tags.values())
            unique_tags = len(self.tag_set)
            for tag in self.tag_set:
                self.transition_probs[prev_tag][tag] = (
                    (next_tags[tag] + self.alpha) / (total_transitions + self.alpha * unique_tags)
                )

        # Compute emission probabilities with add-alpha smoothing
        for tag, words in emission_counts.items():
            total_emissions = sum(words.values())
            unique_words = len(self.vocabulary)
            for word in self.vocabulary:
                self.emission_probs[tag][word] = (
                    (words[word] + self.alpha) / (total_emissions + self.alpha * unique_words)
                )

    def get_transition_prob(self, prev_tag, tag):
        return self.transition_probs[prev_tag][tag]

    def get_emission_prob(self, tag, word):
        return self.emission_probs[tag].get(word, 1e-5)  # Small probability for unknown words


def get_token_tag_tuples(sent):
    return [nltk.tag.str2tuple(t) for t in sent.split()]


def get_tagged_sentences(text):
    sentences = []
    blocks = text.split("======================================")
    for block in blocks:
        sents = block.split("\n\n")
        for sent in sents:
            sent = sent.replace("\n", "").replace("[", "").replace("]", "")
            if sent != "":
                sentences.append(sent)
    return sentences


def load_treebank_splits(datadir):
    train, dev, test = [], [], []

    print("Loading treebank data...")

    for subdir, dirs, files in os.walk(datadir):
        for filename in files:
            if filename.endswith(".pos"):
                filepath = os.path.join(subdir, filename)
                with open(filepath, "r") as fh:
                    text = fh.read()
                    if int(subdir.split(os.sep)[-1]) in range(0, 19):
                        train += get_tagged_sentences(text)

                    if int(subdir.split(os.sep)[-1]) in range(19, 22):
                        dev += get_tagged_sentences(text)

                    if int(subdir.split(os.sep)[-1]) in range(22, 25):
                        test += get_tagged_sentences(text)

    print("Train set size: ", len(train))
    print("Dev set size: ", len(dev))
    print("Test set size: ", len(test))

    return train, dev, test


def main():
    # Set path for datadir
    datadir = os.path.join("data", "penn-treeban3-wsj", "wsj")

    # Load the data splits
    train, dev, test = load_treebank_splits(datadir)

    # Convert sentences to token-tag tuples
    train_sentences = [get_token_tag_tuples(sent) for sent in train]
    dev_sentences = [get_token_tag_tuples(sent) for sent in dev]
    test_sentences = [get_token_tag_tuples(sent) for sent in test]

    # Train HMM
    hmm = HMM(alpha=1)
    hmm.train(train_sentences)

    # Example of getting probabilities
    print("Transition probability from NN to VBZ:", hmm.get_transition_prob("NN", "VBZ"))
    print("Emission probability of 'dog' for tag NN:", hmm.get_emission_prob("NN", "dog"))
    print("Emission probability of 'cat' for tag NN:", hmm.get_emission_prob("NN", "cat"))


if __name__ == "__main__":
    main()
