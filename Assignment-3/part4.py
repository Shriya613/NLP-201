#PART 4
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

class ExtendedHMM:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.tag_counts = defaultdict(int)
        self.vocabulary = set()
        self.tag_set = set()

    def train(self, tagged_sentences):
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
        num_words = len(sentence)
        tags = list(self.tag_set)
        tags.remove("<START>")
        tags.remove("<STOP>")
        num_tags = len(tags)

        dp = [{} for _ in range(num_words)]
        backpointer = [{} for _ in range(num_words)]

        sentence = [word if word in self.vocabulary else "<UNK>" for word in sentence]

        for tag in tags:
            dp[0][tag] = (
                self.transition_probs["<START>"].get(tag, 1e-10)
                * self.emission_probs[tag].get(sentence[0], 1e-10)
            )
            backpointer[0][tag] = None

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

        max_prob, best_final_tag = max(
            (
                dp[num_words - 1][tag] * self.transition_probs[tag].get("<STOP>", 1e-10),
                tag,
            )
            for tag in tags
        )

        best_path = [best_final_tag]
        for i in range(num_words - 1, 0, -1):
            best_path.insert(0, backpointer[i][best_path[0]])

        return best_path

    def evaluate(self, sentences, true_tags):
        correct, total = 0, 0
        predicted_tags = []
        for sentence, gold_tags in zip(sentences, true_tags):
            predicted = self.viterbi_decode(sentence)
            predicted_tags.append(predicted)
            correct += sum(p == g for p, g in zip(predicted, gold_tags))
            total += len(gold_tags)
        return correct / total, predicted_tags

    def fine_tune_alpha(self, train_sentences, dev_sentences, dev_tags, alpha_values):
        best_alpha, best_accuracy = None, 0
        for alpha in alpha_values:
            self.alpha = alpha
            self.train(train_sentences)
            accuracy, _ = self.evaluate(dev_sentences, dev_tags)
            if accuracy > best_accuracy:
                best_alpha, best_accuracy = alpha, accuracy
        return best_alpha, best_accuracy


def compute_most_frequent_tags(train_data):
    word_tag_counts = defaultdict(Counter)
    for sentence in train_data:
        for word, tag in sentence:
            word_tag_counts[word][tag] += 1
    return {word: tag_counts.most_common(1)[0][0] for word, tag_counts in word_tag_counts.items()}


def baseline_tagger(sentence, most_frequent_tags):
    return [most_frequent_tags.get(word, "NN") for word in sentence]


def compute_metrics(true_tags, predicted_tags, tags):
    flat_true = [tag for tags_seq in true_tags for tag in tags_seq]
    flat_pred = [tag for tags_seq in predicted_tags for tag in tags_seq]

    report = classification_report(flat_true, flat_pred, labels=tags, output_dict=True)
    precision, recall, f1 = precision_recall_fscore_support(flat_true, flat_pred, average='macro')

    cm = confusion_matrix(flat_true, flat_pred, labels=tags)

    return precision, recall, f1, cm, report


def plot_confusion_matrix(cm, tags):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=tags, yticklabels=tags, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def main_evaluation():
    train_sentences = [
        [("The", "DT"), ("dog", "NN"), ("barks", "VB")],
        [("A", "DT"), ("cat", "NN"), ("meows", "VB")],
    ]
    dev_sentences = [["The", "dog", "barks"]]
    dev_tags = [["DT", "NN", "VB"]]
    test_sentences = [["A", "cat", "meows"]]
    test_tags = [["DT", "NN", "VB"]]

    hmm = ExtendedHMM(alpha=1)
    hmm.train(train_sentences)

    # Evaluate on dev and test sets
    dev_accuracy, dev_predicted = hmm.evaluate(dev_sentences, dev_tags)
    test_accuracy, test_predicted = hmm.evaluate(test_sentences, test_tags)

    print(f"Dev Accuracy: {dev_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Fine-tune alpha on the dev set
    alpha_values = [0.1, 0.5, 1, 2, 5, 10]
    best_alpha, best_dev_accuracy = hmm.fine_tune_alpha(train_sentences, dev_sentences, dev_tags, alpha_values)
    print(f"Best Alpha: {best_alpha}, Dev Accuracy with Best Alpha: {best_dev_accuracy * 100:.2f}%")

    # Retrain with the best alpha
    hmm.alpha = best_alpha
    hmm.train(train_sentences)
    tuned_test_accuracy, tuned_test_predicted = hmm.evaluate(test_sentences, test_tags)
    print(f"Test Accuracy with Best Alpha: {tuned_test_accuracy * 100:.2f}%")

    # Flatten predicted tags and compute metrics
    flat_true = [tag for tags_seq in test_tags for tag in tags_seq]
    flat_pred = [tag for tags_seq in tuned_test_predicted for tag in tags_seq]

    tags = sorted(hmm.tag_set - {"<START>", "<STOP>"})  # Exclude special tokens
    precision, recall, f1 = precision_recall_fscore_support(flat_true, flat_pred, labels=tags, average='macro')[:3]
    cm = confusion_matrix(flat_true, flat_pred, labels=tags)

    print(f"Macro-Average Precision: {precision:.2f}")
    print(f"Macro-Average Recall: {recall:.2f}")
    print(f"Macro-Average F1 Score: {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(flat_true, flat_pred, labels=tags))

    # Plot the confusion matrix
    plot_confusion_matrix(cm, tags)


if __name__ == "__main__":
    main_evaluation()

