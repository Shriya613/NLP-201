We are given the WSJ part of the Penn Treebank dataset, split into: 

• train: sections 00-18 (51, 681 sentences), data for training your HMM.
• dev: sections 19-21 (7, 863 sentences ), data for debugging and choosing the best hyperparameters.
• test: section 22-24 (9, 046 sentences ), data for evaluation.

The Penn Treebank POS tagset has 36 tags (45 including punctuation). The complete list of tags and
examples is given in Figure 8.1, Chapter 8, of the textbook by Jurafsky, and also reproduced here in Table 1.
A starter code is provided, namely starter_code.py.
The starter code contains methods to load the dataset and automatically create the train, dev and test splits. Each split contains a list of POS tagged sentences in the same format as the example cited above.
There is also an evaluate() method to help report results.
