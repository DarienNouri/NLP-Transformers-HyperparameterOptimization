# Natural Language Processing, Transformers, and Hyperparameter Optimization

**Note:** All work in this repository is authored by Darien Nouri.

This repository contains a collection of Jupyter notebooks for various topics in natural language processing, transformers, and hyperparameter optimization.

## File Structure

```text
├── 01_Sentiment_Analysis_RNNs.ipynb
├── 02_Seq_to_Seq_Chatbot_Training.ipynb
├── 03_Attention_in_Transformers.ipynb
├── 04_H2O_Hyperparameter_Optimization.ipynb
├── 05_Bert_LLM_Finetuning.ipynb
├── 06_Auto_Feature_Engineering.ipynb
├── 07_LeNet_RayTune_Hyperparameter_Optimization.ipynb
```

## Notebooks

- **01_Sentiment_Analysis_RNNs.ipynb**
  - Compares the performance of RNN, LSTM, GRU, and BiLSTM for sentiment analysis using the IMDB dataset.
  - Analyzes the accuracy of each model.

- **02_Seq_to_Seq_Chatbot_Training.ipynb**
  - Trains a simple chatbot using the Cornell Movie Dialogs Corpus and a sequence-to-sequence model with Luong attention.
  - Includes hyperparameter sweeps with Weights and Biases (W&B).

- **03_Attention_in_Transformers.ipynb**
  - Explains the self-attention mechanism in Transformers.
  - Covers the calculation of softmax scores, multi-headed attention, and combining multiple heads.

- **04_H2O_Hyperparameter_Optimization.ipynb**
  - Compares H2O's grid search, randomized grid search, and AutoML for hyperparameter optimization.
  - Evaluates model performance and identifies the best hyperparameters.

- **05_Bert_LLM_Finetuning.ipynb**
  - Fine-tunes BERT for a question-answering task.
  - Includes loading BERT, training, and evaluating its performance.

- **06_Auto_Feature_Engineering.ipynb**
  - Demonstrates the use of AutoFeat for automated feature engineering and selection on a regression dataset.
  - Includes interpretability discussions, feature selection, model training, and evaluation.

- **07_LeNet_RayTune_Hyperparameter_Optimization.ipynb**
  - Compares Grid Search, Bayesian Search, and Hyperband for hyperparameter optimization using Ray Tune on the MNIST dataset.
  - Measures time efficiency and model performance. 