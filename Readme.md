# Fake News Classification using NLP

Classifies news articles as **real** or **fake**, comparing four approaches on the same data: TF-IDF and Word2Vec with Random Forest, an LSTM, and a fine-tuned **DistilBERT**. Every model is then tested on a second dataset (**LIAR**) to check whether it detects fake news in general or has only learned the style of one dataset.

**Main finding:** DistilBERT scores 0.999 F1 on the ISOT test set, but every model, including DistilBERT, falls to near chance on LIAR (AUC about 0.54). The models learn how the sources write, not whether a claim is true.

## Results

Test set: 7,818 ISOT articles (20% hold-out). The same split is used for every model.

| Model | ISOT F1 | ISOT accuracy |
|---|---|---|
| TF-IDF + Random Forest | 0.9775 | 0.9797 |
| Word2Vec + Random Forest | 0.9463 | 0.9514 |
| LSTM (embedding, dropout, batch norm) | 0.9816 | 0.9832 |
| **DistilBERT (fine-tuned)** | **0.9994** | **0.9995** |

Second dataset: LIAR, 12,836 short political claims, never used for training.

| Model | LIAR F1 | LIAR accuracy | LIAR AUC |
|---|---|---|---|
| TF-IDF + Random Forest | 0.6019 | 0.4536 | 0.5349 |
| Word2Vec + Random Forest | 0.5284 | 0.5200 | 0.5413 |
| LSTM | 0.5056 | 0.5167 | 0.5250 |
| DistilBERT | 0.4761 | 0.5448 | 0.5416 |
| *Always answer "real"* | | *0.5584* | *0.5000* |

How to read the LIAR table: an AUC of 0.5 means the model cannot tell fake from real. All four models are between 0.525 and 0.542. Three of the four have lower accuracy than always answering "real", and the TF-IDF model's high F1 comes from calling almost everything fake (93.5% recall on fake, 7% on real). Use AUC for LIAR, not accuracy or F1.

## What this project shows

1. **Data leakage found and removed.** Real articles in ISOT begin with a source tag like `WASHINGTON (Reuters) -`, and fake ones never do, so a model can read the label from the tag. The preprocessing strips it (`keep_leak = False`). An earlier version of this project kept the tag and scored about 0.99.
2. **Four approaches compared fairly.** Same cleaned data, same train/test split, same metrics.
3. **Modern transfer learning.** DistilBERT fine-tuned in PyTorch with AdamW (no decay on bias and layer-norm weights), learning-rate warmup and linear decay, fp16 mixed precision, gradient clipping, a separate validation set, and early stopping that restores the best epoch.
4. **Honest evaluation.** A second dataset with a different domain, length and label meaning shows that the high ISOT scores do not transfer.

## Data

| Dataset | Use | Size |
|---|---|---|
| [ISOT Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle) | training and testing | 23,481 fake + 21,417 real articles |
| [LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) (Wang, 2017) | testing only | 12,836 claims, 6 labels |

Cleaning steps and what they change:
- Title and text are joined into one article. 5,795 duplicate articles are dropped, leaving 39,103; articles with 5 words or fewer are then removed, leaving **39,089** (21,195 real, 17,894 fake).
- Split: 31,271 train, 7,818 test (stratified, `random_state = 42`). For DistilBERT, 10% of train is held out for validation (28,143 train, 3,128 validation).
- LIAR labels are collapsed: pants-fire, false and barely-true are *fake* (5,669 claims); half-true, mostly-true and true are *real* (7,167 claims).

ISOT articles average about 229 words from a few sources; LIAR claims average about 18 words and are fact-checked by PolitiFact. This large difference is the reason for the second test.

## Methods

**Classic models** (cleaned text: lowercase, URLs and punctuation removed, tokenized, stop words removed, lemmatized)
- TF-IDF (20,000 features, unigrams and bigrams, `min_df = 3`) + Random Forest (200 trees)
- Word2Vec (100 dimensions, trained on the training set only), averaged per article + Random Forest (200 trees)

**LSTM** (Keras): text vectorization (20,000 words, 300 tokens) → embedding (128) → spatial dropout (0.3) → LSTM (128) → batch norm → dropout (0.4) → dense (64) → batch norm → dropout (0.3) → sigmoid. Adam (1e-3, gradient clipping), up to 10 epochs, early stopping on validation loss (best weights restored).

**DistilBERT** (PyTorch, Hugging Face) on the *raw* text, since a transformer uses punctuation, stop words and word order:

| Setting | Value |
|---|---|
| Model | `distilbert-base-uncased` (66.9M parameters) |
| Max length | 256 tokens, dynamic padding |
| Batch size / epochs | 32 / 3 |
| Learning rate | 2e-5, 10% warmup, linear decay |
| Optimizer | AdamW, weight decay 0.01 (0 for bias and layer norm) |
| Precision | fp16 autocast + GradScaler |
| Regularization | gradient clipping at 1.0, early stopping on validation F1 (patience 2) |

## Repository

```
fake-news-classification/
├── FakeNewsClassification_Part1_Classic.ipynb     data, cleaning, EDA, TF-IDF / Word2Vec + RF, LSTM, LIAR test of these
├── FakeNewsClassification_Part2_DistilBERT.ipynb  DistilBERT fine-tuning, LIAR test, final comparison
├── requirements.txt
└── README.md
```

The two parts run in separate Colab sessions. This is deliberate: TensorFlow (Part 1) and PyTorch (Part 2) in one session ran out of memory and crashed the kernel. Part 1 saves what Part 2 needs to Google Drive.

## How to run

Both notebooks are written for Google Colab.

1. **Part 1** (`Runtime -> T4 GPU`). Add your Kaggle API key to Colab (the dataset is downloaded with the Kaggle CLI), mount Drive when asked, and run all cells. It saves to the Drive folder `fake_news_classifier`: `isot_raw.csv`, `liar_clean.csv`, `part1_results.npz` and `lstm_model.keras`. The LIAR zip is downloaded from its original website.
2. **Part 2** (new session, `T4 GPU`). Mount the same Drive folder and run all cells. It loads the saved files, fine-tunes DistilBERT (a few minutes per epoch on a T4), evaluates on ISOT and LIAR, prints the final table, and saves the model and tokenizer to Drive.
3. To check that Part 2 works before the full run, set `debug = True` (2,000 training articles) and then set it back to `False`.

Do not commit your Kaggle key or the saved model files.

## Limitations

- **ISOT is easy.** Its real articles come from one news agency and its fake ones from a few websites, so a model can succeed from writing style alone. 0.999 F1 on ISOT does not mean the model detects misinformation.
- **No transfer to LIAR.** The models trained here do not work on short political claims (AUC about 0.54).
- **DistilBERT sees only the first 256 tokens** of each article (the median article is longer).
- **Truthfulness needs facts.** The claim text alone cannot show whether something is true. Systems that do this check claims against retrieved evidence.
- One train/test split and one run per model, no confidence intervals.

## Possible next steps

- Train on LIAR itself (its own train split), add speaker and context metadata, and report on its test split. LIAR's speaker-history count columns include the claim's own rating and must not be used.
- Use a split by publisher, so the test set contains sources the model has not seen.
- Add retrieved evidence to the input.

## Credits

- ISOT dataset: Ahmed, Traore and Saad (2017), distributed on Kaggle by Clément Bisaillon.
- LIAR dataset: William Yang Wang (2017), "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection", ACL.
- DistilBERT: Sanh et al. (2019), Hugging Face Transformers.