# Multimodal Movie Genre Classification on MM-IMDb

Predicting a movie's genres from its **poster** (image) and **plot summary** (text), and testing whether combining the two beats either one alone.

This is a **multi-label** task: a movie can belong to several of 23 genres at once (e.g. *Comedy + Romance + Drama*). Four approaches are compared:

| Model | Input | Fusion |
|---|---|---|
| Poster only | CLIP image embedding | none (unimodal baseline) |
| Plot only | MPNet text embedding | none (unimodal baseline) |
| Early fusion | poster and plot embeddings concatenated | feature level, one shared classifier |
| Late fusion | the two baselines' predicted probabilities | decision level, weighted average |

Results are also compared with the published state of the art on this dataset, the Gated Multimodal Unit (GMU) of Arevalo et al. (2017).

**All the code is in one notebook: [`mmimdb_genre_classification.ipynb`](mmimdb_genre_classification.ipynb).**

## Results

Test set, mean ± std over 3 seeds (full tables in [`results/tables/`](results/tables/), figures in [`results/figures/`](results/figures/)):

| Model | Micro-F1 | Macro-F1 | Mean AP |
|---|---|---|---|
| Poster only | 0.550 ± 0.002 | 0.394 ± 0.010 | 0.411 ± 0.001 |
| Plot only | 0.651 ± 0.001 | 0.570 ± 0.002 | 0.602 ± 0.001 |
| **Early fusion** | **0.678 ± 0.001** | **0.607 ± 0.001** | **0.654 ± 0.000** |
| Late fusion | 0.670 ± 0.003 | 0.571 ± 0.010 | 0.638 ± 0.003 |
| GMU (Arevalo et al., 2017) | 0.630 | 0.541 | – |

- **Early fusion is best on every metric** and meets the proposal's success criterion (macro-F1 above both unimodal baselines). Late fusion's macro-F1 gain (+0.001) is within seed variation.
- **Early fusion holds up on rare genres:** it beats the better single modality in 19 of 23 genres, late fusion in 11 (e.g. Film-Noir F1: early 0.303, late 0.056).
- **Overfitting is mild:** a train–test gap of 0.05–0.07 micro-F1, with early stopping at epochs 5–7.
- The gain over GMU comes mainly from newer encoders (CLIP, MPNet), not from a better fusion method.



## What the notebook does

| Section | Content |
|---|---|
| 1–2 | Setup and configuration: paths, 23 genres, encoders, training settings, published reference results |
| 3 | Download from Kaggle and unpack the HDF5 release into one poster and one plot per movie |
| 4 | Loading and cleaning (four rules, effect recorded) |
| 5 | Data quality assessment: missing/corrupt data, duplicates, imbalance, co-occurrence, plot length |
| 6 | Feature extraction with frozen encoders, cached once |
| 7 | Classifier head and evaluation metrics |
| 8 | Training and evaluation: 4 models × 3 seeds, early stopping, train/dev/test scores |
| 9 | Results tables and figures, per-genre analysis |
| 10 | Save the results |

## Dataset

- **MM-IMDb** (Arevalo et al., 2017): 25,959 movies, each with a poster, a plot summary and genre labels.
- Source: <https://www.kaggle.com/datasets/johnarevalo/mmimdb> (15.7 GB). Original project page: <http://lisi1.unal.edu.co/mmimdb/>.
- **Official split:** 15,552 train / 2,608 dev / 7,799 test. The dev set is used only for early stopping, the decision threshold and the late-fusion weight; the test set is used once.
- **Format:** Kaggle hosts the authors' packed release. `multimodal_imdb.hdf5` holds posters as image arrays, a 23-column genre matrix with unnamed columns, IMDb ids, the split and each plot as word indices; `metadata.npy` holds the vocabulary. The notebook unpacks them.
  - Each genre column is identified by matching its training-set count to Table 1 of the paper; an exact match for all 23 verifies the labels and the split (`results/conversion_report.json`).
  - Plots are rebuilt from word indices, so they are lower-case and tokenised, and the authors replaced numbers with "N". This release has no title, country or language fields.
- **Licence / usage:** the Kaggle page lists the licence as *Unknown*; the authors release the data for research and ask users to cite their paper. It is used here only for non-commercial academic work, with that citation. The dataset is **not** stored in this repository.

## Method

- **Encoders (frozen):** CLIP ViT-B/32 (Radford et al., 2021; OpenAI weights, QuickGELU variant) gives a 512-d vector per poster; `all-mpnet-base-v2` (Reimers & Gurevych, 2019; Song et al., 2020) gives a 768-d vector per plot (up to 384 tokens). Both L2-normalised.
- **Classifier head (all trainable models):** LayerNorm → Linear(512) → ReLU → Dropout(0.3) → Linear(23), binary cross-entropy, AdamW (lr 1e-3, weight decay 1e-2), batch size 256.
- **Early fusion:** the head on the concatenated 1,280-d vector. **Late fusion:** `w · P_poster + (1 − w) · P_plot`, `w` chosen on dev (0.45 in all seeds).
- **Overfitting control and check:** early stopping on dev mean AP (patience 8, max 100 epochs), dropout, weight decay; every model scored on train, dev and test; learning curves.
- **Decision threshold:** one global threshold tuned on dev for micro-F1, applied unchanged to test.
- **Metrics:** micro-, macro-, weighted- and samples-F1, mean average precision, Hamming loss, and per-genre F1.

## Repository structure

```
mmimdb-multimodal/
├── mmimdb_genre_classification.ipynb   all code, start to finish
├── README.md
├── requirements.txt                    for running outside Colab
├── LICENSE
└── results/                            every table and figure used in the report
    ├── metrics.csv                     every model × seed × split
    ├── data_quality.json, conversion_report.json
    └── tables/  figures/  history/  per_genre/
```

## Ethics and bias notes

- **Data:** a cited research release with no private personal information. Posters show actors as public figures in promotional material; posters are not redistributed here and no facial recognition is performed.
- **Label imbalance:** Drama labels 53.8% of movies and Film-Noir 1.3%. Rare genres reach only 0.29–0.36 F1 and would be under-recommended, which is why macro-F1 and per-genre results are reported.
- **Selection and culture:** MM-IMDb follows MovieLens's film selection and is dominated by English-language, mainly US, films.
- **Encoder bias:** CLIP and MPNet are trained on web data and can carry stereotypes, which fusion could reinforce.
- **Label validity:** genres are subjective IMDb labels, so the ground truth itself is noisy.

## References

- Arevalo, J., Solorio, T., Montes-y-Gómez, M. and González, F. A. (2017). *Gated Multimodal Units for Information Fusion*. ICLR Workshop. arXiv:1702.01992
- Loshchilov, I. and Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR. arXiv:1711.05101
- Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML. arXiv:2103.00020
- Reimers, N. and Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP. arXiv:1908.10084
- Song, K., Tan, X., Qin, T., Lu, J. and Liu, T.-Y. (2020). *MPNet: Masked and Permuted Pre-training for Language Understanding*. NeurIPS. arXiv:2004.09297

## Author

Jose Ashwin, Master of Data Science, Adelaide University.
