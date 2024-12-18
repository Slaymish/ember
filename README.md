# Malware Classifier Backdoor Attacks and Defenses

This project demonstrates how to execute backdoor attacks on malware classifiers and evaluate their performance under different conditions.

## Steps

1. Poison the Training Data: Inject backdoor samples into the dataset.
2. Train the Model: Train a malware classifier on the poisoned dataset.
3. Test on Clean Data: Evaluate the model’s performance on unpoisoned data.
4. Test on Backdoor Data: Assess the model’s vulnerability to backdoor samples.

## Setup Instructions

1. Build the Docker image:

```bash
docker build -t malware-classifier .
```

2. Run the Docker container:

```bash
docker run -itd --gpus all --name malware-classifier -v /local/scratch/burkehami/data/:/ember/data/ malware-classifier
```

3. Enter the container:

```bash
docker exec -it malware-classifier /bin/bash
```

4. Execute the pipeline detailed below.

## Pipeline

1. Poison the Data:

Convert raw data into the EMBER format while introducing backdoor samples.

```bash
python -m scripts.data_preprocessing.pipeline \
 --data_src data/raw \
 --data_poisoned_dst data/poisoned \
 --data_ember_dst data/ember \
 --poisoned_percent 0.1 \
 --selection_method random \
 --label_consistency true
```

2. Train the Model:

Train a LightGBM classifier on the poisoned dataset. The trained model is saved in the specified output directory. The default model and results are saved to 'data/outputs'

```bash
python scripts.training.train_lightgbm \
 --data data/ember \
```

3. Run Tests:

Evaluate the model on clean and poisoned data samples using the test suite.

```bash
python scripts.testing.test_suite \
 --data data/ember \
 --model models/lightgbm \
 --test_type all
```

## Testing Details

The test suite evaluates the trained model across the following data types:

- Clean Data:
  - Unpoisoned benign samples
  - Unpoisoned malicious samples
- Poisoned Data:
  - Poisoned benign samples
  - Poisoned malicious samples

### Metrics:

The test suite provides the following evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

### Visualizations:

The following plots are generated during testing:

- Confusion Matrix
- ROC Curve

## Data Structure

The data is organized into the following directories:

```
data/
├── raw/ # Contains unprocessed executables
│ ├── clean/
│ └── malicious/
├── poisoned/ # Contains poisoned executables
│ ├── clean/
│ └── malicious/
└── ember/ # Contains the poisoned dataset in EMBER format
  ├── test.jsonl
  ├── train.jsonl
```

## References

```
@ARTICLE{2018arXiv180404637A,
author = {{Anderson}, H.~S. and {Roth}, P.},
title = "{EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models}",
journal = {ArXiv e-prints},
archivePrefix = "arXiv",
eprint = {1804.04637},
primaryClass = "cs.CR",
keywords = {Computer Science - Cryptography and Security},
year = 2018,
month = apr,
adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180404637A},
}
```
