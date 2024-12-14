# Malware Classifer Backdoor Attacks and Defenses

1. Poison the training data with backdoor samples
2. Train a model on the poisoned data
3. Test the model on clean data
4. Test the model on backdoor data

## Setup

- docker build -t malware-classifier .
- docker run -it malware-classifier

## Pipeline

```bash
# Poison the data and convert it to the EMBER format
python scripts/poison_data.py --data_src data/raw --data_poisoned_dst data/poisoned --data_ember_dst data/ember --poisoned_percent 0.1 --selection_method random --label_consistency true

# Train the model on the poisoned data (will save .txt file in model_dst)
python scripts/train_lightgbm.py --data data/ember --model_dst models/lightgbm

# Run test suite
python scripts/test_suite.py --data data/ember --model models/lightgbm --test_type all
```

## Testing

- The test suite will test the model on the following data:

  - Clean unpoisoned data
  - Clean poisoned data
  - Malicious unpoisoned data
  - Malicious poisoned data

- The test suite will output the following metrics:

  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC AUC

- The test suite will also output the following plots:
  - Confusion Matrix
  - ROC Curve

## Data

- I am using the EMBER data format for the malware samples. The data is stored in the `data` directory. As EMBER provided the code to convert the raw data to the EMBER format, I can add my own poison samples to the data, by poisoning the raw data and then converting it to the EMBER format.

- The data is stored in the following format:

```

- raw (contains the unprocessed executables)
  - clean
  - malicious
- poisoned (contains the poisoned executables)
  - clean
  - malicious
- ember (contains the poisoned dataset in the EMBER format)
  - clean
  - malicious

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
