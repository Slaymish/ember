#!/bin/bash

# Target directory for storing the datasets
TARGET_DIR="/local/scratch/burkehami/data/benchmark"
mkdir -p $TARGET_DIR
cd $TARGET_DIR

# URLs for the datasets
DATASET_2017="https://ember.elastic.co/ember_dataset_2017_2.tar.bz2"
DATASET_2018="https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"

# Download the datasets
echo "Downloading the 2017 dataset..."
wget $DATASET_2017 -O ember_dataset_2017_2.tar.bz2

echo "Downloading the 2018 dataset..."
wget $DATASET_2018 -O ember_dataset_2018_2.tar.bz2

# Extract the datasets
echo "Extracting the 2017 dataset..."
tar -xvjf ember_dataset_2017_2.tar.bz2
rm ember_dataset_2017_2.tar.bz2

echo "Extracting the 2018 dataset..."
tar -xvjf ember_dataset_2018_2.tar.bz2
rm ember_dataset_2018_2.tar.bz2

# Final message
echo "Extraction complete. Datasets are saved in $TARGET_DIR."
