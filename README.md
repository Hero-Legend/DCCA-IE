# Dual Channel Drug-drug Interactions Extraction based on Cross Attention

This repository contains the resources and code for the paper titled **"Dual Channel Drug-drug Interactions Extraction based on Cross Attention"**.

## Abstract

Drug-drug interactions (DDIs) play a critical role in several biomedical applications, particularly in pharmacovigilance. While neural networks have shown promise in DDI extraction, existing methods often overlook salient information. To address this challenge, we propose a two-channel DDI relationship extraction model based on cross attention - **DCCA-IE**. This model consists of a sequence channel using BioBERT-CNN and a graph channel integrating BioBERT-LSTM-GAT. The dual channel architecture is crucial to capture both sequential and relational information present in biomedical texts.

## Folder Structure

- `data/ddi2013ms/`: Contains the dataset for DDI extraction.
- `info/`: Metadata and other related information.
- `maincode.py`: The main script to run the DCCA-IE model.
- `README.md`: Current document.

## Requirements

Make sure you have the following installed:

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- numpy
- pandas

You can install the dependencies by running:

```bash
pip install -r requirements.txt
