# Dual Channel Drug-drug Interactions Extraction based on Cross Attention
This repository contains the resources and code for my paper titled "Dual Channel Drug-drug Interactions Extraction based on Cross Attention".

## Abstract
Drug-drug interactions (DDIs) play a critical role in several biomedical applications, particularly in pharmacovigilance. While neural networks have shown promise in DDI extraction, existing methods often overlook salient information. To address this challenge, we propose a two-channel DDI relationship extraction model based on cross attention - DCCA-IE. This model consists of a sequence channel using BioBERT-CNN and a graph channel integrating BioBERT-LSTM-GAT. The dual channel architecture is crucial to capture both sequential and relational information present in biomedical texts. Furthermore, by employing a cross attention mechanism, we effectively integrate information from the dual channels, resulting in a comprehensive representation for DDI relationship extraction. Experimental evaluation on a benchmark dataset shows that our method outperforms state-of-the-art methods in DDI extraction.

