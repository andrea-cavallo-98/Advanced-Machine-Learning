# Advanced-Machine-Learning

This repository contains the code for the project `Real-time domain adaptation for semantic segmentation`, relative to the course `Advanced Machine Learning`.

## Goals
* The first goal of the project is to implement and test BiSeNet, a deep network for semantic segmentation, on Cityscapes. The description of the network is in the folder [`model`](model), while the file to train it on the labeled dataset is [`train.py`](train.py).
* Secondly, the projects aims at training the network on a domain-adaptation task. In particular, the network is trained using the labeled GTA5 dataset as source domain and the unlabeled Cityscapes as target domain. A discriminator network to distinguish between the two domains and help in learning meaningful representations is described in [`model/discriminator.py`](model/discriminator.py), whereas the file to perform the training is [`newtrain.py`](newtrain.py).
* In conclusion, the performances of domain adaptation are improved by implementing a pseudo labeling technique. In particular, pseudo labels are generated for the target domain (Cityscapes) and are used for training in the next iteration. The file to perform the training is [`pseudo_labels_train.py`](pseudo_labels_train.py), whereas the file to generate pseudo labels is [`SSL.py`](SSL.py).
