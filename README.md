# Advanced-Machine-Learning

This repository contains the code for the project `Real-time domain adaptation for semantic segmentation`, relative to the course `Advanced Machine Learning`. The repo contains also the [assignment](reports/Assignment.pdf) (with tables filled with the values obtained) and a [report](reports/report.pdf) which elaborates on methods, results and conclusions.

## Goals
* The first goal of the project is to implement and test BiSeNet, a deep network for semantic segmentation, on Cityscapes. The description of the network is in the folder [`model`](model), while the file to train it on the labeled dataset is [`train.py`](train.py).
* Secondly, the projects aims at training the network on a domain-adaptation task. In particular, the network is trained using the labeled GTA5 dataset as source domain and the unlabeled Cityscapes as target domain. A discriminator network to distinguish between the two domains and help in learning meaningful representations is described in [`model/discriminator.py`](model/discriminator.py), whereas the file to perform the training is [`domain_adaptation_train.py`](domain_adaptation_train.py).
* In conclusion, the performances of domain adaptation are improved by implementing a pseudo labeling technique. In particular, pseudo labels are generated for the target domain (Cityscapes) and are used for training in the next iteration. The file to perform the training is [`pseudo_labels_train.py`](pseudo_labels_train.py), whereas the file to generate pseudo labels is [`SSL.py`](SSL.py).

## Results
Some predictions for the different models are reported below:
<div>
<img src="images\ground_truth\GT_munster_000026_000019_leftImg8bit.png" alt="drawing" width="300"/>
<img src="images\predictions\baseline_munster_000026_000019_leftImg8bit.png" alt="drawing" width="300"/>
<img src="images\predictions\DA_standard_munster_000026_000019_leftImg8bit.png" alt="drawing" width="300"/>
<img src="images\predictions\DA_light_munster_000026_000019_leftImg8bit.png" alt="drawing" width="300"/>
<img src="images\predictions\DA_PL_fixthr_munster_000026_000019_leftImg8bit.png" alt="drawing" width="300"/>
<img src="images\predictions\DA_PL_varthr_munster_000026_000019_leftImg8bit.png" alt="drawing" width="300"/>
</div>
The images correspond, in order, to: 
<ul>
<li>ground truth</li>
<li>baseline (BiSeNet trained on labelled Cityscapes)</li>
<li>domain adaptation with standard discriminator</li>
<li>domain adaptation with lightweight discriminator</li>
<li>domain adaptation with pseudo labels and fixed threshold</li>
<li>domain adaptation with pseudo labels and variable threshold</li>
</ul>

## Additional files 
* [`demo.py`](demo.py) provides functions to save a png image with the original image overlapped to the label prediction of a model.
* [`eval.py`](eval.py) is used to perform evaluation on the test dataset
* [`loss.py`](loss.py) contains functions used to compute the loss
* [`make_plots.py`](make_plots.py) is used to create plots of the losses during training
* [`utils.py`](utils.py) contains useful functions to compute accuracy and mIoU of predictions
