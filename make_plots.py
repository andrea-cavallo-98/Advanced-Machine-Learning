import numpy as np
from matplotlib import pyplot as plt


loss_seg = []
loss_adv = []
loss_D = []

with open("../DA_training_log.txt", "r") as f:
    for line in f:
        if "loss_seg" in line:
            loss_D.append(float(line[-7:-2]))
            loss_adv.append(float(line[-24:-19]))
            loss_seg.append(float(line[-42:-37]))


plt.figure()
plt.plot(loss_seg)
plt.plot(loss_adv)
plt.plot(loss_D)
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.legend(["Segmentation loss", "Adversarial loss", "Discriminator loss"])
plt.show()