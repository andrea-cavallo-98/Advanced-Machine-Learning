import numpy as np
from matplotlib import pyplot as plt

BASELINE = True
PSEUDO_LABELS = True

loss = []
acc = []
miou = []
loss_seg = []
loss_adv = []
loss_D = []
loss_seg_trg = []

with open("baseline.rtf", "r") as f:
    for line in f:
        if BASELINE:
            if "epoch" in line:
                loss.append(float(line[-11:-3]))
            if "precision" in line:
                acc.append(float(line[-7:-2]))
            if "mIoU" in line:
                miou.append(float(line[-7:-2]))

        elif not PSEUDO_LABELS:
            if "loss_seg" in line:
                loss_D.append(float(line[-7:-2]))
                loss_adv.append(float(line[-24:-19]))
                loss_seg.append(float(line[-42:-37]))

        else:
            if "loss_seg" in line:
                loss_D.append(float(line[-7:-2]))
                loss_adv.append(float(line[-24:-19]))
                loss_seg_trg.append(float(line[-42:-37]))
                loss_seg.append(float(line[-63:-58]))

if BASELINE:
    plt.figure()
    plt.plot(loss)
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.savefig("BASELINE_loss.png")

    x = np.arange(4, 50, 4)
    plt.figure()
    plt.plot(x, acc)
    plt.plot(x, miou)
    plt.xlabel("Number of epochs")
    plt.ylabel("PA and mIoU")
    plt.legend(["PA", "mIoU"])
    plt.savefig("BASELINE.png")


elif PSEUDO_LABELS:
    plt.figure()
    plt.plot(loss_seg)
    plt.plot(loss_seg_trg - 0.001 * np.array(loss_adv))
    plt.plot(loss_adv)
    plt.plot(loss_D)
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend(["Segmentation loss", "Pseudo-labels loss", "Adversarial loss", "Discriminator loss"])
    plt.savefig("DA_PL_loss.png")


else:
    plt.figure()
    plt.plot(loss_seg)
    plt.plot(loss_adv)
    plt.plot(loss_D)
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend(["Segmentation loss", "Adversarial loss", "Discriminator loss"])
    plt.savefig("DA_loss.png")