import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(img):
    if img.max() > 1.0:
        img = img / 255.0

    if type(img) == np.ndarray:
        if len(img.shape) == 3:
            plt.imshow(img)
            plt.show()
        else:
            plt.imshow(img, cmap='gray')
            plt.show()
    else:
        with torch.no_grad():
            img = img.clone()
            img = img.to("cpu")

            if img.min() < 0.0: # "des"normalizar
                img = img / 2 + 0.5 # img * std + med,

            npimg = img.detach().numpy()
            if len(npimg.shape) == 3:
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
                plt.show()
            else:
                plt.imshow(npimg, cmap='gray')
                plt.show()

def show_losses(losses, iteration):
    result = "itr: %s, " %str(iteration)

    for tipo in losses:
        result += "%s: [" %tipo

        for loss in losses[tipo]:
            result += "%s: %.2f, " %(loss, losses[tipo][loss].mean().item())

        result = result[:-1] if result[-1] == ' ' else result
        result += '], '

    result = result[:-2] if result[-2:] == ", " else result
    print(result)