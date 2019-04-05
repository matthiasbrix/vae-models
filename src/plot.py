import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_gallery(images, img_dims, n_row=8, n_col=8):
    gs = gridspec.GridSpec(n_row, n_col)
    # set the space between subplots and the position of the subplots in the figure
    gs.update(wspace=0.0, hspace=0.0, left = 0.1, right = 0.4, bottom = 0.1, top = 0.4) # adjust right and top for size
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    #plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i, g in enumerate(gs): # range(n_row * n_col):
        plt.subplot(g) #n_row, n_col, i + 1)
        image = images[i].reshape(*img_dims).astype(int)
        plt.imshow(image) # , cmap=plt.cm.gray
        plt.axis("off")
        plt.tight_layout()