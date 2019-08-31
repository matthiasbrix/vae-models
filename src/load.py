import skimage
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

# Author: Oswin Krause

# is reading each volumn/set of files and return it as a list of numpy matrices
# typically a volume has 64 images, each with dimension 384, 384
# returns matrix (64, 384, 384)
def load_volume(root):
    sortfunc = lambda f: int(f.split('.')[2].split(' ')[1])
    files = [f for f in listdir(root)]
    files = sorted([f for f in listdir(root) if isfile(join(root, f))], key=sortfunc)
    slices = []
    for f in files:
        data = skimage.img_as_float32(skimage.io.imread(root+f))
        slices.append(data)
    volume = np.stack(slices)
    return volume

def load_all_volumes(volumes):
    return [load_volume(f) for f in volumes]

# remaining code is for interactive viewer. you can change the viewing index with e and r keys
def multi_slice_viewer(volume):
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = 32
    ax.im = 0
    ax.imshow(volume[0][ax.im])
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'e':
        ax.index = (ax.index - 1) % ax.volume[ax.im].shape[0]
    elif event.key == 'r':
        ax.index = (ax.index + 1) % ax.volume[ax.im].shape[0]
    elif event.key == 't':
        ax.im = (ax.im - 1) % len(ax.volume)
    elif event.key == 'y':
        ax.im = (ax.im + 1) % len(ax.volume)
    print(ax.im, ax.index)
    ax.images[0].set_array(ax.volume[ax.im][ax.index])
    fig.canvas.draw()

# Run this script to run the images that are enlisted in the set of images.
if __name__ == "__main__":
    # reading all the sets of images.
    folders = ['../data/lungscans/4uIULSTrSegpltTuNuS44K3t4/1.2.246.352.221.52915333682423613339719948113721836450_OBICone-beamCT/',
            '../data/lungscans/4uIULSTrSegpltTuNuS44K3t4/1.2.246.352.221.55302824863178429077114755927787508155_OBICone-beamCT/',
            '../data/lungscans/4uIULSTrSegpltTuNuS44K3t4/1.2.246.352.221.542181959870340811013566519894670057885_OBICone-beamCT/']
    volumes = [load_volume(f) for f in folders]
    multi_slice_viewer(volumes)
