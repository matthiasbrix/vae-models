import skimage
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

# TODO: use dcim to get timestamps out of it.

def loadVolume(root):
    sortfunc = lambda f: int(f.split('.')[2].split(' ')[1])
    files = sorted([f for f in listdir(root) if isfile(join(root, f))], key=sortfunc)
    slices = []
    for f in files:
        data = skimage.img_as_float32(skimage.io.imread(root+f))
        slices.append(data)
    volume = np.stack(slices)
    return volume

folders = ['data/1.2.246.352.221.52915333682423613339719948113721836450_OBICone-beamCT/',
	    'data/1.2.246.352.221.542181959870340811013566519894670057885_OBICone-beamCT/',
	    'data/1.2.246.352.221.55302824863178429077114755927787508155_OBICone-beamCT/']
volumes = []
for f in folders:
	volumes.append(loadVolume(f))
# volumes = [loadVolume(f) for f in folders]

#remaining code is for interactive viewer. you can change the viewing index with e and r keys
def multi_slice_viewer(volume):
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = 32
    ax.im=0
    ax.imshow(volume[0][ax.index])
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
    
multi_slice_viewer(volumes)