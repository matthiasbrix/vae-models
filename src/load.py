import skimage
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

# is reading each volumn/set of files and return it as a list of numpy matrices
# typically a volume has 64 images, each with dimension 384, 384
# returns tensor (64, 384, 384)
def load_volume(root, resize=1):
    sortfunc = lambda f: int(f.split('.')[2].split(' ')[1])
    files = [f for f in listdir(root)]
    files = sorted([f for f in listdir(root) if isfile(join(root, f))], key=sortfunc)
    slices = []
    for f in files:
        data = skimage.img_as_float32(skimage.io.imread(root+f))
        if resize is not 1 and resize > 1:
            #resize_scale = 1/(data.shape[0] / resize)
            #data = skimage.transform.rescale(data, resize, multichannel=False, anti_aliasing=False)
            data = skimage.transform.resize(data, output_shape=(resize, resize), anti_aliasing=True)
            data = data[34:108, :]
        slices.append(data)
    volume = np.stack(slices)
    return volume

def load_all_volumes(volumes, resize=1):
    return [load_volume(f, resize=resize) for f in volumes]

def multi_slice_viewer(volume):
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = 32
    ax.im = 0
    ax.imshow(volume[ax.im][ax.index])
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

if __name__ == "__main__":
    # reading all the sets of images.
    folders = ['../data/lungscans/4uIULSTrSegpltTuNuS44K3t4/1.2.246.352.221.52915333682423613339719948113721836450_OBICone-beamCT/',
            '../data/lungscans/4uIULSTrSegpltTuNuS44K3t4/1.2.246.352.221.55302824863178429077114755927787508155_OBICone-beamCT/',
            '../data/lungscans/4uIULSTrSegpltTuNuS44K3t4/1.2.246.352.221.542181959870340811013566519894670057885_OBICone-beamCT/']
    volumes = [load_volume(f, 128) for f in folders]
    multi_slice_viewer(volumes)
