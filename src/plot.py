import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import itertools

# Saves figure without white space borders
# from https://fengl.org/2014/07/09/matplotlib-savefig-without-borderframe/
def save_plot_fig(solver, data, cm, name):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=cm)
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/plot_" + name + "_z=" + str(solver.z_dim) + ".png", dpi = height)
    plt.close()

# Plotting train and test losses
def plot_losses(solver, train_loss_history, test_loss_history):
    #fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    #fig.figsize((12,4))
    # add a big axes, hide frame
    #fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    #plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    #plt.grid(False)
    #plt.xlabel("epoch")
    #plt.ylabel("loss")
    #for i, ax in enumerate(axes):
    #    if i == 0:
    #        ax.plot(np.arange(1, len(train_loss_history)+1), train_loss_history, '-o')
    #    if i == 1:
    #        ax.plot(np.arange(1, len(test_loss_history)+1), test_loss_history, '-o')
    #ax = f.add_subplot(121) # 211
    #ax2 = f.add_subplot(122) # 212
    xaxis = np.arange(1, len(test_loss_history)+1)
    plt.figure(figsize=(6, 5))
    # Plotting the train loss
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(1, len(train_loss_history)+1), train_loss_history, '-o')
    plt.title("Train loss") # marginal likelihood log p(x)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xticks(xaxis)

    # Plotting the test loss
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(1, len(test_loss_history)+1), test_loss_history, '-o')
    plt.title("Test loss") # marginal likelihood log p(x)"
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xticks(xaxis)
    
    plt.tight_layout()
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/" + "plot_losses" + "_z=" + str(solver.z_dim) + ".png")
    plt.show()

# Plotting histogram of the latent space's distribution, given the computed \mu and \sigma
def plot_gauss_distributions(solver, num_normal_plots):
    x = np.linspace(-5, 5, 5000)
    plot_cols = np.arange(1, num_normal_plots+1)
    for idx, stats in enumerate(solver.z_stats):
        epoch, mu_z, std_z, varmu_z = stats
        print("epoch: {}, mu(z): {}, stddev(z): {}, var(z): {}, var(mu(z)): {}".format(\
                epoch, mu_z, std_z, np.power(std_z, 2), varmu_z))
        y = (1 / (np.sqrt(2 * np.pi * np.power(std_z, 2)))) * \
                (np.power(np.e, -(np.power((x - mu_z), 2) / (2 * np.power(std_z, 2)))))
        plt.subplot(2, 2, plot_cols[idx])
        plt.plot(x, y)
        plt.title("epoch {}, dim(z)={}".format(epoch, solver.z_dim))
        plt.xlabel("x")
        plt.ylabel("y")

        plt.tight_layout()
        plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/" + "gaussian" + "_z=" + str(solver.z_dim) + ".png")
        plt.show()
    
    #x = np.linspace(0, 2 * np.pi, 400)
    #y = np.sin(x ** 2)
    f, axarr = plt.subplots(int(num_normal_plots/2), int(num_normal_plots/2))
    # TODO: Find out how to compute this and then refactor the plotting!
    # https://stackoverflow.com/questions/3099987/generating-permutations-with-repetitions-in-python
    # https://matplotlib.org/gallery/subplots_axes_and_figures/subplots_demo.html
    axarr[0, 0].plot(x, y)
    axarr[0, 0].set_title('Axis [0,0]')
    axarr[0, 1].scatter(x, y)
    axarr[0, 1].set_title('Axis [0,1]')
    axarr[1, 0].plot(x, y ** 2)
    axarr[1, 0].set_title('Axis [1,0]')
    axarr[1, 1].scatter(x, y ** 2)
    axarr[1, 1].set_title('Axis [1,1]')
    for ax in axarr.flat:
        ax.set(xlabel='x-label', ylabel='y-label')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axarr.flat:
        ax.label_outer()
    
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/" + "plot_gaussian" + "_z=" + str(solver.z_dim) + ".png")

def plot_rl_kl(solver, rls, kls):
    xaxis = np.arange(1, solver.epochs+1)
    plt.figure(figsize=(6, 5))

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(1, len(rls)+1), rls, '-o')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Reconstruction loss / ELBO(q, x)")
    plt.xticks(xaxis)

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(1, len(kls)+1), kls, '-o')
    plt.xlabel("epoch")
    plt.ylabel("divergence")
    plt.title("KL divergence of q(z|x)||p(z)")
    plt.xticks(xaxis)

    plt.tight_layout()
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/" + "plot_rl_kl" + "_z=" + str(solver.z_dim) + ".png")
    plt.show()

def plot_vis_latent_manifold(solver):
    # from https://github.com/Natsu6767/Variational-Autoencoder/blob/master/main.py
    n = 20 # figure with 20x20 grid
    x, y = solver.img_dims
    figure = np.zeros((x*n, y*n))

    #Contruct grid of latent variable values.
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    #Decode for each square in the grid.
    for i, xi in enumerate(grid_x):
        for j, yj in enumerate(grid_y):
            z_sample = np.array([xi, yj])
            z_sample = np.tile(z_sample, solver.batch_size).reshape(solver.batch_size, solver.z_dim)
            z_sample = torch.from_numpy(z_sample).float()
            x_decoded = solver.model.decoder(z_sample).detach().numpy()
            digit = np.reshape(x_decoded[0], list(solver.img_dims))
            figure[i * x: (i+1) * x,
            j * y: (j+1) * y] = digit

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(figure, cmap="bone")
    plt.show()
    save_plot_fig(solver, figure, cm="bone", name="learned_data_manifold")

def plot_latent_space(solver):
    labels = solver.labels.tolist()
    plt.figure(figsize=(8, 6))
    plt.scatter(solver.latent_space[:, 0], solver.latent_space[:, 1], c=labels, cmap='brg')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar() # show color scale
    #plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/" + "latent_space" + "_z=" + str(solver.z_dim) + ".png")
    save_plot_fig(solver, solver.latent_space, cm="brg", name="latent_space")

# Plot gallery of faces for LFW
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
    # TODO: save fig