import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
from scipy.stats import norm

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
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/plot_" + name + "_z=" + str(solver.z_dim) + ".png", dpi=height)
    plt.close()

# Plotting train and test losses
def plot_losses(solver):
    train_loss_history = solver.train_loss_history["train_loss_acc"]
    plt.figure(figsize=(6, 5))
    # Plotting the train loss
    if solver.test_loss_history:
        plt.subplot(2, 1, 1)
    plt.yscale("log")
    plt.loglog(np.arange(1, len(train_loss_history)+1), train_loss_history, basey=10, basex=10)
    ticks = np.arange(min(train_loss_history), max(train_loss_history), ((max(train_loss_history)-min(train_loss_history))/len(train_loss_history)))[::2]
    plt.yticks(ticks)
    plt.title("Train loss on data set {}, dim(z)={}".format(solver.loader.dataset, solver.z_dim)) # marginal likelihood log p(x)
    plt.xlabel("epoch")
    plt.ylabel("log loss")

    # Plotting the test loss
    if solver.test_loss_history:
        plt.subplot(2, 1, 2)
        plt.loglog(np.arange(1, len(solver.test_loss_history)+1), solver.test_loss_history, basey=10, basex=10)
        plt.yticks(np.arange(int(min(solver.test_loss_history)), int(max(solver.test_loss_history))+1, 0.5))
        plt.title("Test loss on data set {}, dim(z)={}".format(solver.loader.dataset, solver.z_dim)) # marginal likelihood log p(x)"
        plt.xlabel("epoch")
        plt.ylabel("log loss")

    plt.tight_layout()
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/" + "plot_losses" + "_z=" + str(solver.z_dim) + ".png")
    plt.show()

# Plotting histogram of the latent space's distribution, given the computed \mu and \sigma
# TODO: could be done better? Maybe just have 1 column and then "num_plots" rows
def plot_gaussian_distributions(solver):
    f, axarr = plt.subplots(2, 2, figsize=(8, 6))
    x = np.linspace(-5, 5, 5000)
    idx_x = 0
    idx_y = 0
    plots = np.arange(1, solver.epochs+1, np.ceil(solver.epochs/4)+1).astype(int) #len(solver.train_loss_history["epochs"])//num_plots - 1
    plots[2:] += 1
    plots[-1] = solver.epochs
    f.subplots_adjust(hspace=0.5, wspace=0.3)
    for idx in plots:
        i = idx-1
        epoch, mu_z, std_z, varmu_z, expected_var_z = solver.train_loss_history["epochs"][i], solver.z_stats_history["mu_z"][i],\
            solver.z_stats_history["std_z"][i], solver.z_stats_history["varmu_z"][i], solver.z_stats_history["expected_var_z"][i]
        var_z = np.power(std_z, 2)
        print("epoch: %d, mu(z): %.4f, stddev(z): %.4f, var(z): %.4f, var(mu(z)): %.4f E[var(q(z|x)]: %.4f" % (\
                epoch, mu_z, std_z, var_z, varmu_z, expected_var_z))
        y = (1 / (np.sqrt(2 * np.pi * var_z))) * \
                (np.power(np.e, -(np.power((x - mu_z), 2) / (2 * var_z))))
        axarr[idx_x, idx_y].plot(x, y)
        axarr[idx_x, idx_y].set_title("epoch %d\nμ(z)=%.4f, σ^2(z)=%.4f" % (epoch, mu_z, var_z))
        if idx_x == idx_y or idx_x > idx_y:
            idx_y += 1
        else: # idx_x < idx_y
            tmp = idx_y
            idx_y = idx_x
            idx_x = tmp

    for ax in axarr.flat:
        ax.set(xlabel='x', ylabel='y')
    
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/" + "plot_gaussian" + "_z=" + str(solver.z_dim) + ".png")

# Plot the reconstruction loss and KL divergence in two separate plots
def plot_rl_kl(solver):
    rls = solver.train_loss_history["recon_loss_acc"]
    kls = solver.train_loss_history["kl_diverg_acc"]
    plt.figure(figsize=(6, 5))

    plt.subplot(2, 1, 1)
    plt.loglog(np.arange(1, len(rls)+1), rls, basey=10, basex=10)
    ticks = np.arange(min(rls), max(rls), ((max(rls)-min(rls))/len(rls)))[::5]
    plt.yticks(ticks)
    plt.xlabel("epoch")
    plt.ylabel("log loss")
    plt.title("Reconstruction loss during training") # marginal log likelihood

    plt.subplot(2, 1, 2)
    plt.loglog(np.arange(1, len(kls)+1), kls, basey=10, basex=10)
    ticks = np.arange(min(kls), max(kls), ((max(kls)-min(kls))/len(kls)))[::10]
    plt.yticks(ticks)
    plt.xlabel("epoch")
    plt.ylabel("log divergence")
    plt.title("KL divergence of q(z|x)||p(z) during training")

    plt.tight_layout()
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/" + "plot_rl_kl" + "_z=" + str(solver.z_dim) + ".png")
    plt.show()

# Plot the latent space as scatter plot
def plot_latent_space(solver):
    labels = solver.labels.tolist()
    plt.figure(figsize=(9, 7))
    plt.scatter(solver.latent_space[:, 0], solver.latent_space[:, 1], s=10, c=labels, cmap=plt.cm.get_cmap('cubehelix', solver.loader.n_classes))
    plt.xlabel("z_1")
    plt.ylabel("z_2")
    plt.colorbar()
    plt.title("Latent space of the VAE on data set {} after {} epochs".format(solver.loader.dataset, solver.epochs))
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/plot_latent_space" + "_z=" + str(solver.z_dim) + ".png")

# Plot the latent space as scatter plot (no labels)
def plot_latent_space_no_labels(solver):
    plt.figure(figsize=(9, 7))
    plt.scatter(solver.latent_space[:, 0], solver.latent_space[:, 1], s=10)
    plt.xlabel("z_1")
    plt.ylabel("z_2")
    plt.title("Latent space of the VAE on data set {} after {} epochs".format(solver.loader.dataset, solver.epochs))
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/plot_latent_space" + "_z=" + str(solver.z_dim) + ".png")

# from https://github.com/Natsu6767/Variational-Autoencoder/blob/master/main.py
# Since the prior of the latent space is Gaussian, linearly spaced coordinates on the unit square
# were transformed through the inverse CDF of the Gaussian to produce values of the latent
# variables z. For each of these values z, we plotted the corresponding generative
# p(x|z) with the learned parameters θ.
def plot_latent_manifold(solver, cm, n=20, fig_size=(10, 10)):
    x, y = solver.loader.img_dims
    figure = np.zeros((x*n, y*n))
    # Construct grid of latent variable values.
    # ppf is percent point function (inverse of CDF)
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    #Decode for each square in the grid.
    for i, xi in enumerate(grid_x):
        for j, yj in enumerate(grid_y):
            z_sample = np.array([xi, yj])
            z_sample = np.tile(z_sample, solver.batch_size).reshape(solver.batch_size, solver.z_dim)
            z_sample = torch.from_numpy(z_sample).float().to(solver.device)
            x_decoded = solver.model.decoder(z_sample).cpu().detach().numpy()
            img = np.reshape(x_decoded[0], list(solver.loader.img_dims))
            figure[i * x: (i+1) * x,
            j * y: (j+1) * y] = img

    plt.figure(figsize=fig_size)
    plt.axis("off")
    plt.imshow(figure, cmap=cm)
    plt.show()
    save_plot_fig(solver, figure, cm=cm, name="learned_data_manifold")

# takes only numpy array in, so mainly for testing puposes
def plot_faces_grid(n, n_cols, solver, fig_size=(10, 8)):
    img_rows, img_cols = solver.loader.img_dims
    n_rows = int(np.ceil(n / float(n_cols)))
    figure = np.zeros((img_rows * n_rows, img_cols * n_cols))
    for k, x in enumerate(solver.loader.data[:n]):
        r = k // n_cols
        c = k % n_cols
        figure[r * img_rows: (r + 1) * img_rows,
               c * img_cols: (c + 1) * img_cols] = x.reshape(list(solver.loader.img_dims))       
    plt.figure(figsize=fig_size)
    plt.imshow(figure, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    save_plot_fig(solver, figure, cm="gray", name="faces_grid")

# plot sample faces in a grid
def plot_faces_samples_grid(n, n_cols, solver, fig_size=(10, 8)):
    img_rows, img_cols = solver.loader.img_dims
    n_rows = int(np.ceil(n / float(n_cols)))
    figure = np.zeros((img_rows * n_rows, img_cols * n_cols))
    samples = torch.randn(n, solver.z_dim).to(solver.device)
    samples = solver.model.decoder(samples).cpu().detach().numpy()
    for k, x in enumerate(samples):
        r = k // n_cols
        c = k % n_cols
        figure[r * img_rows: (r + 1) * img_rows,
               c * img_cols: (c + 1) * img_cols] = x.reshape(list(solver.loader.img_dims))       
    plt.figure(figsize=fig_size)
    plt.imshow(figure, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    save_plot_fig(solver, figure, cm="gray", name="faces_samples_grid")

# Old - can be removed?
# Useful if form is (10586, 1850) so no tensor
# Plot gallery of faces
def plot_gallery_old(images, img_dims, n_row=12, n_col=12):
    gs = gridspec.GridSpec(n_row, n_col)
    # set the space between subplots and the position of the subplots in the figure
    gs.update(wspace=0.0, hspace=0.0, left=0.1, right=0.4, bottom=0.1, top=0.4) # adjust right and top for size
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    #plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i, g in enumerate(gs): # range(n_row * n_col):
        plt.subplot(g) #n_row, n_col, i + 1)
        image = images[i].reshape(*img_dims).astype(int)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
