import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats

# Auxiliary function for saving nice plots
# Saves figure without white space borders
# from https://fengl.org/2014/07/09/matplotlib-savefig-without-borderframe/
def _save_plot_fig(solver, data, cm, name):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=cm)
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/plot_" + name + "_" + solver.loader.dataset + "_z=" + str(solver.z_dim) + ".png", dpi=height)
    plt.close()

def _xticks(ls, ticks_rate):
    labels = np.arange(1, len(ls)+2, (len(ls)//ticks_rate))
    labels[1:] -= 1
    labels[-1] = len(ls)
    return labels.astype(int)

# Plotting train and test losses
def plot_losses(solver, ticks_rate):
    train_loss_history = solver.train_loss_history["train_loss_acc"]
    test_loss_history = solver.test_loss_history
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, len(train_loss_history)+1), train_loss_history, label="Train")
    plt.plot(np.arange(1, len(solver.test_loss_history)+1), test_loss_history, label="Test")
    plt.xticks(_xticks(train_loss_history, ticks_rate))
    plt.title("Loss on data set {}, dim(z)={}".format(solver.loader.dataset, solver.z_dim)) # marginal likelihood log p(x)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(bbox_to_anchor=(1.3, 1), borderaxespad=0)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.2)
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/" + "plot_losses_" + solver.loader.dataset + "_z=" + str(solver.z_dim) + ".png")
    plt.show()

# Plotting histogram of the latent space's distribution, given the computed \mu and \sigma
# TODO: could be done better? Maybe just have 1 column and then "num_plots" rows
def plot_gaussian_distributions(solver):
    f, axarr = plt.subplots(2, 2, figsize=(8, 6))
    x = np.linspace(-5, 5, 5000)
    idx_x = 0
    idx_y = 0
    epochs = len(solver.train_loss_history["train_loss_acc"]) # in case run was canceled
    if epochs % 2 != 0:
        plots = np.arange(1, epochs+1, np.ceil(epochs/4)+1).astype(int)
        plots[2:] += 1
        plots[-1] = epochs
    else:
        plots = np.arange(1, epochs+1, np.ceil(epochs/4)).astype(int)
        plots[1:] -= 1
        plots[-1] = epochs
    f.subplots_adjust(hspace=0.5, wspace=0.3)
    ys = []
    for idx in plots:
        i = idx-1
        epoch, mu_z, std_z, varmu_z, expected_var_z = solver.train_loss_history["epochs"][i], solver.z_stats_history["mu_z"][i],\
            solver.z_stats_history["std_z"][i], solver.z_stats_history["varmu_z"][i], solver.z_stats_history["expected_var_z"][i]
        var_z = np.power(std_z, 2)
        print("epoch: %d, mu(z): %.4f, stddev(z): %.4f, var(z): %.4f, var(mu(z)): %.4f E[var(q(z|x)]: %.4f" % (\
                epoch, mu_z, std_z, var_z, varmu_z, expected_var_z))
        y = (1 / (np.sqrt(2 * np.pi * var_z))) * \
                (np.power(np.e, -(np.power((x - mu_z), 2) / (2 * var_z))))
        ys.append(np.max(y))
        axarr[idx_x, idx_y].plot(x, y, label="Latent distr.")
        axarr[idx_x, idx_y].plot(x, stats.norm.pdf(x, 0, 1), label="Standard\nGaussian distr.")
        axarr[idx_x, idx_y].set_title("epoch %d\nμ(z)=%.4f, σ^2(z)=%.4f" % (epoch, mu_z, var_z))
        if idx_x == idx_y or idx_x > idx_y:
            idx_y += 1
        else: # idx_x < idx_y
            tmp = idx_y
            idx_y = idx_x
            idx_x = tmp

    for ax in axarr.flat:
        maxys = max(ys)
        maxnorm = max(stats.norm.pdf(x, 0, 1))
        ax.set_ylim([0, max(maxys, maxnorm)+0.05])
        ax.set(xlabel='x', ylabel='y')

    with open(solver.folder_prefix + solver.loader.folder_name + "/result_stats_" +\
        solver.loader.dataset + "_z=" + str(solver.z_dim) + ".txt", 'w') as file_res:
        file_res.write("epoch,var(mu(z)),E[var(q(z|x))]\n")
        for idx in plots:
            i = idx-1
            epoch, varmu_z, expected_var_z = solver.train_loss_history["epochs"][i],\
            solver.z_stats_history["varmu_z"][i], solver.z_stats_history["expected_var_z"][i]
            file_res.write(str(epoch) + "," + str(np.around(np.array(varmu_z), 4)) + "," + str(np.around(np.array(expected_var_z.item()), 4)))
            file_res.write("\n")

    f.subplots_adjust(top=0.9, left=0.1, right=0.8, bottom=0.1)
    axarr.flatten()[1].legend(bbox_to_anchor=(1.65, 1.0), borderaxespad=0)
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/" + "plot_gaussian_" + solver.loader.dataset + "_z=" + str(solver.z_dim) + ".png")

# Plot the reconstruction loss and KL divergence in two separate plots
def plot_rl_kl(solver, ticks_rate):
    rls = solver.train_loss_history["recon_loss_acc"]
    kls = solver.train_loss_history["kl_diverg_acc"]
    x = np.arange(1, len(kls)+1)
    plt.figure(figsize=(5, 5))

    plt.subplot(2, 1, 1)
    plt.plot(x, rls)
    plt.xticks(_xticks(rls, ticks_rate))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Reconstruction loss in (training)") # marginal log likelihood

    plt.subplot(2, 1, 2)
    plt.plot(x, kls) # KL div
    plt.xticks(_xticks(kls, ticks_rate))
    plt.xlabel("epoch")
    plt.ylabel("KL divergence")
    plt.title("KL divergence of q(z|x)||p(z) (training)")

    plt.tight_layout()
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/" + "plot_rl_kl_" + solver.loader.dataset + "_z=" + str(solver.z_dim) + ".png")
    plt.show()

# Plot the latent space as scatter plot
def plot_latent_space(solver):
    labels = solver.labels.tolist()
    plt.figure(figsize=(9, 7))
    plt.scatter(solver.latent_space[:, 0], solver.latent_space[:, 1], s=10, c=labels, cmap=plt.cm.get_cmap("Paired", solver.loader.n_classes))
    plt.xlabel("z_1")
    plt.ylabel("z_2")
    plt.colorbar()
    plt.title("Latent space q(z) on data set {} after {} epochs".format(solver.loader.dataset, solver.epochs))
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/plot_latent_space_" + solver.loader.dataset + "_z=" + str(solver.z_dim) + ".png")

# Plot the latent space as scatter plot (no labels)
def plot_latent_space_no_labels(solver):
    plt.figure(figsize=(9, 7))
    plt.scatter(solver.latent_space[:, 0], solver.latent_space[:, 1], s=10, cmap="Paired")
    plt.xlabel("z_1")
    plt.ylabel("z_2")
    plt.title("Latent space of the VAE on data set {} after {} epochs".format(solver.loader.dataset, solver.epochs))
    plt.savefig(solver.folder_prefix + solver.loader.folder_name + "/plot_latent_space_" + solver.loader.dataset + "_z=" + str(solver.z_dim) + ".png")

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
    grid_x = stats.norm.ppf(np.linspace(0.05, 0.95, n)) # np.linspace(-4, 4, n) #
    grid_y = stats.norm.ppf(np.linspace(0.05, 0.95, n)) # np.linspace(-4, 4, n) #

    #Decode for each square in the grid.
    for i, xi in enumerate(grid_x):
        for j, yj in enumerate(grid_y):
            z_sample = np.array([xi, yj])
            z_sample = np.tile(z_sample, solver.batch_size).reshape(solver.batch_size, solver.z_dim) # repeating the sample to batch_size x dim(z)
            z_sample = torch.from_numpy(z_sample).float().to(solver.device) # transform to tensor
            x_decoded = solver.model.decoder(z_sample).cpu().detach().numpy() # decode it
            img = np.reshape(x_decoded[0], list(solver.loader.img_dims))
            figure[i * x: (i+1) * x,
            j * y: (j+1) * y] = img

    plt.figure(figsize=fig_size)
    plt.axis("off")
    plt.imshow(figure, cmap=cm)
    plt.show()
    _save_plot_fig(solver, figure, cm=cm, name="learned_data_manifold")

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
    _save_plot_fig(solver, figure, cm="gray", name="faces_grid")

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
    _save_plot_fig(solver, figure, cm="gray", name="faces_samples_grid")

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
