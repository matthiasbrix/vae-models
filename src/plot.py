import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import scipy.stats as stats

DATASETS = {
    "MNIST": "MNIST",
    "FF": "Frey Faces",
    "LFW": "Labeled Faces in the Wild"
}

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
    plt.savefig(solver.data_loader.result_dir + "/plot_" + name + "_" + \
        solver.data_loader.dataset + "_z=" + str(solver.z_dim) + ".png", dpi=height)
    plt.close()

def _xticks(ls, ticks_rate):
    labels = np.arange(1, len(ls)+2, (len(ls)//ticks_rate))
    labels[1:] -= 1
    labels[-1] = len(ls)
    return labels.astype(int)

# Plotting train and test losses
def plot_losses(solver):
    train_loss_history = solver.train_loss_history["train_loss_acc"]
    test_loss_history = solver.test_loss_history
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, len(train_loss_history)+1), train_loss_history, label="Train")
    plt.plot(np.arange(1, len(solver.test_loss_history)+1), test_loss_history, label="Test")
    ticks_rate = 4 if len(train_loss_history) >= 4 else len(train_loss_history)
    plt.xticks(_xticks(train_loss_history, ticks_rate))
    plt.title("Loss on data set {}, dim(z)={}".format(DATASETS[solver.data_loader.dataset], solver.z_dim)) # marginal likelihood log p(x)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4),
            fancybox=True, shadow=True, ncol=5)
    plt.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.25)
    plt.savefig(solver.data_loader.result_dir + "/" + "plot_losses_" + \
        DATASETS[solver.data_loader.dataset] + "_z=" + str(solver.z_dim) + ".png")
    plt.show()

# Plotting histogram of the latent space's distribution, given the computed \mu and \sigma
# TODO: could be done better? Maybe just have 1 column and then "num_plots" rows
# TODO: test with 5 epochs...
def plot_gaussian_distributions(solver):
    x = np.linspace(-5, 5, 5000)
    idx_x = 0
    idx_y = 0
    epochs = len(solver.train_loss_history["train_loss_acc"]) # in case run was canceled
    #if epochs % 2 != 0:
    #    plots = np.arange(1, epochs+1, np.ceil(epochs/4)+1).astype(int)
    #    plots[2:] += 1
    #    plots[-1] = epochs
    #else:
    #    plots = np.arange(1, epochs+1, np.ceil(epochs/4)).astype(int)
    #    plots[1:] -= 1
    #    plots[-1] = epochs
    plots = np.arange(1, epochs+1, np.ceil(epochs/4)).astype(int)
    plots[-1] = epochs
    if epochs == 1:
        f, axarr = plt.subplots(1, 1, figsize=(8, 2))
    if epochs == 2 or epochs == 3:
        f, axarr = plt.subplots(1, 2, figsize=(8, 4))
    if epochs >= 4:
        f, axarr = plt.subplots(2, 2, figsize=(8, 6))
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
        if epochs <= 3:
            axarr[idx_y].plot(x, y, label="Latent distr.")
            axarr[idx_y].plot(x, stats.norm.pdf(x, 0, 1), label="Standard\nGaussian distr.")
            axarr[idx_y].set_title("epoch %d\nμ(z)=%.4f, σ^2(z)=%.4f" % (epoch, mu_z, var_z))
        else:
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

    # writing stats results of z to file
    with open(solver.data_loader.result_dir + "/result_stats_" +\
        solver.data_loader.dataset + "_z=" + str(solver.z_dim) + ".txt", 'w') as file_res:
        file_res.write("epoch,var(mu(z)),E[var(q(z|x))]\n")
        for idx in plots:
            i = idx-1
            epoch, varmu_z, expected_var_z = solver.train_loss_history["epochs"][i],\
            solver.z_stats_history["varmu_z"][i], solver.z_stats_history["expected_var_z"][i]
            file_res.write(str(epoch) + "," + str(np.around(np.array(varmu_z), 4)) + "," + str(np.around(np.array(expected_var_z.item()), 4)))
            file_res.write("\n")

    if epochs <= 3:
        ax = axarr.flatten()[0]
    if epochs >= 4:
        ax = axarr.flatten()[2]
    f.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.2)
    ax.legend(loc='upper center', bbox_to_anchor=(1.2, -0.25),
            fancybox=True, shadow=True, ncol=5)

    plt.savefig(solver.data_loader.result_dir + "/" + "plot_gaussian_" +\
        solver.data_loader.dataset + "_z=" + str(solver.z_dim) + ".png")

# Plot the reconstruction loss and KL divergence in two separate plots
def plot_rl_kl(solver):
    rls = solver.train_loss_history["recon_loss_acc"]
    kls = solver.train_loss_history["kl_diverg_acc"]
    x = np.arange(1, len(kls)+1)
    plt.figure(figsize=(4.5, 5))

    ticks_rate = 4 if len(rls) >= 4 else len(rls)
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
    plt.title("KL divergence of q(z|x)||p(z), β={} (training)".format(solver.beta))

    plt.tight_layout()
    plt.savefig(solver.data_loader.result_dir + "/" + "plot_rl_kl_" \
        + solver.data_loader.dataset + "_z=" + str(solver.z_dim) + ".png")
    plt.show()

# Plot the latent space as scatter plot with and without labels
def plot_latent_space(solver, space, ticks=None, var=None, title=None, labels=None):
    plt.figure(figsize=(9, 7))
    if solver.data_loader.with_labels:
        if var == "z":
            scatter = plt.scatter(space[:, 0], space[:, 1], s=10, vmin=ticks[0], vmax=ticks[-1], c=labels.tolist(), cmap=plt.cm.get_cmap("Paired", 6))
            clb = plt.colorbar(scatter, ticks=ticks)
            clb.ax.set_title(title)
        elif var == "y":
            scatter = plt.scatter(space[:, 0], space[:, 1], s=10, vmin=ticks[0], vmax=ticks[-1], c=labels.tolist(), cmap="Paired") #plt.cm.get_cmap("Paired", 12)
            clb = plt.colorbar(scatter, ticks=ticks)
            clb.ax.set_title(title)
        else:
            plt.scatter(space[:, 0], space[:, 1], s=10, c=labels.tolist(), cmap=plt.cm.get_cmap("Paired", solver.data_loader.n_classes))
            clb = plt.colorbar()
            clb.ax.set_title(title)
    else:
        plt.scatter(space[:, 0], space[:, 1], s=10, cmap="Paired")
    plt.xlabel("{}_1".format(var))
    plt.ylabel("{}_2".format(var))
    plt.title("Latent space q({}) on data set {} after {} epochs".format(var, DATASETS[solver.data_loader.dataset], solver.epochs))
    plt.savefig(solver.data_loader.result_dir + "/plot_" + str(var) + "_space_" \
        + solver.data_loader.dataset + "_z=" + str(solver.z_dim) + ".png")

# For each of the values z, we plotted the corresponding generative
# p(x|z) with the learned parameters θ.
def plot_latent_manifold(solver, cm, grid_x, grid_y, n=20, fig_size=(10, 10), x_t=None):
    x, y = solver.data_loader.img_dims
    figure = np.zeros((x*n, y*n))
    # Decode for each square in the grid.
    solver.model.eval()
    with torch.no_grad():
        for i, xi in enumerate(grid_x):
            for j, yj in enumerate(grid_y):
                z_sample = np.array([xi, yj])
                z_sample = np.tile(z_sample, 1).reshape(1, solver.z_dim)
                z_sample = torch.from_numpy(z_sample).float().to(solver.device) # transform to tensor
                if solver.cvae_mode:
                    idx = torch.randint(0, solver.data_loader.n_classes, (1,)).item()
                    y_sample = torch.FloatTensor(torch.zeros(z_sample.size(0), solver.data_loader.n_classes)).to(solver.device)
                    y_sample[:, idx] = 1.
                    sample = torch.cat((z_sample, y_sample), dim=-1)
                elif solver.tdcvae_mode:
                    x_t = x_t.to(solver.device)
                    sample = torch.cat((x_t, z_sample), dim=-1)
                else:
                    sample = z_sample
                x_decoded = solver.model.decoder(sample).cpu().detach().numpy()
                img = np.reshape(x_decoded[0], list(solver.data_loader.img_dims))
                figure[i*x:(i+1)*x, j*y:(j+1)*y] = img

    plt.figure(figsize=fig_size)
    plt.axis("off")
    plt.xlabel("z_1")
    plt.ylabel("z_2")
    plt.imshow(figure, cmap=cm)
    plt.show()
    with open(solver.data_loader.result_dir + "/plot_learned_data_manifold_grids_" +\
        solver.data_loader.dataset + "_z=" + str(solver.z_dim) + ".txt", 'w') as file_res:
        file_res.write("grid_x: {}\n".format(grid_x))
        file_res.write("grid_y: {}\n".format(grid_y))
    _save_plot_fig(solver, figure, cm=cm, name="learned_data_manifold")

# Replicating the handstyle image example from Kingma et. al in Semisupervised VAE paper
# Take a single test set image (first from each batch), encode it, use that fixed z, 
# loop over all labels and print a row out with the fixed z but different labeled images
def plot_with_fixed_z(solver, n_rows, n_cols, cm, fig_size=(6, 6)):
    img_rows, img_cols = solver.data_loader.img_dims
    figure = np.zeros((img_rows*n_rows, img_cols*n_cols))
    solver.model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(solver.data_loader.test_loader):
            if i == 10:
                break
            x, y = data.to(solver.device)[0], target.to(solver.device)[0]
            y = y.view(1, 1)
            onehot = solver.model.onehot_encoding(y)
            x = x.view(-1, solver.data_loader.input_dim)
            x_new = torch.cat((x, onehot), dim=-1)
            mu_x, logvar_x = solver.model.encoder(x_new)
            z = solver.model.reparameterization_trick(mu_x, logvar_x)
            figure[i*img_rows:(i+1)*img_rows, 0:img_cols] = x.view(1, *solver.data_loader.img_dims).cpu().numpy() # the test image in leftmost column
            for label in range(10): # just hardcoded to 10 outputs pr. row, otherwise call solver.data_loader.n_classes (but careful)
                onehot = torch.FloatTensor(torch.zeros(y.size(0), solver.model.y_size))
                onehot.zero_()
                onehot[:, label] = 1.
                z_new = torch.cat((z, onehot), dim=-1)
                decoded = solver.model.decoder(z_new).view(1, *solver.data_loader.img_dims).cpu().numpy()
                figure[i*img_rows:(i+1)*img_rows, (label+1)*img_cols: (label+2)*img_cols] = decoded
    
    plt.figure(figsize=fig_size)
    plt.axis("off")
    plt.imshow(figure, cmap=cm)
    plt.show()
    _save_plot_fig(solver, figure, cm=cm, name="fixed_z_all_labels")

# make a bar chart with x being preprocessing group (scale/rotate), y the number of occurences
# of each bin
def plot_prepro_params_distribution(solver, xticks, param, title):
    bins = list(zip(xticks[:-1], xticks[1:]))
    counts = np.zeros(len(bins))
    for theta in solver.prepro.prepro_params[param]:
        for bin_idx, (x, y) in enumerate(bins):
            if theta >= x and theta < y:
                counts[bin_idx] += 1
    plt.figure(figsize=(5, 4))
    paired_cmap = plt.cm.get_cmap("Paired", 12)
    rvb = mcolors.LinearSegmentedColormap.from_list("", paired_cmap.colors)
    xticks = xticks[:-1]
    norm = (xticks - np.min(xticks))/np.ptp(xticks)
    plt.bar(np.arange(0, len(counts)), counts, color=rvb(norm))
    plt.xlabel(param)
    plt.ylabel("Count")
    plt.xticks(np.arange(0, len(counts)), labels=bins, rotation=30)
    plt.title(title)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.25)
    plt.savefig(solver.data_loader.result_dir + "/plot_plot_prepro_params_distribution_" \
        + solver.data_loader.dataset + "_z=" + str(solver.z_dim) + ".png")
    plt.show()

# stacked bar graph, x being the theta groups, y the count, the colors the different classes.
def plot_prepro_params_distribution_categories(solver, xticks, param, title):
    bins = list(zip(xticks[:-1], xticks[1:]))
    classes_bins = np.zeros((len(bins), solver.data_loader.n_classes))
    for batch_idx, theta in enumerate(solver.prepro.prepro_params[param]): # theta=some degree from the list of theta_1
        start = batch_idx*solver.data_loader.batch_size
        end = (batch_idx+1)*solver.data_loader.batch_size
        for bin_idx, (x, y) in enumerate(bins):
            if theta >= x and theta < y:
                for label in solver.data_labels[start:end]:
                    classes_bins[bin_idx][int(label)] += 1 # if dict use (x,y) instead of bin_idx
    # preparation of chart
    plt.figure(figsize=(8, 8))
    snickers_bar = plt.bar
    width = 0.35
    categories = np.arange(solver.data_loader.n_classes)
    bottoms = np.cumsum(classes_bins, axis=0) # for correct shifting of bar
    # colouring here
    xticks = xticks[:-1]
    paired_cmap = plt.cm.get_cmap("Paired", 12)
    rvb = mcolors.LinearSegmentedColormap.from_list("", paired_cmap.colors)
    norm = (xticks - np.min(xticks))/np.ptp(xticks)
    for bin_idx in range(len(bins)):
        distr = classes_bins[bin_idx]
        if bin_idx == 0:
            bar = snickers_bar(categories, distr, width, color=rvb(norm[bin_idx]))
        else:
            bar = snickers_bar(categories, distr, width, color=rvb(norm[bin_idx]), bottom=bottoms[bin_idx-1])
    # labels, legends, ticks and save plot
    plt.xlabel("Labels")
    plt.ylabel("Number of elements in each bin")
    plt.title(title)
    plt.xticks(categories)
    maxy = np.max(np.sum(classes_bins, axis=0))
    yticks = np.arange(0, np.around(int(maxy), decimals=-3)+1, 500)
    plt.yticks(yticks)
    handles = []
    for bin_idx, bucket in enumerate(bins):
        handles.append(mpatches.Patch(color=rvb(norm[bin_idx]), label=bucket))
    plt.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.175),
            fancybox=True, shadow=True, ncol=6)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
    plt.savefig(solver.data_loader.result_dir + "/plot_prepro_params_distribution_categories_" \
            + solver.data_loader.dataset + "_z=" + str(solver.z_dim) + ".png")
    plt.show()

# takes only numpy array in, so mainly for testing puposes
def plot_faces_grid(n, n_cols, solver, fig_size=(10, 8)):
    img_rows, img_cols = solver.data_loader.img_dims
    n_rows = int(np.ceil(n / float(n_cols)))
    figure = np.zeros((img_rows * n_rows, img_cols * n_cols))
    for k, x in enumerate(solver.data_loader.data[:n]):
        r = k // n_cols
        c = k % n_cols
        figure[r*img_rows:(r+1)*img_rows,
               c*img_cols:(c+1)*img_cols] = x.reshape(list(solver.data_loader.img_dims))

    plt.figure(figsize=fig_size)
    plt.imshow(figure, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    _save_plot_fig(solver, figure, cm="gray", name="faces_grid")

# plot sampled faces in a grid
def plot_faces_samples_grid(n, n_cols, solver, fig_size=(10, 8)):
    img_rows, img_cols = solver.data_loader.img_dims
    n_rows = int(np.ceil(n / float(n_cols)))
    figure = np.zeros((img_rows * n_rows, img_cols * n_cols))
    samples = torch.randn(n, solver.z_dim).to(solver.device)
    solver.model.eval()
    with torch.no_grad():
        samples = solver.model.decoder(samples).cpu().detach().numpy()
        for k, x in enumerate(samples):
            r = k // n_cols
            c = k % n_cols
            figure[r*img_rows:(r+1)*img_rows,
                c*img_cols:(c+1)*img_cols] = x.reshape(list(solver.data_loader.img_dims))
    
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
