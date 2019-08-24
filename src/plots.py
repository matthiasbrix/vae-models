import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import scipy.stats as stats
import scipy.spatial.distance as dist
import skimage as ski
from preprocessing import preprocess_sample

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "mnist": "MNIST",
    "ff": "Frey Faces",
    "lfw": "Labeled Faces in the Wild",
    "lungscans": "Lung Scans"
}

def _xticks(ls, ticks_rate):
    labels = np.arange(1, len(ls)+2, (len(ls)//ticks_rate))
    labels[1:] -= 1
    labels[-1] = len(ls)
    return labels.astype(int)

# Plotting train and test losses
def plot_losses(solver, train_loss_history, test_loss_history):
    if len(train_loss_history) == 1 or len(test_loss_history) == 1:
        print("No plots, just 1 epoch. Need > 1.")
        return
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, len(train_loss_history)+1), train_loss_history, label="Train")
    plt.plot(np.arange(1, len(test_loss_history)+1), test_loss_history, label="Test")
    ticks_rate = 4 if len(train_loss_history) >= 4 else len(train_loss_history)
    plt.xticks(_xticks(train_loss_history, ticks_rate))
    plt.title("Loss on data set {}, dim(z)={}".format(DATASETS[solver.data_loader.dataset.lower()], solver.model.z_dim))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4),
            fancybox=True, shadow=True, ncol=5)
    plt.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.25)
    plt.grid(True, linestyle='-.')
    if solver.data_loader.directories.make_dirs:
        plt.savefig(solver.data_loader.directories.result_dir + "/" + "plot_losses_" +\
            solver.data_loader.dataset + "_z=" + str(solver.model.z_dim) + ".png")

# Plotting histogram of the latent space's distribution, given the computed \mu and \sigma
def plot_gaussian_distributions(solver, epochs):
    x = np.linspace(-5, 5, 5000)
    idx_x = 0
    idx_y = 0
    plots = np.arange(1, epochs+1, np.ceil(epochs/4)).astype(int)
    plots[-1] = epochs
    if epochs == 1:
        f, axarr = plt.subplots(1, 1, figsize=(8, 2))
        axarr = [axarr]
    if epochs == 2:
        f, axarr = plt.subplots(1, 2, figsize=(8, 4))
    if epochs == 3:
        f, axarr = plt.subplots(2, 2, figsize=(8, 6))
        axarr[-1, -1].axis('off') # turning off the 4th subplot
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
        if epochs <= 2:
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

    axarr = axarr.flatten() if epochs > 2 else axarr

    for ax in axarr:
        maxys = max(ys)
        maxnorm = max(stats.norm.pdf(x, 0, 1))
        ax.set_ylim([0, max(maxys, maxnorm)+0.05])
        ax.set(xlabel='x', ylabel='y')
        ax.grid(True, linestyle='-.')

    # writing stats results of z to file
    if solver.data_loader.directories.make_dirs:
        with open(solver.data_loader.directories.result_dir + "/result_stats_" +\
            solver.data_loader.dataset + "_z=" + str(solver.model.z_dim) + ".txt", 'w') as file_res:
            file_res.write("epoch,var(mu(z)),E[var(q(z|x))]\n")
            for idx in plots:
                i = idx-1
                epoch, varmu_z, expected_var_z = solver.train_loss_history["epochs"][i],\
                solver.z_stats_history["varmu_z"][i], solver.z_stats_history["expected_var_z"][i]
                file_res.write(str(epoch) + ","\
                    + str(np.around(np.array(varmu_z), 4)) + ","\
                    + str(np.around(np.array(expected_var_z), 4)))
                file_res.write("\n")

    if epochs >= 3:
        ax = axarr.flatten()[2]
    f.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.2)
    ax.legend(loc='upper center', bbox_to_anchor=(1.2, -0.25),
            fancybox=True, shadow=True, ncol=5)
    if solver.data_loader.directories.make_dirs:
        plt.savefig(solver.data_loader.directories.result_dir + "/" + "plot_gaussian_" +\
            solver.data_loader.dataset + "_z=" + str(solver.model.z_dim) + ".png")

# Plot the reconstruction loss and KL divergence in two separate plots
def plot_rl_kl(solver, rls, kls):
    if len(rls) == 1 or len(rls) == 1:
        print("No plots, just 1 epoch. Need > 1.")
        return
    x = np.arange(1, len(kls)+1)
    plt.figure(figsize=(4.5, 5))

    ticks_rate = 4 if len(rls) >= 4 else len(rls)
    plt.subplot(2, 1, 1)
    plt.plot(x, rls)
    plt.xticks(_xticks(rls, ticks_rate))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Reconstruction loss in (training)") # marginal log likelihood
    plt.grid(True, linestyle='-.')

    plt.subplot(2, 1, 2)
    plt.plot(x, kls) # KL div
    plt.xticks(_xticks(kls, ticks_rate))
    plt.xlabel("epoch")
    plt.ylabel("KL divergence")
    plt.title("KL divergence of q(z|x)||p(z), β={} (training)".format(solver.model.beta))

    plt.grid(True, linestyle='-.')
    plt.tight_layout()
    if solver.data_loader.directories.make_dirs:
        plt.savefig(solver.data_loader.directories.result_dir + "/" + "plot_rl_kl_"\
            + solver.data_loader.dataset + "_z=" + str(solver.model.z_dim) + ".png")

# Plot the latent space as scatter plot with and without labels
def plot_latent_space(solver, space, ticks=None, var=None, title=None, labels=None, colors=None, xticks=None, yticks=None, invert_y_axis=False):
    plt.figure(figsize=(9, 7))
    if labels is not None and title:
        if var == "z" and ticks is not None:
            scatter = plt.scatter(space[:, 0], space[:, 1], s=10, vmin=ticks[0], vmax=ticks[-1], c=labels, cmap=plt.cm.get_cmap("Paired", colors))
            clb = plt.colorbar(scatter, ticks=ticks)
            clb.ax.set_title(title)
        elif var == "y" and ticks is not None:
            scatter = plt.scatter(space[:, 0], space[:, 1], s=10, vmin=ticks[0], vmax=ticks[-1], c=labels, cmap="Paired")
            clb = plt.colorbar(scatter, ticks=ticks)
            clb.ax.set_title(title)
        elif ticks is not None:
            scatter = plt.scatter(space[:, 0], space[:, 1], s=10, c=labels, cmap="Paired")
            clb = plt.colorbar(scatter, ticks=ticks)
            clb.ax.set_title(title)
        else:
            plt.scatter(space[:, 0], space[:, 1], s=10, c=labels, cmap=plt.cm.get_cmap("Paired", solver.data_loader.n_classes))
            clb = plt.colorbar()
            clb.ax.set_title(title)
    else:
        plt.scatter(space[:, 0], space[:, 1], s=10, cmap="Paired")
    plt.xlabel("{}_1".format(var))
    plt.ylabel("{}_2".format(var))

    leftx, rightx = plt.xlim()
    lefty, righty = plt.ylim()
    min1, max1 = xticks[0], xticks[-1]
    min2, max2 = yticks[0], yticks[-1]
    x1 = min(min1, leftx)
    x2 = max(max1, rightx)
    y1 = min(min2, lefty)
    y2 = max(max2, righty)
    xs = np.linspace(x1, x2, len(xticks))
    ys = np.linspace(y1, y2, len(xticks))
    if xticks is not None:
        plt.xticks(xs)
    if yticks is not None:
        plt.yticks(ys)
    if invert_y_axis:
        plt.ylim(plt.ylim()[::-1])

    plt.title("Latent space q({}) on data set {}".format(var, DATASETS[solver.data_loader.dataset.lower()]))
    if solver.data_loader.directories.make_dirs:
        plt.savefig(solver.data_loader.directories.result_dir + "/plot_" + str(var) + "_space_" \
            + title + "_" + solver.data_loader.dataset + "_z=" + str(solver.model.z_dim) + ".png")

# For each of the values z, we plotted the corresponding generative
# p(x|z) with the learned parameters θ.
def plot_latent_manifold(decoder, solver, cm, grid_x, grid_y, n=20, fig_size=(10, 10), x_t=None, label=None):
    c, h, w = solver.data_loader.img_dims
    figure = torch.zeros((c, h*n, w*n))
    # Decode for each square in the grid.
    solver.model.eval()
    with torch.no_grad():
        for i, xi in enumerate(grid_x):
            for j, yj in enumerate(grid_y):
                z_sample = np.array([xi, yj])
                z_sample = np.tile(z_sample, 1).reshape(1, solver.model.z_dim)
                z_sample = torch.from_numpy(z_sample).float().to(DEVICE) # transform to tensor
                if solver.cvae_mode:
                    idx = torch.randint(0, solver.data_loader.n_classes, (1,)).item()
                    y_sample = torch.FloatTensor(torch.zeros(z_sample.size(0), solver.data_loader.n_classes)).to(DEVICE)
                    y_sample[:, idx] = 1.
                    sample = torch.cat((z_sample, y_sample), dim=-1)
                elif solver.tdcvae_mode:
                    x_t = x_t.to(DEVICE)
                    sample = torch.cat((x_t, z_sample), dim=-1)
                else:
                    sample = z_sample
                x_decoded = decoder(sample)
                figure[:, i*h:(i+1)*h, j*w:(j+1)*w] = x_decoded.view(*solver.data_loader.img_dims)
    grid_img = torchvision.utils.make_grid(figure)
    plt.figure(figsize=fig_size)
    plt.axis("off")
    plt.imshow(grid_img.permute(1, 2, 0), cmap=cm)
    plt.show()
    if solver.data_loader.directories.make_dirs:
        # save stats of the grid (x,y ranges as defined in the notebook)
        with open(solver.data_loader.directories.result_dir + "/plot_learned_data_manifold_grids_" +\
            solver.data_loader.dataset + "_z=" + str(solver.model.z_dim) + ".txt", 'w') as file_res:
            file_res.write("grid_x: {}\n".format(grid_x))
            file_res.write("grid_y: {}\n".format(grid_y))
        torchvision.utils.save_image(figure, solver.data_loader.directories.result_dir +\
            "/plot_learned_data_manifold_" + str(label) + "_" + solver.data_loader.dataset + "_z=" + str(solver.model.z_dim)+".png")

# Replicating the handstyle image example from Kingma et. al in Semisupervised VAE paper
# Take a single test set image (first from each batch), encode it, use that fixed z,
# loop over all labels and print a row out with the fixed z but different labeled images
def plot_with_fixed_z(solver, cm, fig_size=(6, 6)):
    c, h, w = solver.data_loader.img_dims
    n_rows, n_cols = solver.data_loader.n_classes, solver.data_loader.n_classes+1
    figure = torch.zeros((c, h*n_rows, w*n_cols))
    solver.model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(solver.data_loader.test_loader):
            if i == n_rows:
                break
            # encode a single image from the test set
            x, y = data.to(DEVICE)[0], target.to(DEVICE)[0]
            y = y.view(1, 1)
            onehot = solver.model.onehot_encoding(y)
            x = x.view(-1, solver.data_loader.input_dim)
            x_new = torch.cat((x, onehot), dim=-1)
            mu_x, logvar_x = solver.model.encoder(x_new)
            z = solver.model.reparameterization_trick(mu_x, logvar_x)
            figure[:, i*h:(i+1)*h, 0:w] = x.view(*solver.data_loader.img_dims) # the test image in leftmost column
            # decode an image for each class (looping over the columns basically)
            for label in range(solver.data_loader.n_classes):
                onehot = torch.FloatTensor(torch.zeros(y.size(0), solver.model.y_size)).to(DEVICE)
                onehot.zero_()
                onehot[:, label] = 1.
                z_new = torch.cat((z, onehot), dim=-1).to(DEVICE)
                decoded = solver.model.decoder(z_new).view(*solver.data_loader.img_dims)
                figure[:, i*h:(i+1)*h, (label+1)*w: (label+2)*w] = decoded
    grid_img = torchvision.utils.make_grid(figure)
    plt.figure(figsize=fig_size)
    plt.axis("off")
    plt.imshow(grid_img.permute(1, 2, 0), cmap=cm)
    plt.show()
    if solver.data_loader.directories.make_dirs:
        torchvision.utils.save_image(figure, solver.data_loader.directories.result_dir +\
            "/plot_fixed_z_all_labels_" + solver.data_loader.dataset + "_z=" + str(solver.model.z_dim)+".png")

# takes only numpy array in, so mainly for testing puposes
def plot_faces_grid(n, n_cols, solver, fig_size=(10, 8)):
    c, h, w = solver.data_loader.img_dims
    n_rows = int(np.ceil(n / float(n_cols)))
    figure = torch.zeros((c, h*n_rows, w*n_cols))
    data = torch.zeros(n, *solver.data_loader.img_dims)
    remain = n
    offset = 0
    # filling out the tensor with data from the batch
    for _, batch in enumerate(solver.data_loader.train_loader):
        batch = batch[0] if solver.data_loader.with_labels else batch
        if remain is not 0:
            extract = min(batch.shape[0], remain)
            data[offset:(offset+extract)] = batch[:extract]
            remain -= extract
            offset += extract
    # iterate over the batch and insert to figure tensor
    for k, x in enumerate(data):
        row = k // n_cols
        col = k % n_cols
        figure[:, row*h:(row+1)*h, col*w:(col+1)*w] =\
            x.view(*solver.data_loader.img_dims)
    grid_img = torchvision.utils.make_grid(figure)
    plt.figure(figsize=fig_size)
    plt.axis("off")
    plt.imshow(grid_img.permute(1, 2, 0), cmap="gray")
    plt.show()
    if solver.data_loader.directories.make_dirs:
        torchvision.utils.save_image(figure, solver.data_loader.directories.result_dir +\
            "/plot_faces_grid_" + solver.data_loader.dataset + "_z=" + str(solver.model.z_dim)+".png")

# plot sampled faces in a grid and saves to file
def plot_faces_samples_grid(n, n_cols, solver, fig_size=(10, 8)):
    c, h, w = solver.data_loader.img_dims
    n_rows = int(np.ceil(n/float(n_cols)))
    figure = torch.zeros((c, h*n_rows, w*n_cols))
    samples = torch.randn(n, solver.model.z_dim).to(DEVICE)
    solver.model.eval()
    with torch.no_grad():
        # decode the n samples and iterate over them and insert to figure tensor
        samples = solver.model.decoder(samples)
        for k, x in enumerate(samples):
            row = k // n_cols
            col = k % n_cols
            figure[:, row*h:(row+1)*h, col*w:(col+1)*w] =\
                x.view(*solver.data_loader.img_dims)
    grid_img = torchvision.utils.make_grid(figure)
    plt.figure(figsize=fig_size)
    plt.axis("off")
    plt.imshow(grid_img.permute(1, 2, 0), cmap="gray")
    plt.show()
    if solver.data_loader.directories.make_dirs:
        torchvision.utils.save_image(figure, solver.data_loader.directories.result_dir +\
            "/plot_faces_samples_grid_" + solver.data_loader.dataset + "_z=" + str(solver.model.z_dim)+".png")

def plot_transformed_images(test_loader, batch_size, num_samples=25, nrows=5, theta=None, scale=None,\
    save_plot=False, file_name=None):
    num_samples = min(num_samples, batch_size)
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            data, _ = data
            data = data[:num_samples]
            _, _, H, W = data.shape
            transformed_data = np.zeros((num_samples, H, W))
            for i in range(num_samples):
                transformed_data[i] = preprocess_sample(data[i], theta=theta, scale=(scale, scale)).cpu().numpy()
            transformed_data_tensor = torch.FloatTensor(transformed_data)
            transformed_data_tensor.unsqueeze_(1)
            grid_img = torchvision.utils.make_grid(transformed_data_tensor, nrow=nrows)
            plt.axis("off")
            plt.imshow(grid_img.permute(1, 2, 0))
            break
    if save_plot and file_name is not None:
        torchvision.utils.save_image(grid_img, file_name)

def plot_y_space_thetas(ys, ticks, labels, save_plot, file_name, dataset):
    N, S, T, _ = ys.shape
    plt.figure(figsize=(20, 10))
    for t in range(T):
        labels2 = np.repeat(labels[t], S*N)
        scatter = plt.scatter(ys[:, :, t, 0].flatten(), ys[:, :, t, 1].flatten(),\
            vmin=ticks[0], vmax=ticks[-1], c=labels2, cmap="Paired")
    clb = plt.colorbar(scatter, ticks=ticks)
    clb.ax.set_title("theta")
    plt.xlabel("y_1")
    plt.ylabel("y_2")
    plt.title("Latent space q(y) on data set {} with fixed thetas".format(DATASETS[dataset.lower()]))
    if save_plot and file_name:
        plt.savefig(file_name)

def plot_y_space_scales(ys, ticks, labels, save_plot, file_name, dataset):
    N, S, T, _ = ys.shape
    plt.figure(figsize=(20, 10))
    for s in range(S):
        labels2 = np.repeat(labels[s], T*N)
        scatter = plt.scatter(ys[:, s, :, 0].flatten(), ys[:, s, :, 1].flatten(),\
            vmin=ticks[0], vmax=ticks[-1], c=labels2, cmap="Paired")
    clb = plt.colorbar(scatter, ticks=ticks)
    clb.ax.set_title("scale")
    plt.xlabel("y_1")
    plt.ylabel("y_2")
    plt.title("Latent space q(y) on data set {} with fixed scales".format(DATASETS[dataset.lower()]))
    if save_plot and file_name:
        plt.savefig(file_name)

# Plot of each classes (theta, alpha)
def plot_prepro_alpha_params_distribution(ys, thetas, save_plot, file_name, dataset):
    N, S, T, _ = ys.shape
    # compute alphas
    ysflat = np.reshape(ys, (N * S * T, 2))
    y_mean = np.mean(ysflat, 0, keepdims=True)
    alphas = np.arctan2(ysflat[:, 1] - y_mean[:, 1], ysflat[:, 0] - y_mean[:, 0])
    alphas = np.reshape(alphas, (N * S, T))
    # normalize alpha[0] = theta[0] = 0
    alphas -= alphas[:, 0:1]
    alphas = alphas/(2*np.pi) * 360 # convert from radian to euclidean angles
    alphas = np.where(alphas >= 0, alphas, alphas + 360) # (600, 30) dimension

    # create alpha-theta plot
    thetas_scatter = np.tile(np.expand_dims(thetas, axis=0), (alphas.shape[0], 1))/(2*np.pi) * 360
    plt.figure(figsize=(20, 10))
    plt.scatter(alphas.flatten(), thetas_scatter.flatten())
    maxi = max(np.max(alphas), 360)+1
    plt.xticks(np.arange(np.min(alphas), maxi, 30))
    plt.yticks(np.arange(np.min(thetas), maxi, 30))
    plt.xlabel("alphas")
    plt.ylabel("thetas")
    plt.title("Distribution of thetas/alphas data set {}".format(DATASETS[dataset.lower()]))
    if save_plot and file_name:
        plt.savefig(file_name)

# Plot for each class with (scale, radius) relation
def plot_prepro_radius_params_distribution(ys, scales, save_plot, file_name, dataset):
    N, S, T, _ = ys.shape
    ysflat = np.reshape(ys, (N*S*T, 2))
    # compute mean column wise
    y_mean = np.mean(ysflat, 0, keepdims=True)
    rs = dist.cdist(ysflat, y_mean).ravel()
    rs = np.reshape(rs, (N, S, T))
    # normalize
    rs /= rs[:, (S - 1):S, :] - rs[:, 0:1, :]
    rs += 1 - rs[:, 0:1, :]

    # create radius/scale plot
    scales_scatter = scales[np.newaxis, ..., np.newaxis]
    scales_scatter = np.tile(scales_scatter, (N, 1, T))
    plt.figure(figsize=(20, 10))
    plt.scatter(scales_scatter.flatten(), rs.flatten())
    plt.xlabel("scales")
    plt.ylabel("radiuses")
    plt.title("Distribution of scales/radius data set {}".format(DATASETS[dataset.lower()]))
    if save_plot and file_name:
        plt.savefig(file_name)

def plot_vis_tensor(tensor, nrow=8, padding=1, allkernels=False):
    N, C, W, H = tensor.shape
    if allkernels: tensor = tensor.view(N*C, -1, W, H)
    else: tensor = tensor[0,:,:,:].unsqueeze(dim=1)
    rows = np.min((tensor.shape[0]//nrow + 1, 64))
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow,rows))
    plt.axis('off')
    plt.ioff()
    plt.imshow(grid.detach().cpu().numpy().transpose((1, 2, 0)))
    print()

def plot_lungscans_samples_grid(solver, decoder, x_t, x_next, z_dim, n=4, fig_size=(10, 8)):
    num_samples = min(n, x_t.size(0))
    x_t = x_t[:num_samples]
    z_sample = torch.randn(x_t.shape[0], z_dim, x_t.shape[2], x_t.shape[3]).to(DEVICE)
    xz_t = torch.cat((x_t, z_sample), dim=1).to(DEVICE)
    decoder.eval()
    with torch.no_grad():
        x_decoded = decoder(xz_t)
    recon_diff = x_decoded[:num_samples] - x_next[:num_samples]
    print("Sampling norm(sample, ground truth): {}".format(torch.norm(recon_diff)))
    recon_diff = recon_diff.pow(2)
    rd2 = recon_diff * 100
    output = torch.cat([x_t[:num_samples],\
                x_next[:num_samples],
                x_decoded[:num_samples],
                rd2])
    grid_img = torchvision.utils.make_grid(output, nrow=n)
    plt.figure(figsize=fig_size)
    plt.axis("off")
    plt.imshow(grid_img.permute(1,2,0), cmap="gray")
    plt.show()
    if solver.data_loader.directories.make_dirs:
        torchvision.utils.save_image(grid_img, solver.data_loader.directories.result_dir +\
            "/plot_samples_grid_" + solver.data_loader.dataset + "_z=" + str(solver.model.z_dim)+".png", nrow=4)