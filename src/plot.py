import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Plotting train and test losses
def plot_losses(train_loss_history, test_loss_history):
    #xaxis = np.arange(1, len(test_loss_history)+1)
    f = plt.figure(figsize=(12, 4))
    ax = f.add_subplot(121) # 211
    ax2 = f.add_subplot(122) # 212
    # Plotting the train loss
    #plt.subplot(2, 1, 1)
    # plt.plot(np.arange(1, len(train_loss_history)+1), train_loss_history, '-o')
    ax.plot(np.arange(1, len(train_loss_history)+1), train_loss_history, '-o')
    ax.set_title("train loss / marginal likelihood log p(x)")
    #plt.xlabel("epoch")
    #plt.ylabel("loss")
    #plt.title("train loss / marginal likelihood log p(x)")
    #plt.legend(['train loss'], loc='upper right')
    #plt.xticks(xaxis)

    # Plotting the test loss
    #plt.subplot(2, 1, 2)
    ax2.plot(np.arange(1, len(test_loss_history)+1), test_loss_history, '-o')
    ax2.set_title("test loss / marginal likelihood log p(x)")
    #plt.xlabel("epoch")
    #plt.ylabel("loss")
    #plt.title("test loss / marginal likelihood log p(x)")
    #plt.legend(['test loss'], loc='upper right')
    #plt.xticks(xaxis)

    #plt.tight_layout()
    plt.show()

# Plotting histogram of the latent space's distribution, given the computed \mu and \sigma
def plot_histogram(num_normal_plots, z_stats, z_dim):
    x = np.linspace(-5, 5, 5000)
    plot_cols = np.arange(1, num_normal_plots+1)
    print("test", len(z_stats))
    for idx, stats in enumerate(z_stats):
        epoch, mu, std, varmuz = stats
        print("epoch: {}, mu(z): {}, stddev(z): {}, var(z): {}, var(mu(z)): {}".format(\
                epoch, mu, std, np.power(std, 2), varmuz))
        y = (1 / (np.sqrt(2 * np.pi * np.power(std, 2)))) * \
                (np.power(np.e, -(np.power((x - mu), 2) / (2 * np.power(std, 2)))))
        plt.subplot(2, 1, plot_cols[idx])
        plt.plot(x, y)
        plt.title("epoch {}, dim(z)={}".format(epoch, z_dim))
        plt.xlabel("x")
        plt.ylabel("y")

        plt.tight_layout()
        plt.show()

def plot_rl_kl(epochs, rls, kls):
    xaxis = np.arange(1, epochs+1)
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(1, len(rls)+1), rls, '-o')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Reconstruction loss / likelihood lower bound of x")
    #plt.legend(['recon. loss'], loc='upper right')
    plt.xticks(xaxis)

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(1, len(kls)+1), kls, '-o')
    plt.xlabel("epoch")
    plt.ylabel("y")
    plt.title("KL divergence of q(z|x)||p(z)")
    #plt.legend(['KL div.'], loc='upper right')
    plt.xticks(xaxis)

    plt.tight_layout()
    plt.show()

def plot_latent_space(solver):
    labels = solver.labels.tolist()
    plt.figure(figsize=(8,6))
    plt.scatter(solver.latent_space[:, 0], solver.latent_space[:, 1], c=labels, cmap='brg') # edgecolors='black' cmap='hsv'
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar() # show color scale

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