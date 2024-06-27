
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize


def plot_archetypes_features(X, C_list, K):
    """
    Plot the archetypes of the substage.

    Parameters
    ----------
    X : np.ndarray
        The data matrix.
    C_list : list of np.ndarray
        The list of archetypes for each substage.
    K : int
        The number of archetypes.

    Returns
    -------
    plot of substage archetypes
    """

    features = ['slowdelta', 'fastdelta', 'slowtheta', 'fasttheta', 'alpha', 'beta', 'logrms']

    plt.figure(figsize=(3*K, 3))
    for i in range(K):
        plt.subplot(1, K, i+1)
        XC = X @ C_list[K-2][:, i]
        plt.bar(range(7), XC)
        plt.title('Component {}'.format(i+1))
        plt.xlabel('Feature')
        plt.ylabel('Weight')
        plt.xticks(range(7), features)
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()



def plot_single_K_AA(X, C_list, S_list, y, K):
    """
    Plot the archetypes of the substage.

    Parameters
    ----------
    X : np.ndarray
        The data matrix.
    C_list : list of np.ndarray
        The list of archetypes for each substage.
    S_list : list of np.ndarray
        The list of coefficients for each substage.
    y : np.ndarray
        The labels of the data.
    K : int
        The number of archetypes.

    Returns
    -------
    AA plot of a single K split by labels
    """

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    num_colums = 4  # All labels, then each label separately

    unique_labels = np.unique(y)
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=np.min(unique_labels), vmax=np.max(unique_labels))

    point_size = 0.01
    alpha = 0.5

    for column in range(num_colums):
        xs, ys = np.cos(2 * np.pi / K * np.arange(K) + np.pi / 2) * 0.4 + 0.5, \
                    np.sin(2 * np.pi / K * np.arange(K) + np.pi / 2) * 0.4 + 0.5
        archetypes = np.stack((xs, ys))

        pca = PCA(n_components=2)
        XCS = X @ C_list[K-2] @ S_list[K-2]
        XC = X @ C_list[K-2]

        pca.fit_transform(XCS.T)
        XC_pca = pca.transform(XC.T)
        angles = (np.arctan2(XC_pca[:,1], XC_pca[:,0]) + 2 * np.pi) % (2 * np.pi)
        sorted_indices = np.argsort(angles)
        final_indices = np.argsort(sorted_indices)

        sorted_archetypes = archetypes[:, final_indices]
        sorted_reconstruction = sorted_archetypes @ S_list[K-2]

        ax[column].set_aspect('equal')
        ax[column].set_axis_off()

        if column == 0:
            ax[column].scatter(sorted_reconstruction[0, :], sorted_reconstruction[1, :], c=y, cmap=cmap, norm=norm, s=point_size, alpha=alpha)
        else:
            mask = y == column
            ax[column].scatter(sorted_reconstruction[0, mask], sorted_reconstruction[1, mask], c=y[mask], cmap=cmap, norm=norm, s=point_size, alpha=alpha)

        ax[column].scatter(sorted_archetypes[0, :], sorted_archetypes[1, :], c='r', s=40, zorder=2)
        for k in range(K):
            ax[column].text(sorted_archetypes[0, k], sorted_archetypes[1, k], str(k+1), fontsize=12, color='black')
        ax[column].set_title(f"K={K} (Label {column if column > 0 else 'All'})")




def plot_archetype_features_and_AA(X, C_list, S_list, y, K):
    """
    Plot the archetypes of the substage.

    Parameters
    ----------
    X : np.ndarray
        The data matrix.
    C_list : list of np.ndarray
        The list of archetypes for each substage.
    S_list : list of np.ndarray
        The list of coefficients for each substage.
    y : np.ndarray
        The labels of the data.
    K : int

    Returns
    -------
    plot of substage archetypes and AA plot
    """

    features = ['slowdelta', 'fastdelta', 'slowtheta', 'fasttheta', 'alpha', 'beta', 'logrms']
    columns = 2
    # Create a main figure and axes
    fig, ax = plt.subplots(1, columns, figsize=(5*columns, 5))
    unique_labels = np.unique(y)
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=np.min(unique_labels), vmax=np.max(unique_labels))

    # Calculate archetypes and their positions
    xs, ys = np.cos(2 * np.pi / K * np.arange(K) + np.pi / 2) * 0.4 + 0.5, \
                np.sin(2 * np.pi / K * np.arange(K) + np.pi / 2) * 0.4 + 0.5
    archetypes = np.stack((xs, ys))

    # Perform PCA on the component space
    pca = PCA(n_components=2)
    XCS = X @ C_list[K-2] @ S_list[K-2]
    XC = X @ C_list[K-2]
    pca.fit_transform(XCS.T)
    XC_pca = pca.transform(XC.T)
    angles = (np.arctan2(XC_pca[:,1], XC_pca[:,0]) + 2 * np.pi) % (2 * np.pi)
    sorted_indices = np.argsort(angles)
    final_indices = np.argsort(sorted_indices)
    sorted_archetypes = archetypes[:, final_indices]
    sorted_reconstruction = sorted_archetypes @ S_list[K-2]

    for column in range(columns):
        ax[column].set_aspect('equal')
        ax[column].set_axis_off()

        if column == 0:
            # Add bar plots for each archetype
            for i in range(K):
                # Convert archetype locations to plot coordinates
                ax_coord_x = sorted_archetypes[0, i]
                ax_coord_y = sorted_archetypes[1, i]

                # Adjust inset position and size dynamically based on plot coordinates
                inset_width = 0.3  # Inset width
                inset_height = 0.3  # Inset height
                inset_x = ax_coord_x - inset_width / 2  # Center inset on archetype
                inset_y = ax_coord_y - inset_height / 2  # Center inset on archetype

                inset_ax = ax[column].inset_axes([inset_x, inset_y, inset_width, inset_height])
                inset_ax.bar(range(7), X @ C_list[K-2][:, i])
                inset_ax.set_xticks(range(7))
                inset_ax.set_xticklabels(features, fontsize=8, rotation=45, ha='right', rotation_mode='anchor')
                inset_ax.tick_params(axis='y', labelsize=8)

        if column == 1:
            # Scatter plot for all labels
            ax[column].scatter(sorted_reconstruction[0, :], sorted_reconstruction[1, :], c=y, cmap=cmap, norm=norm, s=0.005, alpha=0.5)
            
        else:
            # Scatter plot for individual labels
            mask = y == column - 1
            ax[column].scatter(sorted_reconstruction[0, mask], sorted_reconstruction[1, mask], c=y[mask], cmap=cmap, norm=norm, s=0.005, alpha=0.5)

        # Archetypes and their labels
        if column != 0:
            ax[column].scatter(sorted_archetypes[0, :], sorted_archetypes[1, :], c='red', s=40, zorder=2)
            for k in range(K):
                ax[column].text(sorted_archetypes[0, k], sorted_archetypes[1, k], str(k+1), fontsize=16, color='black', zorder=3)
            ax[column].set_title(f"K={K} (Label {column if column > 1 else 'All'})")

    plt.tight_layout()