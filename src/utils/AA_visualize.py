import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

def plot_AA(X: np.ndarray, C: np.ndarray, S: np.ndarray, K: int, loss: np.ndarray, labels: np.ndarray):
    """
    Params:
        N = Number of data points,
        D = Number of features,
        K = Number of components

        X: np.ndarray, shape (D, N)
            Data matrix
        C: np.ndarray, shape (N, K)
            Archetypes matrix
        S: np.ndarray, shape (K, N)
            Mixing matrix
        K: int
            Number of components
        loss: np.ndarray, shape (n_iter,)
            Loss curve
        labels: np.ndarray, shape (N,)
            Labels of the data points

    Returns:
        Plots the:
        1: Data points 
        2: C matrix
        3: Archetypes and convex hull
        4: S matrix
        5: Loss curve 
    """
    XC = X@C

    fig, axs = plt.subplots(1,5, figsize=(25,5))
    
    # Plot 1 - Synthetic data
    axs[0].scatter(X[0,:],X[1,:],c=labels)
    axs[0].set_title('Synthetic data')
    axs[0].set_xlabel('Feature 1')
    axs[0].set_ylabel('Feature 2')
    
    # Plot 2 - C (transposed) matrix
    for k in range(K):
        axs[1].plot(C[:,k]+k)
    axs[1].set_title('C (transposed)')
    # use K
    axs[1].set_yticks([k+1 for k in range(K)])
    axs[1].set_yticklabels([k+1 for k in range(K)])
    axs[1].set_ylabel('Component')
    axs[1].set_xlabel('Sample index')
    axs[1].set_yticklabels([k+1 for k in range(K)])
    axs[1].set_xlabel('Sample index')

    # Plot 3 - Archetypes and convex hull
    hull = ConvexHull(X.T)
    for simplex in hull.simplices:
        axs[2].plot(X[0,simplex],X[1,simplex],'k-', alpha=0.5)
    axs[2].scatter(X[0,:],X[1,:], c=labels)
    for k in range(K):
        axs[2].scatter(XC[0,k],XC[1,k], s=60, zorder=2, marker='D')
    legend_labels = ['Data', 'Convex Hull', 'Archetypes']
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='teal', markersize=10, label=legend_labels[0]), 
                      plt.Line2D([0], [0], color='k', lw=2, label=legend_labels[1]), 
                      plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='r', markersize=10, label=legend_labels[2])]
    axs[2].legend(legend_handles, legend_labels)
    axs[2].set_title('Archetypal Analysis on toy data')
    axs[2].set_xlabel('Feature 1')
    axs[2].set_ylabel('Feature 2')

    # Plot 4 - S matrix
    for k in range(K):
        axs[3].plot(S[k,:]+k)
    axs[3].set_title('S')
    axs[3].set_yticks([k+1 for k in range(K)])
    axs[3].set_ylabel('Component')

    # Plot 5 - Loss curve
    axs[4].plot(loss)
    axs[4].set_title('Train loss')
    axs[4].set_xlabel('Iteration')
    axs[4].set_ylabel('SSE')

    plt.tight_layout()
    plt.show()


def plot_AA_simple(X: np.ndarray, C: np.ndarray, S: np.ndarray, K: int):
    """
    Params:
        N = Number of data points,
        D = Number of features,
        K = Number of components

        X: np.ndarray, shape (D, N)
            Data matrix
        C: np.ndarray, shape (N, K)
            Archetypes matrix
        S: np.ndarray, shape (K, N)
            Mixing matrix
        K: int
            Number of components

    Returns:
        Plots the:
        1: C matrix
        2: Archetypes and convex hull
        3: S matrix
    """

    XC = X@C
    fig, axs = plt.subplots(1,3,figsize=(15,5),layout='constrained')

    # Plot 1 - C (transposed) matrix
    for k in range(K):
        axs[0].plot(C[:,k]+k)
    axs[0].set_title('C (transposed)')
    # use K
    axs[0].set_yticks([k+1 for k in range(K)])
    axs[0].set_yticklabels([k+1 for k in range(K)])
    axs[0].set_ylabel('Component')
    axs[0].set_xlabel('Sample index')
    axs[0].set_yticklabels([k+1 for k in range(K)])
    axs[0].set_xlabel('Sample index')

    # Plot 2 - Archetypes and convex hull
    axs[1].scatter(X[0,:],X[1,:], c='teal')
    hull = ConvexHull(X.T)
    for simplex in hull.simplices:
        axs[1].plot(X[0,simplex],X[1,simplex],'k-', alpha=0.5)
    for k in range(K):
        axs[1].scatter(XC[0,k],XC[1,k], s=60, zorder=2, marker='D')
    legend_labels = ['Data', 'Convex Hull', 'Archetypes']
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='teal', markersize=10, label=legend_labels[0]), 
                      plt.Line2D([0], [0], color='k', lw=2, label=legend_labels[1]), 
                      plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='r', markersize=10, label=legend_labels[2])]

    axs[1].legend(legend_handles, legend_labels)
    axs[1].set_title('Archetypal Analysis on toy data')
    axs[1].set_xlabel('Feature 1')
    axs[1].set_ylabel('Feature 2')

    # Plot 3 - S matrix
    for k in range(K):
        axs[2].plot(S[k,:]+k)
    axs[2].set_title('S')
    axs[2].set_yticks([k+1 for k in range(K)])
    axs[2].set_ylabel('Component')


def plot_AA_reconstructed(X: np.ndarray, C: np.ndarray, S: np.ndarray, K: int, y: np.ndarray, recon_order_points: int=10, point_size: int=10):
    """
    Params:
        N = Number of data points,
        D = Number of features,
        K = Number of components

        X: np.ndarray, shape (D, N)
            Data matrix
        C: np.ndarray, shape (N, K)
            Archetypes matrix
        S: np.ndarray, shape (K, N)
            Mixing matrix
        K: int
            Number of components
        y: np.ndarray, shape (N,)
            Labels of the data points
        recon_order_points: int
            Number of closest points to each archetype to consider for reconstruction
        point_size: int
            Size of the data points

    Returns:
        Plots the:
        1: Reconstructed data points
        2: Archetypes
    """

    # Contructing archetypes, evenly K spaced points on a circle
    xs, ys = np.cos(2*np.pi/K*np.arange(K)+np.pi/2)*0.4 + 0.5, np.sin(2*np.pi/K*np.arange(K)+np.pi/2)*0.4 + 0.5
    archetypes = np.stack((xs, ys))

    # Finding the closest archetypes to each data point
    distances = np.sqrt(np.sum((archetypes[:, :, np.newaxis] - (archetypes@S)[:, np.newaxis, :])**2, axis=0))
    closest_indices = np.argsort(distances, axis=1)[:, :recon_order_points]
    majority_labels = np.array([np.argmax(np.bincount(y[indices])) for indices in closest_indices])
    sorted_indices = np.argsort(majority_labels)
    sorted_archetypes = archetypes[:, sorted_indices]
    sorted_reconstruction = sorted_archetypes@S

    # Plotting
    plt.scatter(sorted_reconstruction[0, :], sorted_reconstruction[1, :], c=y, s=point_size)
    plt.scatter(sorted_archetypes[0, :], sorted_archetypes[1, :], c='r', s=40, zorder=2)
    plt.show()
    
    
from sklearn.decomposition import PCA
    
def plot_AA_reconstructed_angle(X: np.ndarray, C: np.ndarray, S: np.ndarray, K: int, y: np.ndarray, point_size: int=10):
    """
    Params:
        N = Number of data points,
        D = Number of features,
        K = Number of components

        X: np.ndarray, shape (D, N)
            Data matrix
        C: np.ndarray, shape (N, K)
            Archetypes matrix
        S: np.ndarray, shape (K, N)
            Mixing matrix
        K: int
            Number of components
        y: np.ndarray, shape (N,)
            Labels of the data points
        point_size: int
            Size of the data points

    Returns:
        Plots the:
        1: Reconstructed data points
        2: Archetypes
    """

    # Contructing archetypes, evenly K spaced points on a circle
    xs, ys = np.cos(2*np.pi/K*np.arange(K)+np.pi/2)*0.4 + 0.5, np.sin(2*np.pi/K*np.arange(K)+np.pi/2)*0.4 + 0.5
    archetypes = np.stack((xs, ys))
    
    pca = PCA(n_components=2)
    XCS = X @ C @ S
    XC = X @ C
    
    # Fit PCA on the data points
    pca.fit_transform(XCS.T)
    
    # Transform the archetypes
    XC_pca = pca.transform(XC.T)
    
    # Calculate the angles of the archetypes [0, 2*pi]
    angles = (np.arctan2(XC_pca[:,1],XC_pca[:,0]) + 2*np.pi) % (2*np.pi)
    
    # Sort the archetypes by angle
    sorted_indices = np.argsort(angles)
    
    final_indices = np.argsort(sorted_indices)

    sorted_archetypes = archetypes[:, final_indices]
    sorted_reconstruction = sorted_archetypes@S

    # Plotting
    plt.scatter(sorted_reconstruction[0, :], sorted_reconstruction[1, :], c=y, s=point_size)
    plt.scatter(sorted_archetypes[0, :], sorted_archetypes[1, :], c='r', s=40, zorder=2)
    for i in range(K):
        plt.text(sorted_archetypes[0, i], sorted_archetypes[1, i], str(i+1), fontsize=12, color='black')
    plt.show()




def plot_AA_reconstructed_angles_multiple(X: np.ndarray, Cs: list, Ss: list, Ks: list, y: np.ndarray, point_size: int=10):
    """
    Params:
        X: np.ndarray, shape (D, N)
            Data matrix
        Cs: list of np.ndarray, each shape (N, K_i)
            List of archetypes matrices
        Ss: list of np.ndarray, each shape (K_i, N)
            List of mixing matrices
        Ks: list of int
            List of numbers of components
        y: np.ndarray, shape (N,)
            Labels of the data points
        point_size: int
            Size of the data points
    """
    # Number of subplots
    num_plots = len(Ks)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    for idx, (C, S, K) in enumerate(zip(Cs, Ss, Ks)):
        # Archetypes placed on a circle
        xs, ys = np.cos(2 * np.pi / K * np.arange(K) + np.pi / 2) * 0.4 + 0.5, \
                 np.sin(2 * np.pi / K * np.arange(K) + np.pi / 2) * 0.4 + 0.5
        archetypes = np.stack((xs, ys))
        
        pca = PCA(n_components=2)
        XCS = X @ C @ S
        XC = X @ C
        
        # Fit PCA on the data points
        pca.fit_transform(XCS.T)
        
        # Transform the archetypes
        XC_pca = pca.transform(XC.T)
        
        # Calculate the angles of the archetypes [0, 2*pi]
        angles = (np.arctan2(XC_pca[:,1], XC_pca[:,0]) + 2 * np.pi) % (2 * np.pi)
        
        # Sort the archetypes by angle
        sorted_indices = np.argsort(angles)
        final_indices = np.argsort(sorted_indices)

        sorted_archetypes = archetypes[:, final_indices]
        sorted_reconstruction = sorted_archetypes @ S

        ax = axes if num_plots == 1 else axes[idx]

        ax.set_aspect('equal')
        ax.set_axis_off()
        
        # Plotting
        ax.scatter(sorted_reconstruction[0, :], sorted_reconstruction[1, :], c=y, s=point_size)
        ax.scatter(sorted_archetypes[0, :], sorted_archetypes[1, :], c='r', s=40, zorder=2)
        for i in range(K):
            ax.text(sorted_archetypes[0, i], sorted_archetypes[1, i], str(i+1), fontsize=12, color='black')
        ax.set_title(f"Plot for K={K}")

    plt.show()


from matplotlib.colors import ListedColormap, Normalize

def plot_AA_reconstructed_angles_multiple_sep(X: np.ndarray, Cs: list, Ss: list, Ks: list, y: np.ndarray, point_size: int=10):
    """
    Params:
        X: np.ndarray, shape (D, N)
            Data matrix
        Cs: list of np.ndarray, each shape (N, K_i)
            List of archetypes matrices
        Ss: list of np.ndarray, each shape (K_i, N)
            List of mixing matrices
        Ks: list of int
            List of numbers of components
        y: np.ndarray, shape (N,)
            Labels of the data points
        point_size: int
            Size of the data points
    """
    # Number of subplots
    num_plots = len(Ks)
    num_rows = 4  # All labels, then each label separately
    fig, axes = plt.subplots(num_rows, num_plots, figsize=(5 * num_plots, 5 * num_rows))

    unique_labels = np.unique(y)
    cmap = ListedColormap(plt.cm.get_cmap('viridis', len(unique_labels))(np.arange(len(unique_labels))))
    norm = Normalize(vmin=np.min(unique_labels), vmax=np.max(unique_labels))

    for idx, (C, S, K) in enumerate(zip(Cs, Ss, Ks)):
        for row in range(num_rows):
            xs, ys = np.cos(2 * np.pi / K * np.arange(K) + np.pi / 2) * 0.4 + 0.5, \
                     np.sin(2 * np.pi / K * np.arange(K) + np.pi / 2) * 0.4 + 0.5
            archetypes = np.stack((xs, ys))
            
            pca = PCA(n_components=2)
            XCS = X @ C @ S
            XC = X @ C
            
            pca.fit_transform(XCS.T)
            XC_pca = pca.transform(XC.T)
            angles = (np.arctan2(XC_pca[:,1], XC_pca[:,0]) + 2 * np.pi) % (2 * np.pi)
            sorted_indices = np.argsort(angles)
            final_indices = np.argsort(sorted_indices)

            sorted_archetypes = archetypes[:, final_indices]
            sorted_reconstruction = sorted_archetypes @ S

            ax = axes[row, idx]
            ax.set_aspect('equal')
            ax.set_axis_off()

            if row == 0:
                scatter = ax.scatter(sorted_reconstruction[0, :], sorted_reconstruction[1, :], c=y, cmap=cmap, norm=norm, s=point_size)
            else:
                mask = y == row
                ax.scatter(sorted_reconstruction[0, mask], sorted_reconstruction[1, mask], c=y[mask], cmap=cmap, norm=norm, s=point_size)
            
            ax.scatter(sorted_archetypes[0, :], sorted_archetypes[1, :], c='r', s=40, zorder=2)
            for k in range(K):
                ax.text(sorted_archetypes[0, k], sorted_archetypes[1, k], str(k+1), fontsize=12, color='black')
            ax.set_title(f"K={K} (Label {row if row > 0 else 'All'})")

    plt.tight_layout()
    plt.show()


def pca_plot_AA(X: np.ndarray, C_list: list, S_list: list, K_list: list, y: np.ndarray):
    """
    Params:
        X: np.ndarray, shape (D, N)
            Data matrix
        C_list: list of np.ndarray, each shape (N, K_i)
            List of archetypes matrices
        S_list: list of np.ndarray, each shape (K_i, N)
            List of mixing matrices
        K_list: list of int
            List of numbers of components
    Returns:
        Plots the:
        1: PCA plot of the data points colored by class
        2: PCA plot of the archetypes colored by angle
    """
    
    fig, axs = plt.subplots(3, 3, figsize=(12, 12), layout='constrained')
    
    unique_labels = np.unique(y)
    
    cmap = ListedColormap(plt.cm.get_cmap('viridis', len(unique_labels))(np.arange(len(unique_labels))))
    norm = Normalize(vmin=np.min(unique_labels), vmax=np.max(unique_labels))

    for i, K in enumerate(K_list):
        XC = X@C_list[i]

        ax = axs[(K-2)//3][(K-2)%3]
        ax.set_aspect('equal')
        ax.set_axis_off()

        pca = PCA(n_components=2)
        XCS = X @ C_list[i] @ S_list[i]
        
        # Fit PCA on the data points
        XCS_pca = pca.fit_transform(XCS.T)
        
        # Transform the archetypes
        XC_pca = pca.transform(XC.T)
        
        # Calculate the angles of the archetypes [0, 2*pi]
        angles = (np.arctan2(XC_pca[:,1],XC_pca[:,0]) + 2*np.pi) % (2*np.pi)
        
        # Plot on PCA, use y to cut the scatter
        point_size = 2
        ax.scatter(XCS_pca[y==1,0],XCS_pca[y==1,1],c=y[y==1],cmap=cmap,norm=norm,s=point_size)
        ax.scatter(XCS_pca[y==2,0],XCS_pca[y==2,1],c=y[y==2],cmap=cmap,norm=norm,s=point_size)
        ax.scatter(XCS_pca[y==3,0],XCS_pca[y==3,1],c=y[y==3],cmap=cmap,norm=norm,s=point_size)
        
        # plot XC
        ax.scatter(XC_pca[:,0],XC_pca[:,1],c='r',label='Archetypes',s=40,zorder=2)
        angle_idx = np.argsort(angles)
        angles_sorted = angles[angle_idx]
        for i, angle in enumerate(angles_sorted):
            x_end = np.cos(angle) * 0.3
            y_end = np.sin(angle) * 0.3
            ax.plot([0, x_end], [0, y_end], label=f'Angle: {np.degrees(angle):.0f}°', color='black')
            # show angle idx at point
            XC_pca_sorted = XC_pca[angle_idx]
            ax.text(XC_pca_sorted[i,0], XC_pca_sorted[i,1], str(i+1), fontsize=12)
            
        ax.set_title(f'K={K}')

    plt.tight_layout()
    plt.show()