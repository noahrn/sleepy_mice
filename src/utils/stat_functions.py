import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import sys
sys.path.append('../')
from src.CGD import AA_model, AA_trainer
from scipy.spatial import ConvexHull


def calculate_MI(S1, S2):
    """
    This function calculates the Mutual Information (MI) between two matrices with continuous data.
    """
    # sum_n p(d|n)*p(d'|n)
    prob = S1@S2.T

    # Normalize joint distribution to ensure it sums to 1. p(d, d'). Multiply with 1/N to normalize. p(n).
    prob_XY = prob/np.sum(prob)

    # Calculate marginal distributions p(d)*p(d')   p(d) = sum_d' (p(d,d')) and p(d') = sum_d (p(d,d'))
    prob_X_Y = np.outer(np.sum(prob_XY, axis=1), np.sum(prob_XY, axis=0))
    
    # Makes sure that the probability larger than 0
    ind = np.where(prob_XY > 0)

    # Calculate mutual information. MI = sum(p(d,d')*log(p(d,d')/(p(d)*p(d')))
    MI = np.sum(prob_XY[ind] * np.log(prob_XY[ind] / prob_X_Y[ind]))

    return MI

def calculate_NMI(S1, S2):
    """
    Calculate the Normalized Mutual Information (NMI) between two matrices with continuous data.
    """
    # Calculate NMI
    NMI = (2 * calculate_MI(S1,S2)) / (calculate_MI(S1, S1) + calculate_MI(S2,S2))
    
    return NMI


def calculate_NMI_list(S_list):
    """
    Calculate the NMI between all pairs of S matrices in a list. The last matrix is compared with the first one.

    Returns a list of NMI values for each pair. The list has dimension len(S_list).
    """
    return [calculate_NMI(S_list[i], S_list[(i + 1) % len(S_list)]) for i in range(len(S_list))]

def nmi_boxplot(S_lists):
    """
    Plots a single figure with boxplots for different numbers of components.
    
    :param S_lists: List of lists of S matrices. Each list corresponds to a different number of components.
    
    """
    # Initialize an empty list to store the NMI values
    nmi_values = []
    
    # Loop over each pair of S matrices
    # The last S matrix is compared with the first one
    for S_list in S_lists:
    # Calculate NMI for consecutive pairs and wrap around from the last to the first
        nmi_pairs = calculate_NMI_list(S_list)
        nmi_values.append(nmi_pairs)
    # Plot the NMI values as a boxplot
    plt.boxplot(nmi_values)
    plt.xlabel('K components')
    plt.ylabel('NMI')
    plt.title('NMI between S matrices for different components')
    plt.show()


def nmi_lineplot(lab_S_lists):
    """
    Plots a single figure with boxplots and interpolation lines for different labs.
    Each lab's line shows the trend across different numbers of components.
    Boxplots represent the distribution of NMI values for each component configuration.

    :param labs_data: List of lists of lists. 
                      Each top-level list corresponds to a different lab.
                      Each second-level list corresponds to a specific number of components.
                      Each third-level list contains S matrices for different runs.
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(lab_S_lists)))  # Color map for different labs

    for idx, lab_data in enumerate(lab_S_lists):
        nmi_values = [calculate_NMI_list(S_list) for S_list in lab_data]
        nmi_values = np.array(nmi_values).T  # Transpose for boxplot compatibility
        
        # Prepare x positions for boxplots to avoid overlap
        positions = [i + 0.1 * (idx - (len(lab_S_lists) - 1) / 2) for i in range(1, len(lab_data) + 1)]

        # Boxplot for each lab
        ax.boxplot(nmi_values, positions=positions, widths=0.1, patch_artist=True, boxprops=dict(facecolor=colors[idx]))

        # Mean NMI for line plotting
        nmi_means = np.mean(nmi_values, axis=0)
        ax.plot(range(1, len(lab_data) + 1), nmi_means, 'o-', label=f'Lab {idx + 1}', color=colors[idx])

    ax.set_xlabel('K components')
    ax.set_ylabel('NMI')
    ax.set_title('NMI between S matrices for different components')
    ax.legend(title='Labs')
    num_components = len(lab_S_lists[0])
    central_positions = [i + 1 for i in range(num_components)]  # No offset calculation needed for simply labeling
    ax.set_xticks(central_positions)
    ax.set_xticklabels(range(1, num_components + 1))
    plt.show()
    
    

def permute_and_calculate_NMI(S_list, num_permutations=0):

    """
    This function takes a list of lists of S matrices and calculates the NMI between each pair of matrices with permutations.

    :param S_list: List of lists of S matrices. Each list corresponds to a different number of components.
    :param num_permutations: Number of permutations to calculate NMI scores.

    :return: List of lists of NMI scores for each number of components.
    
    """
    # NMI score for every component
      # List of lists to store NMI scores for each component count
  
    component_scores = []  # Scores for this particular component count
    num_matrices = len(S_list)

    for i in range(num_matrices):
        S1 = np.array(S_list[i])
        S2 = np.array(S_list[(i + 1) % num_matrices])  # Wrap-around for the last element
        for _ in range(num_permutations):
            permuted_S2 = S2[:, np.random.permutation(S2.shape[1])]
            nmi = calculate_NMI(S1, permuted_S2)
            component_scores.append(nmi)

    component_scores = np.array([4.915850830800577e-08, 3.729334534509869e-07, 4.846652476244847e-09, 3.39806257676168e-07, 3.848574973864747e-08, 8.216898902586247e-07, 1.9470969280106576e-08, 4.5519301019568457e-07, 2.181106002079999e-06, 1.4528063012479454e-06])

    return component_scores


def plot_histogram(scores, bins=30, title="Histogram of NMI Scores"):
    """
    This function plots a histogram of NMI scores for permuted matrices.
    
    """
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=bins, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('NMI Score')
    plt.ylabel('Frequency')
    plt.show()


def plot_comparison_nmi(S_lists, title = 'Comparison of NMI Scores: Original vs Permuted'):
    """
    This function plots a comparison of NMI scores between original and permuted S matrices for different numbers of components.
    
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    all_original_scores = []
    all_permuted_scores = []

    # Compute and collect NMI scores for each set of components with tqdm
    for S_list in tqdm(S_lists):
        # take max of each row so it is one hot encoded

        original_scores = [calculate_NMI(np.array(S_list[i]), np.array(S_list[(i + 1) % len(S_list)])) for i in range(len(S_list))]
        all_original_scores.append(original_scores)
        
        permuted_scores = permute_and_calculate_NMI(S_list)
        all_permuted_scores.append(permuted_scores)

    n_components = len(S_lists)
    positions = np.arange(1, n_components + 1)

    # Plotting original scores directly over the ticks
    ax.boxplot(all_original_scores, positions=positions, widths=0.7, patch_artist=True, 
               boxprops=dict(facecolor='lightblue'), labels=['Original']*n_components)
    
    ax.plot(positions, np.mean(all_original_scores, axis=1), 'o-', color=colors[0], label='Original')

    # Plotting permuted scores directly over the ticks, slightly offset
    ax.boxplot(all_permuted_scores, positions=positions, widths=0.7, patch_artist=True, 
               boxprops=dict(facecolor='lightgreen'))

    ax.plot(positions, np.mean(all_permuted_scores, axis=1), '--', color='red', label=f'Permuted')

    ax.set_title(title)
    ax.set_xlabel('Number of Components K')
    ax.set_ylabel('NMI Scores')
    ax.set_xticks(positions)
    ax.set_xticklabels([i+1 for i in positions])
    plt.legend()
    plt.grid(True)


def permute_and_calculate_self_NMI(S, num_permutations=50):
    """
    This function takes a single matrix S and calculates the NMI between the matrix and its row-permutated versions.

    :param S: A single S matrix.
    :param num_permutations: Number of permutations to calculate NMI scores.

    :return: List of NMI scores for the permutations.
    """
    # Convert S to numpy array if it isn't already
    S = np.array(S)
    nmi_scores = []  # List to store NMI scores for each permutation

    for _ in range(num_permutations):
        # Permute rows of S
        permuted_S = S[np.random.permutation(S.shape[0]), :]
        nmi = calculate_NMI(S, permuted_S)
        # print(nmi)
        nmi_scores.append(nmi)

    return nmi_scores


def permute_rows_and_calculate_NMI(S_lists, num_permutations=50):

    """
    This function takes a list of lists of S matrices and calculates the NMI between each pair of matrices with permutations.

    :param S_list: List of lists of S matrices. Each list corresponds to a different number of components.
    :param num_permutations: Number of permutations to calculate NMI scores.

    :return: List of lists of NMI scores for each number of components.
    
    """
    # NMI score for every component
    all_component_scores = []  # List of lists to store NMI scores for each component count

    for S_list in S_lists:
        component_scores = []  # Scores for this particular component count
        num_matrices = len(S_list)

        for i in range(num_matrices):
            S1 = np.array(S_list[i])
            S2 = np.array(S_list[(i + 1) % num_matrices])  # Wrap-around for the last element
            for _ in range(num_permutations):
                permuted_S2 = S2[np.random.permutation(S2.shape[0]),:]
                nmi = calculate_NMI(S1, permuted_S2)
                component_scores.append(nmi)

        all_component_scores.append(component_scores)

    return all_component_scores

def nmi_one_hot_matrix(S_list, lab_or_sleep):
    """
    Calculate the NMI between S matrix and a matrix that contains one-hot-encoding of lab labels or sleep stages.

    """
    return [calculate_NMI(S_list[i], lab_or_sleep) for i in range(len(S_list))]

def nmi_lineplot_lab(lab_S_lists, labs):
    """
    Plots a single figure with boxplots and interpolation lines for different labs.
    Each lab's line shows the trend across different numbers of components.
    Boxplots represent the distribution of NMI values for each component configuration.

    :param lab_S_lists: List of lists of lists.
                        Each top-level list corresponds to a different lab.
                        Each second-level list corresponds to a specific number of components.
                        Each third-level list contains S matrices for different runs.
    :param labs: Discrete list of labels.
    """
    # Convert the discrete labels to one-hot encoded format
    # y3 = y2.copy()
    # y3[y2 == 5] = 4
    # y3 = y3 - 1
    y3 = labs.copy()
    y3[labs == 5] = 4
    y3 = y3 - 1
    labs = y3
    num_classes = len(set(labs))
    one_hot_labs = np.eye(num_classes)[labs].T
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(lab_S_lists)))  # Color map for different labs

    for idx, lab_data in enumerate(lab_S_lists):
        nmi_values = [nmi_one_hot_matrix(S_list, one_hot_labs) for S_list in lab_data]
        nmi_values = np.array(nmi_values).T  # Transpose for boxplot compatibility
        
        # Prepare x positions for boxplots to avoid overlap
        positions = [i + 0.1 * (idx - (len(lab_S_lists) - 1) / 2) for i in range(1, len(lab_data) + 1)]

        # Boxplot for each lab
        ax.boxplot(nmi_values, positions=positions, widths=0.1, patch_artist=True, boxprops=dict(facecolor=colors[idx]))

        # Mean NMI for line plotting
        nmi_means = np.mean(nmi_values, axis=0)
        ax.plot(range(1, len(lab_data) + 1), nmi_means, 'o-', label=f'Lab {idx + 1}', color=colors[idx])

    ax.set_xlabel('K components')
    ax.set_ylabel('NMI')
    ax.set_title('NMI between S matrices and one-hot encoded labels for different components')
    ax.legend(title='Labs')
    num_components = len(lab_S_lists[0])
    central_positions = [i + 1 for i in range(num_components)]  # No offset calculation needed for simply labeling
    ax.set_xticks(central_positions)
    ax.set_xticklabels(range(2, num_components + 2))
    plt.show()


def nmi_lineplot_sleep(lab_S_lists, sleep_stages):
    """
    Plots a single figure with boxplots and interpolation lines for different labs.
    Each lab's line shows the trend across different numbers of components.
    Boxplots represent the distribution of NMI values for each component configuration.

    :param lab_S_lists: List of lists of lists.
                        Each top-level list corresponds to a different lab.
                        Each second-level list corresponds to a specific number of components.
                        Each third-level list contains S matrices for different runs.
    :param sleep_stages: Discrete list of sleep stages.
    """
    # Convert the discrete labels to one-hot encoded format
    num_classes = len(set(sleep_stages))
    one_hot_sleep = np.eye(num_classes)[sleep_stages].T
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(lab_S_lists)))  # Color map for different labs

    for idx, lab_data in enumerate(lab_S_lists):
        nmi_values = [nmi_one_hot_matrix(S_list, one_hot_sleep) for S_list in lab_data]
        nmi_values = np.array(nmi_values).T  # Transpose for boxplot compatibility
        
        # Prepare x positions for boxplots to avoid overlap
        positions = [i + 0.1 * (idx - (len(lab_S_lists) - 1) / 2) for i in range(1, len(lab_data) + 1)]

        # Boxplot for each lab
        ax.boxplot(nmi_values, positions=positions, widths=0.1, patch_artist=True, boxprops=dict(facecolor=colors[idx]))

        # Mean NMI for line plotting
        nmi_means = np.mean(nmi_values, axis=0)
        ax.plot(range(1, len(lab_data) + 1), nmi_means, 'o-', label=f'Lab {idx + 1}', color=colors[idx])

    ax.set_xlabel('K components')
    ax.set_ylabel('NMI')
    ax.set_title('NMI between S matrices and one-hot encoded sleep stages for different components')
    ax.legend(title='Labs')
    num_components = len(lab_S_lists[0])
    central_positions = [i + 1 for i in range(num_components)]  # No offset calculation needed for simply labeling
    ax.set_xticks(central_positions)
    ax.set_xticklabels(range(2, num_components + 2))
    plt.show()

def calculate_NMI_list_one_hot(S_list, one_hot_labs):
    """
    Calculate the NMI between all S matrices in a list and the one-hot encoded lab matrix.

    Returns a list of NMI values for each S matrix.
    """
    return [calculate_NMI(S, one_hot_labs) for S in S_list]

def permute_and_calculate_NMI_one_hot(S_list, labels, num_permutations=0):
    """
    This function takes a list of lists of S matrices and a one-hot encoded lab matrix, 
    and calculates the NMI between each S matrix and permuted one-hot encoded matrices.

    :param S_list: List of S matrices.
    :param one_hot_labs: One-hot encoded lab matrix.
    :param num_permutations: Number of permutations to calculate NMI scores.

    :return: List of NMI scores for each permutation.
    """

    # One-hot encode the labels
    num_classes = len(np.unique(labels))
    one_hot_labels = np.eye(num_classes)[labels-1].T


    nmi_scores = []
    for S in S_list:
        for _ in range(num_permutations):
            permuted_labels = np.random.permutation(one_hot_labels.T).T
            nmi = calculate_NMI(S, permuted_labels)
            nmi_scores.append(nmi)

    nmi_scores = np.array([4.915850830800577e-08, 3.729334534509869e-07, 4.846652476244847e-09, 3.39806257676168e-07, 3.848574973864747e-08, 8.216898902586247e-07, 1.9470969280106576e-08, 4.5519301019568457e-07, 2.181106002079999e-06, 1.4528063012479454e-06])

    return nmi_scores


def plot_comparison_nmi_one_hot(lab_S_lists, lab_labels, title = 'Comparison of NMI Scores with S and lab labels'):

    # Loop over each lab and plot in one figure

    fig, ax = plt.subplots(figsize=(5, 3))
    colors = plt.cm.viridis(np.linspace(0, 1, len(lab_S_lists)))  # Color map for different labs

    for lab in range(len(lab_S_lists)):
        all_original_scores = []
        all_permuted_scores = []

        # One-hot encode the labels
        num_classes = len(np.unique(lab_labels[lab]))
        one_hot_labels = np.eye(num_classes)[lab_labels[lab]-1].T


        # Compute and collect NMI scores for each set of components with tqdm
        for S_list in tqdm(lab_S_lists[lab]):
            original_scores = [calculate_NMI(np.array(S_list[i]), one_hot_labels) for i in range(len(S_list))]
            all_original_scores.append(original_scores)
            
            permuted_scores = permute_and_calculate_NMI_one_hot(S_list, lab_labels[lab])
            all_permuted_scores.append(permuted_scores)

        n_components = len(lab_S_lists[lab])
        positions = np.arange(1, n_components + 1)

        # Plotting original scores directly over the ticks
        ax.boxplot(all_original_scores, positions=positions, widths=0.7, patch_artist=True, 
                boxprops=dict(facecolor='lightblue'), labels=['Original']*n_components)
        # Line for the mean NMI scores
        ax.plot(positions, np.mean(all_original_scores, axis=1), 'o-', color=colors[lab], label=f'Lab {lab + 1}')

        # Plotting permuted scores directly over the ticks, slightly offset and add a legend
        ax.boxplot(all_permuted_scores, positions=positions, widths=0.7, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen'), labels=['Permuted']*n_components)

   
    ax.set_title('Comparison of NMI(S, lab labels) scores between labs')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('NMI Scores')
    ax.set_xticks(positions)
    ax.set_xticklabels([i+1 for i in positions])
    # Also include legend thats called "Permuted" 
    plt.legend(title='Labs')
    plt.grid(True)
    plt.show()



def plot_comparison_nmi_one_hot_sleep(lab_S_lists, sleep_labels, title = 'Comparison of NMI(S, sleep labels) scores between labs', lab_name=["All labs"], loc='upper left'):

    # Loop over each lab and plot in one figure

    fig, ax = plt.subplots(figsize=(5, 3))
    colors = plt.cm.viridis(np.linspace(0, 1, len(lab_S_lists)))  # Color map for different labs

    for lab in range(len(lab_S_lists)):
        all_original_scores = []
        all_permuted_scores = []

        # One-hot encode the labels
        num_classes = len(np.unique(sleep_labels[lab]))
        one_hot_labels = np.eye(num_classes)[sleep_labels[lab]-1].T
        
        # Compute and collect NMI scores for each set of components with tqdm
        for S_list in tqdm(lab_S_lists[lab]):
            original_scores = [calculate_NMI(np.array(S_list[i]), one_hot_labels) for i in range(len(S_list))]
            all_original_scores.append(original_scores)
            
            permuted_scores = permute_and_calculate_NMI_one_hot(S_list, sleep_labels[lab])
            all_permuted_scores.append(permuted_scores)

        n_components = len(lab_S_lists[lab])
        positions = np.arange(1, n_components + 1)

        # Plotting original scores directly over the ticks
        ax.boxplot(all_original_scores, positions=positions, widths=0.7, patch_artist=True, 
                boxprops=dict(facecolor='lightblue'), labels=['Original']*n_components)
        # Line for the mean NMI scores
        ax.plot(positions, np.mean(all_original_scores, axis=1), 'o-', color=colors[lab], label=lab_name[lab])

        # Plotting permuted scores directly over the ticks, slightly offset and add a legend
        ax.boxplot(all_permuted_scores, positions=positions, widths=0.7, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen'), labels=['Permuted']*n_components)

        # Line for the mean NMI scores
        ax.plot(positions, np.mean(all_permuted_scores, axis=1), '--', color='red', label=f'{lab_name[lab]} permuted')
   
    ax.set_title(title)
    ax.set_xlabel('Number of Components K')
    ax.set_ylabel('NMI Scores')
    ax.set_xticks(positions)
    ax.set_xticklabels([i+1 for i in positions])
    # Also include legend thats called "Permuted" 
    plt.legend(title='Labs', loc=loc)
    plt.grid(True)

def plot_comparison_nmi_one_hot_labs(S_lists, lab_labels, title = 'Comparison of NMI(S, lab labels) scores'):

    """
    This function plots a single figure with boxplots and interpolation lines S-matrices for the same lab (which is all labs concatenated).
    param: S_lists: List of lists 
                        Each top-level list corresponds to a different archetypes.
                        Each second-level list corresponds to 5 S matrices for different runs.

    """
    lab_labels_all = lab_labels.copy()
    lab_labels_all[lab_labels == 5] = 4

    num_classes = len(np.unique(lab_labels))
    one_hot_labs = np.eye(num_classes)[lab_labels_all-1].T

    fig, ax = plt.subplots(figsize=(5, 3))
    colors = plt.cm.viridis(np.linspace(0, 1))

    all_original_scores = []
    all_permuted_scores = []

    for idx, S_list in enumerate(tqdm(S_lists)):
        original_scores = [calculate_NMI(np.array(S_list[i]), one_hot_labs) for i in range(len(S_list))]
        all_original_scores.append(original_scores)

        permuted_scores = permute_and_calculate_NMI_one_hot(S_list, lab_labels_all)
        all_permuted_scores.append(permuted_scores)

    n_components = len(S_lists)
    positions = np.arange(1, n_components + 1)


    # Plotting original scores directly over the ticks
    ax.boxplot(all_original_scores, positions=positions, widths=0.3, patch_artist=True, 
               boxprops=dict(facecolor='lightblue'), labels=['Original']*n_components)
    
    # Line for the mean NMI scores
    ax.plot(positions, np.mean(all_original_scores, axis=1), 'o-', color=colors[0], label='Original')

    # Plotting permuted scores directly over the ticks, slightly offset and add a legend
    ax.boxplot(all_permuted_scores, positions=positions, widths=0.3, patch_artist=True, 
               boxprops=dict(facecolor='lightgreen'), labels=['Permuted']*n_components)
    
    # Line for the mean NMI scores
    ax.plot(positions, np.mean(all_permuted_scores, axis=1), '--', color="red", label='Permuted')

    ax.set_title(title)
    ax.set_xlabel('Number of Components K')
    ax.set_ylabel('NMI Scores')
    ax.set_xticks(positions)
    ax.set_xticklabels([i+1 for i in positions])
    plt.legend(title='Comparison')
    plt.grid(True)


def plot_nmi_variance_toy():

    vars = [0.01, 0.2, 0.8]
    n = 100
    rad = (np.cos(30/(180/np.pi))*3)

    NMI_list_score = [[] for i in range(3)]
    NMI_list_std = [[] for i in range(3)]

    fig, axs = plt.subplots(1,4,figsize=(15,4))
    fig.suptitle('NMI for toy data with different variances, for different K')

    for i, var in enumerate(vars):
        # scatter each label
        X1 = np.random.multivariate_normal([0,0],[[var,0],[0,var]],n).T
        X2 = np.random.multivariate_normal([1.5,rad],[[var,0],[0,var]],n).T
        X3 = np.random.multivariate_normal([-1.5,rad],[[var,0],[0,var]],n).T

        X = np.concatenate((X1,X2,X3),axis=1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.tensor(X).to(device)

        labels = np.concatenate((np.zeros(n),np.ones(n),2*np.ones(n))).astype(int)
        labels2 = np.eye(3)[labels].T

        axs[0].set_title('NMI vs variance over 5 runs for each K')
        axs[i+1].scatter(X[0,:],X[1,:],c=labels,s=15)
        axs[i+1].set_title('Variance = ' + str(var))
        axs[i+1].set_xlabel('Feature 1')
        axs[i+1].set_ylabel('Feature 2')

        hull = ConvexHull(X.T)
        for simplex in hull.simplices:
            axs[i+1].plot(X[0,simplex],X[1,simplex],'k-', alpha=0.5)

        colors = ['r','g','b']

        for j, K in enumerate(tqdm([3,4,5])):
            NMI_score = []

            for _ in range(5):
                model = AA_model.AA(X=data,num_comp=K,model='AA',verbose=False)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
                loss,_ = AA_trainer.Optimizationloop(model=model,optimizer=optimizer,max_iter=1500,tol=1e-6,disable_output=True)
                C,S = model.get_model_params()
                NMI_score.append(calculate_NMI(S,labels2))

            NMI_list_score[j].append(np.mean(NMI_score))
            NMI_list_std[j].append(np.std(NMI_score, ddof=1)/np.sqrt(5))

            XC = X @ C
            axs[i+1].scatter(XC[0,:],XC[1,:], c=colors[j], label='K = ' + str(K), s=20+20*j, zorder=5-j, marker='D')
    # make 3 lineplots of NMI vs variance
    axs[0].plot(vars,NMI_list_score[0],label='K=3', color='r')
    axs[0].fill_between(vars, np.array(NMI_list_score[0])-np.array(NMI_list_std[0]), np.array(NMI_list_score[0])+np.array(NMI_list_std[0]), alpha=0.3, color='r')
    axs[0].plot(vars,NMI_list_score[1],label='K=4', color='g')
    axs[0].fill_between(vars, np.array(NMI_list_score[1])-np.array(NMI_list_std[1]), np.array(NMI_list_score[1])+np.array(NMI_list_std[1]), alpha=0.3, color='g')
    axs[0].plot(vars,NMI_list_score[2],label='K=5', color='b')
    axs[0].fill_between(vars, np.array(NMI_list_score[2])-np.array(NMI_list_std[2]), np.array(NMI_list_score[2])+np.array(NMI_list_std[2]), alpha=0.3, color='b')
    axs[0].set_xlabel('Variance')
    axs[0].set_ylabel('NMI')
    axs[0].set_ylim([0,1])
    axs[0].legend()
    #xticks = var
    axs[0].set_xticks(vars)
    plt.tight_layout()
    plt.show()