from utils.stat_functions import *
import pickle
import matplotlib.pyplot as plt

# Load the data from a pickle file
S_lists = pickle.load(open("data/S_list_list_lab1.pkl", "rb"))

nmi_boxplot(S_lists)

for S_list in S_lists:
    print(calculate_NMI_list(S_list))
    print("\n")


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

# plot_histogram(permute_and_calculate_NMI(S_lists, num_permutations=50)[0], bins=20, title="Histogram of permutated NMI for K = 2")


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
        print(nmi)
        nmi_scores.append(nmi)

    return nmi_scores

plot_histogram(permute_and_calculate_self_NMI(S_lists[3][0], num_permutations=1000), bins=20, title="Histogram of permutated NMI(S1, S1_permu) for K = 5")