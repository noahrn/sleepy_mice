from preproccesing.data_loader import load_and_process_data
from sklearn import svm

data = load_and_process_data(normalize=True, lab="all")