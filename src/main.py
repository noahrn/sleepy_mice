from preprocessing.data_loader import load_and_process_data
import pandas as pd

# example to check functionality of repository & config file
df = load_and_process_data(remove_outliers = True, normalize=False, lab="all", verbose=True, narcolepsy=True)