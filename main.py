import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import Apriori as apriori_module

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data    

def main():
    data = load_data('data/Groceries_dataset.csv')
    apriori_module.run_apriori(data)


if __name__ == "__main__":
    main()