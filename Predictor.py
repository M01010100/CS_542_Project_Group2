import sklearn
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data    

def sort_member_numbers(data):
    # Sorts data by Member_number and returns a dictionary of all items per member
    sorted_data = data.sort_values(by="Member_number", ascending=True)
    items = dict()
    for row in sorted_data.iterrows():
        member_number = row[1]['Member_number']
        item_description = row[1]['itemDescription']
        if member_number not in items:
            items[member_number] = []
        items[member_number].append(item_description)
    return items

def main():
    data = load_data('data/Groceries_dataset.csv')
    sorted_items = sort_member_numbers(data)
    print(data.head())
    print(data.sort_values(by="Member_number", ascending=True).head())
    member_items = sort_member_numbers(data)
    for member, items in list(member_items.items())[:5]:
        print(f"Member {member}: {items}")

if __name__ == "__main__":
    main()