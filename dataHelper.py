import pandas as pd
import numpy as np
from pymongo import MongoClient

FILE_PATH = 'data/Groceries_dataset.csv'

def load_data(file_path=FILE_PATH):
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

def print_member_items(data, x):
    member_items = sort_member_numbers(data)
    for member, items in list(member_items.items())[:x]:
        print(f"Member {member}: {items}")

def load_data_from_mongodb(connection_string, database_name, collection_name):
    #Load data from a MongoDB collection and return as a DataFrame.
    # connection_string: MongoDB connection URI (e.g., 'mongodb://localhost:27017/')
    # database_name: Name of the database
    # collection_name: Name of the collection to read from
    

    client = MongoClient(connection_string)
    db = client[database_name]
    collection = db[collection_name]
    
    # Fetch all documents from the collection
    documents = list(collection.find())
    
    # Close the connection
    client.close()
    
    # Convert to DataFrame
    if documents:
        df = pd.DataFrame(documents)
        # Optionally remove MongoDB's _id field if present
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])
        return df
    else:
        return pd.DataFrame()