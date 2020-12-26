import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys

def extract(messages_filepath, categories_filepath):
    """
    Load messages and categories data

    Parameters:
        messages_filepath (str): csv file path of messages
        categories_filepath (str): csv file path of categories

    Returns:
        messages (DataFrame): message data in pandas dataframe format
        categories (DataFrame): categories data in pandas dataframe format
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages, categories

def transform(messages, categories):
    """
    Transform data from messages and categories into expected output. 

    Parameters:
        messages (DataFrame): message data in pandas dataframe format
        categories (DataFrame): categories data in pandas dataframe format

    Returns:
        df (DataFrame): transformed data
    """

    # Merge the messages and categories datasets using the common id
    df = messages.merge(categories, how="inner", on="id")

    # create a dataframe of the 36 individual category columns
    categories = categories["categories"].str.split(";", expand=True)

    # select the first row of the categories dataframe to extract a list of new column names for categories.
    category_colnames =  [col.split("-")[0] for col in categories.iloc[0,:].to_list()]
    categories.columns = category_colnames

    # Convert category values to just numbers 0 and 1
    for column in categories:
        categories[column] = categories[column].astype(str).str.split("-").str[1]
        categories[column] = categories[column].astype(int)

    # Replace categories column in df with new category columns.
    df.drop(["categories"], axis=1, inplace=True)
    df = pd.concat([df,categories], join="inner", axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def load(df, db_file_path, table_name):
    """
    Load transformed data into sqlite database

    Parameters:
        df (DataFrame): transformed data
        db_file_path (str): path to stored sqlite database
        table_name (str): Table name to store df

    Returns:
        None
    """
    # Save the clean dataset into an sqlite database
    engine = create_engine("sqlite:///" + db_file_path)
    df.to_sql(table_name, engine, index=False, if_exists="replace")

def main():
    """
    Main function to run the data pipeline to extract, transform and load data into target sqlite database
    """ 
    print(sys.argv)

    if len(sys.argv) == 5:
        messages_filepath, categories_filepath, db_file_path, table_name = sys.argv[1:]

        print("Extract messages and categories from csv files")
        messages, categories = extract(messages_filepath, categories_filepath)

        print("Transform data")
        df = transform(messages, categories)

        print("Load transformed data into sqlite database")
        load(df, db_file_path, table_name)

        print("Data pipeline executed successfully!")
    else:
        print(
            """
            Please pass the arguments in correctly orders as below example:
            python process_data.py messages.csv categories.csv disaster_response.db disaster_response
            """
        )

if __name__=="__main__":
    main()
