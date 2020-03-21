import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """The load_data function load the csv file into pandas dataframe

    Args:
        messages_filepath (filepath): the filepath for the messages
        categories_filepath (filepath): the filepath for the categories data



    Returns:
        DataFrame: The DataFrame for analysis

    """

    # load the messages csv
    mess = pd.read_csv(messages_filepath)

    # load the categories csv
    catt = pd.read_csv(categories_filepath)

    # merge the datasets
    data = mess.merge(catt, on="id")

    return data


def clean_data(dataset):
    """The clean_data function will return a clean DataFrame after removing,
        replacing and cleaning the DataFrame to  a suitable form for further
        saving to database for analysis

    Args:
        dataset (DataFrame): the DataFrame for data wrangling

    Returns:
        DataFrame: The DataFrame for saving to database and later analysis

    """
    # Split the values in the categories column on the ; character so that
    # each value becomes a separate column
    cat = dataset.categories.str.split(";", expand=True)

    # Use the first row of categories dataframe to create column names for the
    # categories data.
    row = cat.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])

    # Rename columns of categories with new column names.
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
    # category_colnames = row.str.extract(r'([\w]+)', expand=False)
>>>>>>> eee255ada9eddf0cb8057930709b318d8a4d5262
>>>>>>> fc5f29392dcef867646c760353a56392c7e8847e
    cat.columns = category_colnames

    # extract only the digits in categories columns
    for column in cat:
        # set each value to be the last character of the string
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
        # categories[column] = (
        #     categories[column].str.extract(
        #         r"(\d+)", expand=False).astype(int))
>>>>>>> eee255ada9eddf0cb8057930709b318d8a4d5262
>>>>>>> fc5f29392dcef867646c760353a56392c7e8847e
        cat[column] = cat[column].str[-1:]

        # convert column from string to numeric
        cat[column] = cat[column].astype(int)

    # drop the original categories column from dataset
    dataset.drop(columns=["categories"], inplace=True)

    # concatenate the original dataframe with the new `categories`
    # dataframe
    df_ = pd.concat([dataset, cat], axis=1)

    # drop duplicates and columns not essential for further analysis
    df_.drop_duplicates(keep="first", inplace=True)
<<<<<<< HEAD
=======
    df_.drop(columns=["id", "original"], inplace=True)

    # drop columns not needed
>>>>>>> fc5f29392dcef867646c760353a56392c7e8847e
    df_.drop(columns=["id", "original"], inplace=True)

    return df_


def save_data(dtss, database_filepath):
    """The save_data function save the dataframe to sql database

    Args:
        dtss (DataFrame): the DataFrame to save to sql
        database_filepath (filepath): filepath of the sql database


    Returns: None

    """
    # create engine 
    engine = create_engine(f"sqlite:///{database_filepath}", echo=False)

    # save to database
    dtss.to_sql(
        "disasterTable",
        engine,
        index=False,
        if_exists="replace")


def main():
    if len(sys.argv) == 4:
        print(" ")
        messages_filepath, categories_filepath, database_filepath = sys.argv[
            1:
        ]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            ),
            end="\n\n",
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...\n\n")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(
            database_filepath), "\n\n", )
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db")


if __name__ == "__main__":
    main()
