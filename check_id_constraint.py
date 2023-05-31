import pandas as pd
from data_utils import get_data_from_csv as get_data

def is_column_unique(dataframe, column_name):
    id_series = dataframe[column_name].squeeze()
    return id_series.is_unique

def check_id_unique_constraint(dataframe, id_column_name="id"):
    return is_column_unique(dataframe, id_column_name)

if __name__=="__main__":
    df = get_data()
    df.info()
    print("Is ID unique constraint satisfied {}".format(check_id_unique_constraint(df)))
    print("Is title column unique: {}".format(is_column_unique(df, "title")))
