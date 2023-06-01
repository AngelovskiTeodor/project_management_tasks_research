import pandas

def get_data_from_csv():
    """Reads Jira Tasks data from CSV file and returns pandas Dataframe"""
    dataframe = pandas.read_csv("./jira_dataset/jira_database_public_jira_issue_report.csv")
    return dataframe

def get_unique_values(dataframe, column_name):
    """ Returns a list of all unique values in provided column """
    return dataframe[column_name].unique()

def get_rows_with_missing_values(dataframe, column_name):
    """ Return rows with empty or missing values under column_name """
    sub_dataframe = dataframe[dataframe[column_name].isnull()]
    return sub_dataframe

if __name__=="__main__":
    df = get_data_from_csv()
    df.info()
    
    print("Task statuses: ", end='')
    print(get_unique_values(df,"status"))

    print("Task types: ", end='')
    print(get_unique_values(df,"type"))

    sub_df = get_rows_with_missing_values(df, "description")
    sub_df.info()
