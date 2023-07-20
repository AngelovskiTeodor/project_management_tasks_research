import pandas

def get_jira_tasks_from_csv():
    """Reads Jira Tasks data from CSV file and returns pandas Dataframe"""
    #DIRECTORY = '/content/drive/MyDrive/Faks/research_uiktp/jira_dataset/jira_database_public_jira_issue_changelog_item.csv'   # GOOGLE DRIVE PATH when using Google Collab
    DIRECTORY = "./jira_dataset/jira_database_public_jira_issue_report.csv"     #   Local repo path
    dataframe = pandas.read_csv(DIRECTORY)
    return dataframe

def get_unique_values(dataframe, column_name):
    """ Returns a list of all unique values in provided column """
    return dataframe[column_name].unique()

def get_rows_with_missing_values(dataframe, column_name):
    """ Return rows with empty or missing values under column_name """
    sub_dataframe = dataframe[dataframe[column_name].isnull()]
    return sub_dataframe

def get_task_descriptions():
    """Returns the descriptions for the tasks"""
    df = get_jira_tasks_from_csv()
    df = df[df["description"].notnull()]
    df = df[["id", "description"]]
    return df

if __name__=="__main__":
    df = get_jira_tasks_from_csv()
    df.info()
    
    print("Task statuses: ", end='')
    print(get_unique_values(df,"status"))

    print("Task types: ", end='')
    print(get_unique_values(df,"type"))

    sub_df = get_rows_with_missing_values(df, "description")
    sub_df.info()

    print("Descriptions:")
    df = get_task_descriptions()
    df.info()
