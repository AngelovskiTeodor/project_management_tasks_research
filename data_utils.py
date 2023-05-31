import pandas

def get_data_from_csv():
    """Reads Jira Tasks data from CSV file and returns pandas Dataframe"""
    dataframe = pandas.read_csv("./jira_dataset/jira_database_public_jira_issue_report.csv")
    return dataframe

if __name__=="__main__":
    df = get_data_from_csv()
    df.info()