import os
import torch
import pandas

class GlobalConstants:
    def __init__(self):
        google_collab_environment = True if os.getenv("COLAB_RELEASE_TAG") else False
        if google_collab_environment:
            self.PROCESSED_DATA_PATH = "/content/drive/MyDrive/Faks/research_uiktp/processed_data/processed_data.csv"
            self.CSV_DATASET_PATH = "/content/drive/MyDrive/Faks/research_uiktp/processed_data/processed_data.csv"
            self.MODEL_PATH = "/content/drive/MyDrive/pytorch_model.bin"
            self.DIRECTORY = '/content/drive/MyDrive/Faks/research_uiktp/jira_dataset/jira_database_public_jira_issue_changelog_item.csv'
            self.CHANGELOGS_CSV_FILE_PATH = '/content/drive/MyDrive/Faks/research_uiktp/jira_dataset/jira_database_public_jira_issue_changelog_item.csv'
        else:
            self.PROCESSED_DATA_PATH = "./processed_data/processed_data.csv"
            self.CSV_DATASET_PATH = "./processed_data/processed_data.csv"
            self.MODEL_PATH = "pytorch_model.bin"
            self.DIRECTORY = "./jira_dataset/jira_database_public_jira_issue_report.csv"
            self.CHANGELOGS_CSV_FILE_PATH = './jira_dataset/jira_database_public_jira_issue_changelog_item.csv'

        if torch.cuda.is_available():
            self.DEVICE_STRING = "cuda"
        else:
            print("GPU is not available. CPU will be used to train the model")
            self.DEVICE_STRING = "cpu"
        self.DEVICE = torch.device(self.DEVICE_STRING)
    

def get_jira_tasks_from_csv():
    """Reads Jira Tasks data from CSV file and returns pandas Dataframe"""
    dataframe = pandas.read_csv(get_global_constants().DIRECTORY)
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

def reverse_dataframe(dataframe:pandas.DataFrame):
    """Reverses the order of the rows"""
    dataframe = dataframe.iloc[::-1]
    return dataframe

def rename_columns(dataframe:pandas.DataFrame, current_column_names:list=['description', 'duration'], new_column_names=['text','label']):
    """Renames column names provided in arguments as list/tuple, to column names provided in arguments as list/tuple"""
    if len(current_column_names) != len(new_column_names):
        print("Can not pair current names to new names")
        raise RuntimeError()
    renames = dict()
    for i in range(len(current_column_names)):
        renames[current_column_names[i]] = new_column_names[i]
    dataframe = dataframe.rename(columns=renames, inplace=False)
    return dataframe

def get_global_constants():
    """Returns object with required constants relevant to the environment that the code is running on"""
    return GlobalConstants()

def serialize_to_csv(dataframe:pandas.DataFrame, file_path=get_global_constants().PROCESSED_DATA_PATH):
    dataframe.to_csv(file_path, sep=',', index=False, encoding='utf-8')
    return dataframe

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
