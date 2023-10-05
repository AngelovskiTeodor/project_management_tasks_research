import os
import torch
import pandas
import progressbar
#from tqdm import tqdm

class GlobalConstants:
    def __init__(self):
        google_collab_environment = True if os.getenv("COLAB_RELEASE_TAG") else False
        if google_collab_environment:
            self.PROCESSED_DATA_PATH = "/content/drive/MyDrive/Faks/research_uiktp/processed_data/processed_data.csv"
            self.CSV_DATASET_PATH = "/content/drive/MyDrive/Faks/research_uiktp/processed_data/processed_data.csv"
            self.MODEL_PATH = "/content/drive/MyDrive/Faks/research_uiktp/pytorch_model.bin"
            self.DIRECTORY = '/content/drive/MyDrive/Faks/research_uiktp/jira_dataset/jira_database_public_jira_issue_report.csv'
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
        self.RANDOM_STATE = 42
    
def progress_bar(max_value):
    """Progress bar"""
    widgets = [
        ' [', 
            progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
        '] ',
        progressbar.Bar('*'),' (',
        progressbar.ETA(), ') ',
    ]
    bar = progressbar.ProgressBar(maxval=max_value, widgets=widgets)
    return bar

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

def count_samples_from_class(dataframe:pandas.DataFrame, data_class, column_name='label'):
    """Counts the samples from provided label/class/duration"""
    sub_df = dataframe.loc[dataframe[column_name] == data_class]
    number_of_samples = len(sub_df)
    return number_of_samples

def get_dataset_classes(dataframe:pandas.DataFrame, column_name='label'):
    """Returns a list of all different classes in dataframe"""
    return get_unique_values(dataframe, column_name)

def count_samples_per_class(dataframe:pandas.DataFrame, column_name='label'):
    """Returns dictionary with number of samples for each class in the dataset"""
    classes = get_dataset_classes(dataframe, column_name=column_name)
    ret = {}
    for c in classes:
        ret[c] = count_samples_from_class(dataframe, c, column_name=column_name)
    return ret

def get_class_with_the_least_samples(dataframe:pandas.DataFrame):
    """Returns the class/label/duration with the least samples"""
    number_of_samples_per_class = count_samples_per_class(dataframe)
    classes = number_of_samples_per_class.keys()
    classes = list(classes)
    
    print(classes)  #   debugging

    min_class = classes[0]
    min_value = number_of_samples_per_class[min_class]
    for c in classes:
        if number_of_samples_per_class[c] < min_value:
            min_class = c
            min_value = number_of_samples_per_class[c]
    return min_class

def get_maximum_sample_size_for_down_sampling(dataframe:pandas.DataFrame):
    """Returns the number of samples in each class for balancing and downsampling the dataset"""
    min_class = get_class_with_the_least_samples(dataframe)
    number_of_samples = count_samples_from_class(dataframe, min_class)
    return number_of_samples

def down_sample_dataframe(dataframe:pandas.DataFrame):
    """Returns balanced dataframe in which the number of samples for each class
    is the lowest number of samples between the classes"""
    sample_size = get_maximum_sample_size_for_down_sampling(dataframe)
    dataframe = (dataframe.groupby('label', as_index=False)
        .apply(lambda x: x.sample(n=sample_size, random_state=get_global_constants().RANDOM_STATE))
        .reset_index(drop=True))
    return dataframe

def balance_dataframe(dataframe:pandas.DataFrame):
    """Returns dataframe with the same number of samples for each class"""
    ret = down_sample_dataframe(dataframe)
    return ret

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
