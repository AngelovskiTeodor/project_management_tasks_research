# %%

import pandas
from datetime import datetime
from data_utils import get_unique_values, reverse_dataframe
from check_id_constraint import check_id_unique_constraint

CHANGELOGS_CSV_FILE_PATH = './jira_dataset/jira_database_public_jira_issue_changelog_item.csv'
#CHANGELOGS_CSV_FILE_PATH = '/content/drive/MyDrive/Faks/research_uiktp/jira_dataset/jira_database_public_jira_issue_changelog_item.csv'
TASK_ID_COLUMN_NAME = "issue_report_id"
STARTING_STATUS_VALUES = ["Open", "Reopened", "New", "Patch Available"]
ENDING_STATUS_VALUES = ["Closed", "Resolved", "Done", "Submitted"]

class ChangelogError(RuntimeError):
    def __init__(self, changelog_entry:pandas.Series | pandas.DataFrame, message) -> None:
        super().__init__(message)
        self.changelog_entry = changelog_entry

class ChangelogEntryDoesNotConatinChangeToStatusFieldError(ChangelogError):
    def __init__(self, changelog_entry:pandas.Series, message="This changelog entry does not contain any change to the status field. "):
        super().__init__(changelog_entry, message=message)

class EmptyChangelogDataframeError(ChangelogError):
    def __init__(self, changelog_entry: pandas.DataFrame, message="This dataframe does not contain any changelog") -> None:
        super().__init__(changelog_entry, message)

class DataframeWithoutEnoughChangelogsError(ChangelogError):
    def __init__(self, changelog_entry: pandas.DataFrame, message="This dataframe does not contain changelog with the requred value or contains only one changelog") -> None:
        super().__init__(changelog_entry, message)

class DateByStatusNotFoundError(ChangelogError):
    def __init__(self, changelog_entry: pandas.DataFrame, message="Date not found because the dataframe does not contain status value that starts or ends tasks") -> None:
        super().__init__(changelog_entry, message)

def get_changelogs_crom_csv(csv_file=CHANGELOGS_CSV_FILE_PATH):
    """Returns Pandas Dataframe with changelog data for jira issues"""
    dataframe = pandas.read_csv(csv_file, parse_dates=['date'], date_format="%Y-%m-%d %H:%M:%S.%f")#, index_col="id")
    return dataframe

def changelogs_raw_dataframe_validation(dataframe=None):
    if dataframe is None:
        dataframe = get_changelogs_crom_csv()
    print("\nAll changelogs dataframe:") # debugging
    print(dataframe) # debugging
    dataframe = get_status_changelogs(dataframe)
    print("Status changelogs dataframe:") # debugging
    print(dataframe) # debugging
    return dataframe

def filter_dataframe_by_column_value(dataframe:pandas.DataFrame, column_name, column_value):
    """Returns dataframe with rows that have the column_value in the column_name column"""
    ret = dataframe[ dataframe[column_name] == column_value ]
    return ret

def get_all_changelogs_for_single_task_by_id(changelogs:pandas.DataFrame, task_id):
    """Returns dataframe with changelogs that contain changes for the task provided as argument"""
    ret = changelogs[ changelogs[TASK_ID_COLUMN_NAME] == task_id ]
    return ret

def get_status_changelogs(dataframe):
    """The changelogs data contains logs for all column
    This method filters logs that apply to the 'status' column"""
    COLUMN_NAME = "field_name"
    COLUMN_VALUE = "status"
    ret = filter_dataframe_by_column_value(dataframe, column_name=COLUMN_NAME, column_value=COLUMN_VALUE)
    return ret 

def filter_durations_in_range(dataframe, min=1, max=10, column_name='duration'):
    """Removes all tasks with duration less than min and greater than max"""
    ret = dataframe[
        (dataframe[column_name] >= min) & \
        (dataframe[column_name] <= max)
    ]
    return ret

def plot_histogram_of_column(dataframe, column_name, min=None, max=None):
    """Plots histogram of a column in the dataset provided as agruments
    param dataframe: Pandas Dataframe
    param min: Minimal value for x-axis
    param max: Maximum value for x-axis
    """
    if min==None or max==None:
        return dataframe.hist(column=column_name, grid=True, edgecolor='black', align='left')
    else:
        return dataframe.hist(column=column_name, grid=True, range=(min, max), edgecolor='black', align='left', bins=max-min)

def plot_durations_histogram(dataframe, min=None, max=None, column_name='duration'):
    """Plots histogram for the task durations
    param dataframe: Pandas Dataframe containing task durations
    param min: Minimal value for x-axis
    param max: Maximum value for x-axis
    """
    return plot_histogram_of_column(dataframe, column_name, min, max)

def count_subdataframes_with_more_than_one_changelog_entries(changelog_dataframes:dict):
    ret = 0
    for id in changelog_dataframes.keys():
        if len(changelog_dataframes[id].index):
            ret += 1
    return ret

def get_changelogs_by_task_sorted(changelogs:pandas.DataFrame, ids:list):
    """Returns dictionary with the changelog for each id
    the changelogs are sorted by date"""
    status_changelogs_by_task_id = dict()   # for each task there is a dataframe containing the changelogs for the specified task
    for id in ids:
        status_changelogs_for_task = get_all_changelogs_for_single_task_by_id(changelogs, id)
        status_changelogs_for_task.sort_values(by=['date'])
        status_changelogs_by_task_id[id] = status_changelogs_for_task
    status_changelogs_for_task.info()   # debugging
    print("Sorted changelogs by date for task id {}".format(id))    # debugging
    print(status_changelogs_for_task)   # debugging

    return status_changelogs_by_task_id

#def validate_changelog_countaining_change_to_status_field(changelog, changelog_column_name="new_value", changed_field="status"):

def is_containing_any_value(series:pandas.Series, values, changelog_column_name="new_value", changed_field="status"):
    """Returns True if at least one of the values provided as argument can be found in the series"""
    CHANGED_FIELD_COLUMN_NAME = "field_name"
    if changed_field != series[CHANGED_FIELD_COLUMN_NAME]:
        print("series.index = {}".format(series.index))
        raise ChangelogEntryDoesNotConatinChangeToStatusFieldError(series)
    ret = False
    for value in values:
        if value in series[changelog_column_name]:
            ret = True
            return ret
    return ret

def validate_changelog_dataframe_length(changelogs:pandas.DataFrame):
    """Returns True if the dataframe has more than one changelog"""
    if changelogs is None: return False
    if changelogs.empty: return False
    if len(changelogs.index) < 2: return False
    return True

def validate_changelog_containing_status_values(changelogs:pandas.DataFrame, status_values:list=STARTING_STATUS_VALUES.extend(ENDING_STATUS_VALUES)):
    for i, row in changelogs.iterrows():
        if is_containing_any_value(row, status_values): return True
    return False

def validate_positive_duration(starting_date, ending_date):
    if starting_date >= ending_date: return False
    return True

def get_all_changelogs_with_status_value(changelogs:pandas.DataFrame, value):
    FIELD_COLUMN_NAME = 'field_name'
    VALUE_COLUMN_NAME = 'new_value'
    ret = changelogs[ changelogs[VALUE_COLUMN_NAME] == value ]
    return ret

def count_changelogs_with_status_value(changelogs:pandas.DataFrame, value):
    filtered_changelogs = get_all_changelogs_with_status_value(changelogs, value)
    ret = len(filtered_changelogs.index)
    return ret

def get_tasks_with_status_value(changelogs:pandas.DataFrame, value):
    VALUE_COLUMN_NAME = 'new_value'
    ret = changelogs[ changelogs[VALUE_COLUMN_NAME] == value ]
    ret = get_unique_values(changelogs, TASK_ID_COLUMN_NAME)
    return ret

def count_tasks_with_current_status(changelogs:pandas.DataFrame, value):
    ret = get_tasks_with_status_value(changelogs, value)
    ret = len(ret)
    return ret

def is_starting_status(changelog_entry:pandas.Series, status_values:list = STARTING_STATUS_VALUES):
    """Returns True if new value of status is changed to
    any status in for the NEW_VALUES list"""
    return is_containing_any_value(changelog_entry, status_values)

def is_ending_status(changelog_entry:pandas.Series, status_values:list = ENDING_STATUS_VALUES):
    """Returns True if new value of status is changed to
    any status in for the NEW_VALUES list"""
    return is_containing_any_value(changelog_entry, status_values)

def find_first_date_by_status(changelogs:pandas.DataFrame, status_values):
    COLUMN_CONTAINING_STATUS_VALUE = "new_value"
    if changelogs is None or changelogs.empty:
        raise EmptyChangelogDataframeError(changelogs)
    if not validate_changelog_dataframe_length(changelogs):
        raise DataframeWithoutEnoughChangelogsError(changelogs)
    
    for i, changelog in changelogs.iterrows():
        current_status = changelog[COLUMN_CONTAINING_STATUS_VALUE]
        current_date = changelog['date']
        if current_status in status_values:
            return current_date
    raise DateByStatusNotFoundError(changelogs)

def find_starting_date_by_status(changelogs:pandas.DataFrame):
    return find_first_date_by_status(changelogs, STARTING_STATUS_VALUES)

def find_ending_date_by_status(changelogs:pandas.DataFrame):
    changelogs = reverse_dataframe(changelogs)
    return find_first_date_by_status(changelogs, ENDING_STATUS_VALUES)

def calculate_all_durations_per_issue(dataframe=None):
    return calculate_durations_by_first_and_last_changelog(dataframe)

def calculate_all_durations_by_starting_and_ending_status_values(dataframe=None):
    """Returns the duration for each jira task
    The duration is calculated as time between setting status
    to some value that means beginning of task for the first time
    and changing it to another value that means ending of task for the last time"""
    dataframe = changelogs_raw_dataframe_validation(dataframe)
    
    ids = get_unique_values(dataframe, TASK_ID_COLUMN_NAME)
    print("Total unique ids in changelogs: ", end='')   # debugging
    print(len(ids))     # debugging

    COLUMN_NAMES = ["issue_id","duration"]

    status_changelogs_by_task_id = get_changelogs_by_task_sorted(dataframe, ids)    # one subDataFrame per each id
    print("Total changelog subDataFrames: {}".format(len(status_changelogs_by_task_id)))    # debugging
    print("Total subdataframes with more than two changelog entries: ", end='')     # debugging
    print(count_subdataframes_with_more_than_one_changelog_entries(status_changelogs_by_task_id))   # debugging

    durations_per_issue = []
    total_failed_calculations = 0
    for id in status_changelogs_by_task_id.keys():
        #throws exceptions if the dataframe is empty dataframe. duh...
        #throws exception if the dataframe has only one value
        if not validate_changelog_dataframe_length(status_changelogs_by_task_id[id]): 
            print("Failed dataframe length validation")    # debugging
            print(status_changelogs_by_task_id[id])     # debugging
            total_failed_calculations += 1
            continue
        #throws exceptions if changelogs dont countain open, reopened, new or patch awaiting status
        if not validate_changelog_containing_status_values(status_changelogs_by_task_id[id], STARTING_STATUS_VALUES): 
            print("Starting status values not found in dataframe")    # debugging
            print(status_changelogs_by_task_id[id])     # debugging
            total_failed_calculations += 1
            continue
        #throws exceptions if changelogs dont countain closed, resolved, done or submitted
        if not validate_changelog_containing_status_values(status_changelogs_by_task_id[id], ENDING_STATUS_VALUES): 
            print("Ending status values not found in dataframe")    # debugging
            print(status_changelogs_by_task_id[id])     # debugging
            total_failed_calculations += 1
            continue

        starting_date = find_starting_date_by_status(status_changelogs_by_task_id[id])
        ending_date = find_ending_date_by_status(status_changelogs_by_task_id[id])

        if not validate_positive_duration(starting_date, ending_date): 
            print("Negative duration")    # debugging
            print(status_changelogs_by_task_id[id])     # debugging
            total_failed_calculations += 1
            continue
        duration = ending_date - starting_date

        try:
            duration = duration.tolist()[0].days    # sometimes the list is empty
        except:
            print("Failed to convert timedelta to days")    # debugging
            print(duration)     # debugging
            total_failed_calculations += 1
            continue
        durations_per_issue.append([id, duration])
        print("Calculated {} days for task_id {}".format(duration, id))     # debugging

    print("Total failed calculations of duration: {}".format(total_failed_calculations))
    ret = pandas.DataFrame(durations_per_issue, columns=COLUMN_NAMES)
    ret.info()
    return ret

def calculate_durations_by_first_and_last_changelog(dataframe=None):
    """Returns the duration for each jira task
    The duration is calculated as time between
    the first changelog and the last changelog for each task"""
    dataframe = changelogs_raw_dataframe_validation(dataframe)

    ids = get_unique_values(dataframe, TASK_ID_COLUMN_NAME)
    print("Total unique ids in changelogs: ", end='')   # debugging
    print(len(ids))     # debugging

    COLUMN_NAMES = ["issue_id","duration"]

    status_changelogs_by_task_id = get_changelogs_by_task_sorted(dataframe, ids)    # one subDataFrame per each id
    print("Total schangelogs subDataFrames: {}".format(len(status_changelogs_by_task_id)))  # debugging
    print("Total subdataframes with more than two changelog entries: ", end='')     # debugging
    print(count_subdataframes_with_more_than_one_changelog_entries(status_changelogs_by_task_id))   # debugging

    durations_per_issue = []
    total_failed_calculations = 0
    for id in status_changelogs_by_task_id.keys():
        #throws exceptions if the dataframe is empty dataframe
        #throws exception if the dataframe has only one value
        if not validate_changelog_dataframe_length(status_changelogs_by_task_id[id]): 
            print("Failed dataframe length validation")    # debugging
            print(status_changelogs_by_task_id[id])     # debugging
            total_failed_calculations += 1
            continue

        starting_date = status_changelogs_by_task_id[id]['date'].iloc[0]
        ending_date = status_changelogs_by_task_id[id]['date'].iloc[-1]

        if not validate_positive_duration(starting_date, ending_date): 
            print("Negative duration")    # debugging
            print(status_changelogs_by_task_id[id])     # debugging
            total_failed_calculations += 1
            continue
        duration = ending_date - starting_date

        # try:
        duration = duration.days #tolist()[0].days    # sometimes the list is empty
        # except:
        #     print("Failed to convert timedelta to days: ", end='')    # debugging
        #     print(duration)     # debugging
        #     print("Duration variable type: {}".format(type(duration)))
        #     total_failed_calculations += 1
        #     continue
        durations_per_issue.append([id, duration])
        print("Calculated {} days for task_id {}".format(duration, id))     # debugging
    
    print("Total failed calculations of duration: {}".format(total_failed_calculations))
    ret = pandas.DataFrame(durations_per_issue, columns=COLUMN_NAMES)
    ret.info()
    return ret

def calculate_durations_by_in_progress_duration(dataframe=None):
    """Returns the duration for each jira task
    The duration is calculated as time between setting status
    to 'In Progress' and changing it to something else"""
    dataframe = changelogs_raw_dataframe_validation(dataframe)

    COLUMN_NAMES = ["issue_id","duration"]

    started_issues = dataframe[ (dataframe["field_name"] == "status") & \
                                (dataframe["new_value"] == "In Progress")]
    completed_issues = dataframe[   (dataframe["field_name"] == "status") & \
                                    (dataframe["original_value"] == "In Progress")]
    
    durations_per_issue = []
    for i, row in started_issues.iterrows():
        issue_id = row["issue_report_id"]
        starting_date = row["date"]#.values
        ending_date = completed_issues[completed_issues["issue_report_id"] == issue_id]["date"]#.values

        if not validate_positive_duration(starting_date, ending_date): continue
        duration = ending_date - starting_date

        try:
            duration = duration.tolist()[0].days    # sometimes the list is empty
        except:
            continue

        #print(duration.name)                    # debugging
        #print(duration)                         # debugging
        #print(duration.values)                  # debugging
        #print(duration.values[0])               # debugging
        #print(duration.values[0][0].days)       # debugging
        #print(duration.tolist()[0].days)        # debugging

        #duration = duration.tolist()[0].days
        durations_per_issue.append([issue_id, duration])

    ret = pandas.DataFrame(durations_per_issue, columns=COLUMN_NAMES)
    ret.info()
    return ret

if __name__=="__main__":

    df = get_changelogs_crom_csv()
    df_filtered = get_status_changelogs(df)
    #df['date'] = df['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))
    # df.info()

    #print("Is id column unique: ", check_id_unique_constraint(df))

    changelog_sample = df[df["id"] == 308]  #df.iloc[308]
    changelog_sample2 = df[df["id"] == 328] #df.iloc[328]
    
    # print(changelog_sample)
    # print(changelog_sample2)
    
    #print(changelog_sample["date"].astype(datetime)[0])

    # print("Greater date: ", changelog_sample['date'].values[0] > changelog_sample2['date'].values[0])

    durations = calculate_all_durations_per_issue(df_filtered)
    changelog_sample3 = durations.iloc[0]['duration']
    print(changelog_sample3)
    durations.info()
    plot_durations_histogram(durations)

    filtered_durations = filter_durations_in_range(durations, -10,20)
    filtered_durations.info()
    plot_durations_histogram(filtered_durations, -1, 11)

# %%
