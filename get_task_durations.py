# %%

import pandas
from datetime import datetime
from check_id_constraint import check_id_unique_constraint

def get_changelogs_crom_csv(csv_file="./jira_dataset/jira_database_public_jira_issue_changelog_item.csv"):
    """Returns Pandas Dataframe with changelog data for jira issues"""
    dataframe = pandas.read_csv(csv_file, parse_dates=['date'], date_format="%Y-%m-%d %H:%M:%S.%f")#, index_col="id")
    return dataframe

def get_status_changelogs(dataframe):
    """The changelogs data contains logs for all column
    This method filters logs that apply to the 'status' column"""
    column_name = "status"
    return dataframe[ dataframe["field_name"] == column_name ]

def filter_durations_in_range(dataframe, min=1, max=10, column_name='duration'):
    """Removes all tasks with duration less than min and greater than max"""
    ret = dataframe[
        (dataframe[column_name] >= min) & \
        (dataframe[column_name] <= max)
    ]
    return ret

def plot_durations_histogram(dataframe, min=None, max=None, column_name='duration'):
    """Plots histogram for the task durations
    param dataframe: Pandas Dataframe containing task durations
    param min: Minimal value for x-axis
    param max: Maximum value for x-axis
    """
    if min==None or max==None:
        dataframe.hist(column=column_name, grid=True, edgecolor='black', align='left')
    else:
        dataframe.hist(column=column_name, grid=True, range=(min, max), edgecolor='black', align='left', bins=max-min)

def get_durations_per_issue(dataframe):
    """Returns the duration for each jira task
    The duration is defined as time between setting status
    to 'In Progress' and changing it to something else"""
    column_names = ["issue_id","duration"]
    started_issues = dataframe[ (dataframe["field_name"] == "status") & \
                                (dataframe["new_value"] == "In Progress")]
    completed_issues = dataframe[   (dataframe["field_name"] == "status") & \
                                    (dataframe["original_value"] == "In Progress")]
    
    durations_per_issue = []
    for i, row in started_issues.iterrows():
        issue_id = row["issue_report_id"]
        starting_date = row["date"]#.values
        ending_date = completed_issues[completed_issues["issue_report_id"] == issue_id]["date"]#.values
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

    ret = pandas.DataFrame(durations_per_issue, columns=column_names)
    ret.info()
    return ret

if __name__=="__main__":

    df = get_changelogs_crom_csv()
    df_filtered = get_status_changelogs(df)
    #df['date'] = df['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))
    df.info()

    #print("Is id column unique: ", check_id_unique_constraint(df))

    changelog_sample = df[df["id"] == 308]  #df.iloc[308]
    changelog_sample2 = df[df["id"] == 328] #df.iloc[328]
    
    print(changelog_sample)
    print(changelog_sample2)
    
    #print(changelog_sample["date"].astype(datetime)[0])

    print("Greater date: ", changelog_sample['date'].values[0] > changelog_sample2['date'].values[0])

    durations = get_durations_per_issue(df_filtered)
    changelog_sample3 = durations.iloc[0]['duration']
    print(changelog_sample3)
    durations.info()
    plot_durations_histogram(durations)

    filtered_durations = filter_durations_in_range(durations, -10,20)
    filtered_durations.info()
    plot_durations_histogram(filtered_durations, -1, 11)

# %%
