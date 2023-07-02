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
        durations_per_issue.append([issue_id, duration])

    ret = pandas.DataFrame(durations_per_issue, columns=column_names)
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