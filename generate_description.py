import sys
import pandas as pd
import openai
from time import sleep as wait
from api_key import get_api_key
from data_utils import get_jira_tasks_from_csv as get_csv_data
from get_titles_with_missing_descriptions import get_titles_with_missing_descrtiptions as get_db_data
from persistence import actions as db_actions

def get_gpt_message_object(question, role="user"):
    """OpenAI API requires specific API request format."""
    message_object = {"role":role,"content":question}
    return message_object

def run_script_for_minutes(minutes):
    """OpenAI API accepts 3 requests per minute.
    This method assumes that it takes 3 requests per minute
    This method returns the number of requests that can be run in
    the number of minutes provided as parameter
    :param minutes: Number of minutes that you wish this script to be running
    """
    if minutes:
        REQUESTS_PER_MINUTE = 3
        total_number_of_requests = minutes * REQUESTS_PER_MINUTE
        return total_number_of_requests
    else:
        return None

def execute_api_request(title, model="gpt-3.5-turbo"):
    """This method sends API request to generate description for a single title
    """
    gpt_question = "This is a title for a Jira task:\n{} \
        \n Write a description in the style of a software developer for this title.".format(title)

    gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                get_gpt_message_object(gpt_question)
            ]
        )
    generated_description = gpt_response["choices"][0]["message"]["content"]
    return generated_description
    
def generate_description(database_connection, jira_issue_id, jira_issue_title):
    """Generates description and saves it into database"""
    generated_description = execute_api_request(jira_issue_title)
    print("Generated description for title {} is: {}".format(jira_issue_title, generated_description))
    
    database_input_values = (jira_issue_id, jira_issue_title, generated_description)
    db_actions.update_description(database_connection, jira_issue_title, generated_description)

    return generated_description
    
def iterate_description_generation(database_connection, dataframe):
    """Generates descriptions for titles provided in pandas dataframe as argument"""
    openai.api_key = get_api_key()
    for index, row in dataframe.iterrows():
        row.description = generate_description(database_connection, row.id, row.title)
        wait(20.01)     # wait 20 seconds before another request
    return dataframe

def run_description_generation(database_connection, duration_in_minutes=None):
    """Generates descriptions using gpt-3.5-turbo for duration provided as argument or indefinitely"""
    dataframe = get_db_data(db, run_script_for_minutes(duration_in_minutes))
    generated_descriptions =  iterate_description_generation(database_connection, dataframe)
    return generated_descriptions

if __name__ == '__main__':
    number_of_minutes = None
    cli_arguments = sys.argv
    if len(cli_arguments) > 1:
        if cli_arguments[1].isdigit():
            number_of_minutes = int(cli_arguments[1])

    db = db_actions.get_db_connection()
    
    #print(dataframe.head())
    #print(dataframe.info())
    #print(dataframe.describe())

    if number_of_minutes is not None:
        dataframe = run_description_generation(db, number_of_minutes)
    else:
        dataframe = run_description_generation(db)

    print(dataframe.info())
