import pandas as pd
import openai
from api_key import get_api_key
from data_utils import get_data_from_csv as get_csv_data
from get_titles_with_missing_descriptions import get_titles_with_missing_descrtiptions as get_db_data
from persistence import actions as db_actions

def get_minutes_from_cli_arguments():
    """If provided, gets and parses the first argument from command line as number of minutes"""
    cli_arguments = sys.argv
    try:
        number_of_minutes = int(cli_arguments[1]) if len(cli_arguments) > 1 else None
    except Exception as err:
        print(err)
        number_of_minutes = None
    return number_of_minutes

def get_gpt_message_object(title, role="user"):
    """OpenAI API requires specific API request format."""
    gpt_question = "This is a title for a Jira task:\n{} \
        \n Write a description in the style of a software developer for this title.".format(title)
    message_object = {"role":role,"content":gpt_question}
    return message_object

def number_of_requests_for_minutes(minutes):
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
    """This method sends API request to generate description for a single title"""
    gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                get_gpt_message_object(title)
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
    TIME_BETWEEN_REQUESTS = 20.01
    openai.api_key = get_api_key()
    for index, row in dataframe.iterrows():
        try:
            row.description = generate_description(database_connection, row.id, row.title)
        except Exception as ex:
            print(ex)
        wait(TIME_BETWEEN_REQUESTS)     # wait 20 seconds before another request
    return dataframe

def run_description_generation(database_connection, duration_in_minutes=None):
    """Generates descriptions using gpt-3.5-turbo for duration provided as argument or indefinitely"""
    dataframe = get_db_data(db, number_of_requests_for_minutes(duration_in_minutes))
    return iterate_description_generation(database_connection, dataframe)

if __name__ == '__main__':
    number_of_minutes = get_minutes_from_cli_arguments()
    db = db_actions.get_db_connection()
    
    #print(dataframe.head())
    #print(dataframe.info())
    #print(dataframe.describe())

    dataframe = run_description_generation(db, number_of_minutes)

    print(dataframe.info())