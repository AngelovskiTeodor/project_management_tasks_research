import pandas as pd
import openai
from time import sleep as wait
from api_key import get_api_key
from data_utils import get_data_from_csv as get_csv_data
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
    REQUESTS_PER_MINUTE = 3
    total_number_of_requests = minutes * REQUESTS_PER_MINUTE
    return total_number_of_requests

if __name__ == '__main__':
    db = db_actions.get_db_connection()
    dataframe = get_db_data(db, run_script_for_minutes(2))

    #print(dataframe.head())
    #print(dataframe.info())
    #print(dataframe.describe())

    openai.api_key = get_api_key()

    for index, row in dataframe.iterrows():

        gpt_question = "This is a title for a Jira task:\n{} \
            \n Write a description in the style of a software developer for this title.".format(row.title)

        gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                get_gpt_message_object(gpt_question)
            ]
        )
        generated_description = gpt_response["choices"][0]["message"]["content"]
        
        print("Generated description for title {} is: {}".format(row.title, generated_description))
        
        row.description = generated_description
        database_input_values = (row.id, row.title, generated_description)
        db_actions.update_description(db,row.title,generated_description)

        wait(20.01)     # wait 20 seconds before another request

    print(dataframe.info())