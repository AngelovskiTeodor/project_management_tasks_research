import pandas as pd
import openai
from api_key import get_api_key

if __name__ == '__main__':
    dataframe = pd.read_csv("./jira_dataset/jira_database_public_jira_issue_report.csv")

    #print(dataframe.head())
    #print(dataframe.info())
    #print(dataframe.describe())

    openai.api_key = get_api_key()

    for i in range(0,4):
        sample_desc = dataframe.loc[i, ['description']]
        if sample_desc.values[0] == "" or \
            sample_desc.values[0] == " " or \
                sample_desc.isna().iloc[0]:
            
            sample_title = dataframe.loc[i, ['title']].values[0]
            print("Generating description for issue with title: \"" + sample_title + "\"")
            chatgpt_question = "generate description for this title: " + sample_title
            
            chatgpt_response = openai.Completion.create(
                engine="gpt-3.5-turbo",#"text-davinci-002",
                prompt=chatgpt_question,
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.1,
            )

            generated_description = chatgpt_response.choices[0].text
            #print("The generated description is: " + generated_description)
            dataframe.loc[i, ['description']] = [generated_description]
            print("Generated Dataframe description for this title is: " \
                  + dataframe.loc[i, ['description']].values[0])

    print(dataframe.info())