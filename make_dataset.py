import numpy
import pandas
from data_utils import get_task_descriptions as get_descriptions, serialize_to_csv
from get_task_durations import calculate_all_durations_per_issue as get_durations, filter_durations_in_range as filter_durations, plot_histogram_of_column as plot_hist

class WeightsNotEqualToOneError(RuntimeError):
    def __init__(self, message):
        super.__init__(message)

class NegativeValueOfWeightsError(RuntimeError):
    def __init__(self, message="The value of the weights can not be a negative number"):
        super.__init__(message)

def get_jira_tasks():
    """Returns dataframe with description and duration for the tasks"""
    df_desc = get_descriptions()
    df_durations = get_durations()
    print("All durations dataframe") # debugging
    print(df_durations) # debugging
    MINUMUM_DURATION = 1
    MAXIMUM_DURATION = 10
    df_durations = filter_durations(df_durations, MINUMUM_DURATION, MAXIMUM_DURATION)
    print("Filtered durations dataframe:")  # debugging
    print(df_durations) # debugging
    df_durations.info() # debugging

    column_names = ["description", "duration"]
    dataset = []

    for i, row in df_durations.iterrows():
        id = row['issue_id']

        # if id not in df_desc['id'].values.tolist():
        #     print("ID {} is not paired with any description".format(id))    # debugging
        #     continue

        duration = row["duration"]
        desc = df_desc[df_desc["id"] == id]
        try:
            print("Before getting description as a string:") # debugging
            print(desc) # debugging
            desc = desc.iat[0,1]    #.at[0,'description']
            print("After getting description as a string:") # debugging
            print(desc) # debugging
        except IndexError as err:
            # print("=== EXCEPTION ===")
            # print(err)
            # print(desc)
            # print(id)
            # print(df_desc['id'])
            # print(id not in df_desc['id'])  # debugging
            continue

        # print("Description for row number {}: {}".format(i, desc))  # debugging

        dataset.append([desc, duration])
    dataset = pandas.DataFrame(dataset, columns=column_names)
    print(serialize_to_csv(dataset))
    return dataset

def filter_long_descriptions(tokenizer, df_dataset:pandas.DataFrame, max_tokens_per_input_sequence=256):
    """Removes tasks with description number of tokens morte than max_tokens"""
    filtered_dataset = []
    dataset = df_dataset.values.tolist()
    number_of_tokens_per_description = tokenizer(dataset, padding=False, truncation=False, return_length=True)['length']
    for num_of_tokens, row in zip(number_of_tokens_per_description, dataset):
        if num_of_tokens > max_tokens_per_input_sequence:
            filtered_dataset.append(row)
    filtered_dataframe = pandas.DataFrame(filtered_dataset,columns=df_dataset.columns)
    #filtered_dataframe.info()
    #plot_hist(pandas.DataFrame(number_of_tokens_per_description,columns=['token_length']), 'token_length')
    return filtered_dataframe

def split_dataset(dataset, train_set_length=.8, test_set_length=.1, validation_set_length=None, axis=1):
    """Splits the dataset into three datasets: train, test and validation"""
    if train_set_length<0 or test_set_length<0 or validation_set_length<0:
        raise NegativeValueOfWeightsError()
    
    weights_sum = train_set_length + test_set_length + validation_set_length if validation_set_length is not None else 0
    if weights_sum > 1:
        raise WeightsNotEqualToOneError("The weight of the splitted sets cannot exceed 1")
    
    first_breaking_point = int(len(dataset)*train_set_length)
    second_breaking_point = first_breaking_point + int(len(dataset)*test_set_length) +1
    train_set, test_set, validation_set = numpy.split(dataset, [first_breaking_point, second_breaking_point], axis=axis)
    return train_set, test_set, validation_set

def inputs_masks_labels(dataset, inputs_column_name="input", masks_column_name="mask", labels_column_name="label"):
    """Splits the columns of the dataset
    Returns two numpy arrays: inputs and labels"""
    inputs = numpy.array(dataset[inputs_column_name])
    masks = numpy.array(dataset[masks_column_name])
    labels = numpy.array(dataset[labels_column_name])
    return inputs, masks, labels

if __name__=="__main__":
    df = get_jira_tasks()
    print("Descriptions and durations:")
    df.info()

    print("Description example: ")
    df.iat[0,1]
    