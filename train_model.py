import numpy
import torch
from data_utils import progress_bar
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, Trainer, TrainingArguments
from make_dataset import get_jira_tasks as get_dataset, filter_long_descriptions, split_dataset, inputs_masks_labels
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape, mean_squared_error as mse, median_absolute_error as mdae

class BertRegressor(torch.nn.Module):
    """Implementation of Bert Regressor Model"""
    def __init__(self, drop_rate=0.2, freeze_bert=False) -> None:
        super(BertRegressor, self).__init__()
        D_in = 768
        D_out = 1
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.regressor = torch.nn.Sequential(
            torch.nn.Dropout(drop_rate),
            torch.nn.Linear(D_in, D_out)
        )
    
    def forward(self, input_ids, attention_masks):
        outputs = self.bert(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs


def create_dataloader(inputs, masks, labels, batch_size):
    """Returns dataloader"""
    input_tensor = torch.tensor(inputs.tolist())
    mask_tensor = torch.tensor(masks.tolist())
    label_tensor = torch.tensor(labels.tolist())
    dataset = TensorDataset(input_tensor, mask_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def run_epochs(model, optimizer, scheduler, loss_function, epochs, train_dataloader, device, clip_value=2):
    """Trains the model by executing each epoch"""
    for epoch in range(epochs):
        print("Epoch No. {}".format(epoch))
        best_loss=1e10
        model.train()
        for step, batch in enumerate(train_dataloader):
            print("Step No. {} of epoch no. {}".format(step, epoch))
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
            model.zero_grad()
            outputs = model(batch_inputs, batch_masks)
            loss = loss_function(outputs.squeeze(), batch_inputs.squeeze())
            loss.backward()
            clip_grad_norm(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
    return model

def run_trainer(model, train_dataloader, output_directory='outputs/test_trainer'):
    """Trains the model using Trainer class from transformers module"""
    training_arguments = TrainingArguments(output_dir=output_directory)
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataloader
    )
    trainer.train()

def train_regressor_model():
    """Returns trained model for estimating task duration"""
    MAX_TOKENS_PER_INPUT_SEQUENCE=64    # 64 for development, 256 in release

    df_dataset = get_dataset()

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    encoded_corpus = tokenizer(text=df_dataset.description.values.tolist(),
                            add_special_tokens=True,
                            padding='max_length',
                            truncation='longest_first',
                            max_length=MAX_TOKENS_PER_INPUT_SEQUENCE,
                            return_attention_mask=True)
    input_ids = encoded_corpus['input_ids']
    attention_mask = encoded_corpus['attention_mask']
    
    descriptions = df_dataset[['description']] #descriptions = filter_long_descriptions(tokenizer, df_dataset['description'], MAX_TOKENS_PER_INPUT_SEQUENCE)   # needs to be debugged

    input_ids = numpy.array(input_ids)[descriptions]
    attention_mask = numpy.array(attention_mask)[descriptions]
    labels = df_dataset.duration.to_numpy()#[descriptions]

    import pandas as pd
    import numpy as np
    transformed_dataset = pd.DataFrame({'input':input_ids, 'mask':attention_mask, 'label':labels})
    train_set, test_set, validation_set = split_dataset(transformed_dataset, train_set_length=.8, test_set_length=.1, validation_set_length=.1, axis=0)
    train_inputs, train_masks, train_labels = inputs_masks_labels(train_set)
    test_inputs, test_masks, test_labels = inputs_masks_labels(test_set)
    validation_inputs, validation_masks, validation_labels = inputs_masks_labels(validation_set)
    
    duration_scaler = MinMaxScaler(feature_range=(-1,1))
    duration_scaler.fit(train_labels)
    train_labels = duration_scaler.transform(train_labels)
    test_labels = duration_scaler.transform(test_labels)
    validation_labels = duration_scaler.transform(validation_labels)

    batch_size = 32
    train_dataloader = create_dataloader(inputs=train_inputs, masks=train_masks, labels=train_labels, batch_size=batch_size)
    test_dataloader = create_dataloader(inputs=test_inputs, masks=test_masks, labels=test_labels, batch_size=batch_size)
    validation_dataloader = create_dataloader(inputs=validation_inputs, masks=validation_masks, labels=validation_labels, batch_size=batch_size)

    model = BertRegressor(drop_rate=0.2)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("GPU is not available. CPU will be used to train the model")
        device = torch.device("cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-1)
    epochs = 5
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_function = torch.nn.MSELoss()

    train_set, test_set, validation_set = split_dataset(df_dataset, train_set_length=.8, test_set_length=.1, validation_set_length=.1, axis=0)

    #model = run_epochs(model, optimizer, scheduler, loss_function, epochs, train_dataloader, device, clip_value=2)
    model = run_trainer(model, train_set)

def predict_durations_for_inputs_unscaled(model, inputs, device):
    """Returns list of predicted values for each description in inputs argument"""
    # do i need masks for dataloader
    model.eval()
    output = []
    batch_size = 32
    dataloader = create_dataloader(inputs, batch_size=batch_size)
    for batch in dataloader:
        batch_inputs = (b.to(device) for b in batch)
        with torch.no_grad():
            output += model(batch_inputs).view(1,-1).tolist()[0]
    return output

def predict_tokenized_tensor_batch(model, input_ids, input_masks):
    """Predicts durations for subset of tokenized ids and masks of type torch.Tensor"""
    predicted_durations = model.forward(input_ids, input_masks)     #   loaded_model.forward requires arguments of type torch.Tensor
    predicted_durations = softmax(predicted_durations)
    return predicted_durations  

def predict_durations_for_tokenized_tensor_inputs(model, input_ids, input_masks, batch_size=16):
    """Predicts the durations for provided tokenized ids and masks of type torch.Tensor"""
    if (input_ids.size(dim=0) != input_masks.size(dim=0)):
        raise RuntimeError("input_ids.size ({}) and input_masks.size ({}) do not match".format(input_ids.size(dim=0), input_masks.size(dim=0)))
    
    input_size = input_ids.size(dim=0)
    model.eval()
    
    if input_size <= batch_size:
        predicted_durations = predict_tokenized_tensor_batch(model, input_ids, input_masks)
        return predicted_durations
    
    #bar = progress_bar(input_size//batch_size).start()
    predict_durations = []
    for i in range(input_size//batch_size):
        subset_ids = input_ids[i*batch_size:(i+1)*batch_size]
        subset_masks = input_masks[i*batch_size:(i+1)*batch_size]
        predicted_subset = predict_tokenized_tensor_batch(model, subset_ids, subset_masks)
        predict_durations.extend(predicted_subset)
        #bar.update(i)
    if input_size%batch_size != 0:
        subset_ids = input_ids[(input_size//batch_size)*batch_size:]
        subset_masks = input_masks[(input_size//batch_size)*batch_size:]
        predicted_subset = predict_tokenized_tensor_batch(model, subset_ids, subset_masks)
        predict_durations.extend(predicted_subset)
    return predict_durations

def predict_durations_for_inputs(model, inputs, scaler):
    """Returns list of predicted durations for each description in inputs argument"""
    predicted_values = predict_durations_for_inputs_unscaled(model, inputs)
    predicted_durations = scaler.inverse_transform(predicted_values)
    return predicted_durations

def validate(true_values, predicted_values):
    """Calculates metrics (MAE, MAPE, MSE, MDAE) for provided predicted and true values"""
    metrics = {
        "mean_absolute_error": mae(true_values, predicted_values),
        "mean_absolute_percentage_error": mape(true_values, predicted_values),
        "mean_squared_error": mse(true_values, predicted_values),
        "median_absolute_error": mdae(true_values, predicted_values)
    }
    return metrics

def run_validation(model, validation_inputs, validation_labels, scaler):
    """Makes predictions and calculates metrics: MAE, MAPE, MSE, MDAE"""
    """TODO: add softmax"""
    predicted_durations = predict_durations_for_inputs_unscaled(model, validation_inputs)
    metrics = validate(validation_labels, predicted_durations)
    return metrics

def get_max_index(array):
    """Finds the maximum value in an array and returns its index"""
    max_index = 0
    max_value = array[max_index]
    for elem, i in zip(array, range(0,len(array))):
        if elem > max_value:
            max_value = elem
            max_index = i
    for elem, i in zip(array, range(0, len(array))):
        if i == max_index:
            continue
        if elem == max_value:
            raise RuntimeError("The array has two max values")
    return max_index
    
def softmax(sequence_classifier_output, softmax_function=numpy.argmax):
    """Softmax function that converts raw model output to predicted class/value"""
    predicted_values = []
    logits = sequence_classifier_output.logits
    logits = logits.tolist()
    for logit in logits:
        predicted_value = softmax_function(logit)
        predicted_values.append(predicted_value)
    return predicted_values

if __name__=="__main__":
    model = train_regressor_model()
