import numpy
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, Trainer, TrainingArguments
from make_dataset import get_jira_tasks, filter_long_descriptions, split_dataset, inputs_and_labels

class BertRegressor(torch.nn.Module):
    """Implementation of Bert Regressor Model"""
    def __init__(self, drop_rate=0.2, freeze_bert=False) -> None:
        super(BertRegressor, self).__init__()
        D_in = 768
        D_out = 1
        self.bert = BertModel.from_pretrained('bert-base')
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
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    label_tensor = torch.tensor(labels)
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

    df_dataset = get_jira_tasks()

    tokenizer = BertTokenizer.from_pretrained('bert-base')

    encoded_corpus = tokenizer(text=df_dataset.description.tolist(),
                            add_special_tokens=True,
                            padding='max_length',
                            truncation='longest_first',
                            max_length=MAX_TOKENS_PER_INPUT_SEQUENCE,
                            return_attention_mask=True)
    input_ids = encoded_corpus['input_ids']
    attention_mask = encoded_corpus['attention_mask']
    
    descriptions = filter_long_descriptions(tokenizer, df_dataset, MAX_TOKENS_PER_INPUT_SEQUENCE)

    input_ids = numpy.array(input_ids)[descriptions]
    attention_mask = numpy.array(attention_mask)[descriptions]
    labels = df_dataset.duration.to_numpy()[descriptions]

    train_set, test_set, validation_set = split_dataset(descriptions, test_set_length=.8, train_set_length=.1, validation_set_length=.1)
    train_inputs, train_labels = inputs_and_labels(train_set)
    test_inputs, test_labels = inputs_and_labels(test_set)
    validation_inputs, validation_labels = inputs_and_labels(validation_set)
                                         
    duration_scaler = MinMaxScaler(feature_range=(-1,1))
    duration_scaler.fit(train_labels)
    train_labels = duration_scaler.transform(train_labels)
    test_labels = duration_scaler.transform(test_labels)
    validation_labels = duration_scaler.transform(validation_labels)

    batch_size = 32     # what is batch size?
    train_dataloader = create_dataloader(train_inputs, train_labels, batch_size)
    test_dataloader = create_dataloader(test_inputs, test_labels, batch_size)
    validation_dataloader = create_dataloader(validation_inputs, validation_labels, batch_size)

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

    #model = run_epochs(model, optimizer, scheduler, loss_function, epochs, train_dataloader, device, clip_value=2)
    model = run_trainer(model, train_dataloader)

if __name__=="__main__":
    model = train_regressor_model()
