import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(model_path, num_of_labels):
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_of_labels)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)

    # Set the device to use for inference
    model.to(device)
    print('DEVICE:', device)

    return model, tokenizer

def apply_BERT_model(df, input_col, pred_col, model, tokenizer,
                     batch_size=32, max_seq_length=128):
    """
    Apply a fine-tuned BERT model to a DataFrame column of text and classify each row based on the model's prediction.
    Load the fine-tuned model's weights and configuration, create the classifier, apply the function to the 'text_column'
    returns: pandas.DataFrame: The df with the specified text column replaced by the predicted labels for each row.
    """

    # Split the DataFrame into batches
    batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]

    # Classify each batch of rows using the model and tokenizer
    results = []
    for batch in batches:
        # print('Assigning category for this input:  ', input_col)

        texts = batch[input_col].tolist()
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            labels = logits.argmax(dim=1).tolist()

        # Map the labels to the corresponding string values
        LABEL_MAP = {
            0: "NEGATIVE IMPACT",
            1: "NO IMPACT",
            2: "POSITIVE IMPACT",
            3: "NO INFO ABOUT RELATION",
        }
        for label in labels:
            results.append(LABEL_MAP.get(label))
    df[pred_col] = pd.Series(results, dtype='category').tolist()

    return df
