from googletrans import Translator
from sklearn.metrics import f1_score
import numpy as np
from transformers import BertTokenizer

def translate_to_azerbaijani(text, translator):
    
    translator.raise_Exception = True 
    translation = translator.translate(text, dest='az')
    return translation.text



def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')


def test_data_tokenizer(review):
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True
    )

    encoded_data_train = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            return_attention_mask=True,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt')
    
    input_ids = encoded_data_train['input_ids']
    attention_mask = encoded_data_train['attention_mask']

    #print(input_ids.shape, attention_mask.shape)

    return input_ids.squeeze(), attention_mask.squeeze()


def label_interpreter(prediction):
    if(prediction==0):
        print("Negative")
    else:
        print("positive")