from googletrans import Translator
from sklearn.metrics import f1_score
import numpy as np

def translate_to_azerbaijani(text, translator):
    
    translator.raise_Exception = True 
    translation = translator.translate(text, dest='az')
    return translation.text



def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')


