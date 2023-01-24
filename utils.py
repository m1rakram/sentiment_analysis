from googletrans import Translator

def translate_to_azerbaijani(text, translator):
    
    translator.raise_Exception = True 
    translation = translator.translate(text, dest='az')
    return translation.text



