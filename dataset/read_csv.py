import torch
import pandas as pd
import csv
from transformers import BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer


tfidf = TfidfVectorizer(ngram_range=(1,5), max_features=5000)

csv_path = "dataset/full_translated_imdb.csv"

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)
#print(tokenizer.batch_encode_plus.__doc__)

class review_data(torch.utils.data.Dataset):
    def __init__(self, mode, language = "az", csv_path = csv_path):
        self.mode = mode
        with open(csv_path, "r", encoding = "utf8") as file:

            data = pd.read_csv(file) 

        if(language == "az"):
            self.samples = data["az_review"].to_list()

        else:
            self.samples = data["eng_review"].to_list()


        self.labels = self.convert_binary(data["label"].to_list())

        test_list = []
        test_label = []
        train_list = []
        train_label = []
        val_list = []
        val_label = []

        

        for i in range(len(self.samples)):
            if (i%25==0):
                test_list.append(self.samples[i])
                test_label.append(self.labels[i])
            elif(i%33 == 0):
                val_list.append(self.samples[i])
                val_label.append(self.labels[i])
            else:
                train_list.append(self.samples[i])
                train_label.append(self.labels[i])


        if(self.mode == "train"):
            self.list = train_list
            self.label = train_label
        elif(self.mode =="test"):
            self.list = test_list
            self.label = test_label
        else:
            self.list = val_list
            self.label = val_label

        print(sum(self.label))
        

        

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        
        encoded_data_train = tokenizer.encode_plus(
            self.list[idx],
            add_special_tokens=True,
            return_attention_mask=True,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt')
                
        input_ids = encoded_data_train['input_ids']
        attention_mask = encoded_data_train['attention_mask']

        #print(input_ids.shape, attention_mask.shape)

        return input_ids.squeeze(), attention_mask.squeeze(), torch.tensor(self.label[idx])
        

    def convert_binary(self, label):
        for i in range(len(label)):
            if(label[i] == "negative"):
                label[i] = 0
            else:
                label[i] = 1
        return label



class kapital(torch.utils.data.Dataset):
    def __init__(self, mode, language = "az", csv_path = "dataset/kapital.csv"):
        self.mode = mode
        with open(csv_path, "r", encoding = "utf8") as file:

            data = pd.read_csv(file) 

        print(data.info())
        data = data[data['Sentment'].notna()]
        data =  data.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
        data = data[data['text'] !=  ""]
        data = data[data['Sentment'] !=  "Pozitiv"]

        self.samples = data["text"].to_list()
        

        

        self.labels = self.convert_binary(data["Sentment"].to_list())

        test_list = []
        test_label = []
        train_list = []
        train_label = []
        val_list = []
        val_label = []

        

        for i in range(len(self.samples)):
            if (i%25==0):
                test_list.append(self.samples[i])
                test_label.append(self.labels[i])
            elif(i%33 == 0):
                val_list.append(self.samples[i])
                val_label.append(self.labels[i])
            else:
                train_list.append(self.samples[i])
                train_label.append(self.labels[i])


        if(self.mode == "train"):
            self.list = train_list
            self.label = train_label
        elif(self.mode =="test"):
            self.list = test_list
            self.label = test_label
        else:
            self.list = val_list
            self.label = val_label

        print(sum(self.label), len(self.label))
        

        

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        
        encoded_data_train = tokenizer.encode_plus(
            self.list[idx],
            add_special_tokens=True,
            return_attention_mask=True,
            padding='max_length',
            max_length=100,
            truncation=True,
            return_tensors='pt')
                
        input_ids = encoded_data_train['input_ids']
        attention_mask = encoded_data_train['attention_mask']

        #print(input_ids.shape, attention_mask.shape)

        return input_ids.squeeze(), attention_mask.squeeze(), torch.tensor(self.label[idx])
        

    def convert_binary(self, label):

        count0 = 0
        count1 = 0
        count2 = 0
        for i in range(len(label)):
            if(label[i] == "Neytral"):
                label[i] = 1
                count0+=1
            elif(label[i] == "Pozitiv"):
                count1+=1
                label[i] = 1
            else:
                count2+=1
                label[i]=0
        print(count2, count1, count0)
        return label



class kapital_mixed(torch.utils.data.Dataset):
    def __init__(self, mode, language = "az", csv_path = "dataset/kapital.csv"):
        self.mode = mode
        with open(csv_path, "r", encoding = "utf8") as file:

            data = pd.read_csv(file) 

        data = data[data['Sentment'].notna()]
        data =  data.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
        data = data[data['text'] !=  ""]

        self.samples = data["text"].to_list()
        self.labels = data["Sentment"].to_list()


        csv_path = "dataset/new_add.csv"
        with open(csv_path, "r", encoding = "utf8") as file:

            data = pd.read_csv(file) 


        self.samples.extend(data["az_review"].to_list())
        self.labels.extend(data["label"].to_list())

        self.labels = self.convert_binary(self.labels)

        test_list = []
        test_label = []
        train_list = []
        train_label = []
        val_list = []
        val_label = []
        

        for i in range(len(self.samples)):
            if (i%25==0):
                test_list.append(self.samples[i])
                test_label.append(self.labels[i])
            elif(i%33 == 0):
                val_list.append(self.samples[i])
                val_label.append(self.labels[i])
            else:
                train_list.append(self.samples[i])
                train_label.append(self.labels[i])


        if(self.mode == "train"):
            self.list = train_list
            self.label = train_label
        elif(self.mode =="test"):
            self.list = test_list
            self.label = test_label
        else:
            self.list = val_list
            self.label = val_label

        

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        
        encoded_data_train = tokenizer.encode_plus(
            self.list[idx],
            add_special_tokens=True,
            return_attention_mask=True,
            padding='max_length',
            max_length=100,
            truncation=True,
            return_tensors='pt')
                
        input_ids = encoded_data_train['input_ids']
        attention_mask = encoded_data_train['attention_mask']

        #print(input_ids.shape, attention_mask.shape)

        return input_ids.squeeze(), attention_mask.squeeze(), torch.tensor(self.label[idx])

        # return torch.tensor(self.samples[idx]), torch.tensor(self.label[idx])
        

    def convert_binary(self, label):

        count0 = 0
        count1 = 0
        count2 = 0
        for i in range(len(label)):
            if(label[i] == "Neytral"):
                label[i] = 0
                count0+=1
            elif(label[i] == "Pozitiv" or label[i] == "positive"):
                count1+=1
                label[i] = 1
            else:
                count2+=1
                label[i]=2
        print(count2, count1, count0)
        return label


    
kapital_mixed("test")
        