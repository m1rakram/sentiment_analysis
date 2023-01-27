import torch
import pandas as pd
import csv


csv_path = "dataset/half_translated_imdb.csv"

class review_data(torch.utils.data.Dataset):
    def __init__(self, mode, csv_path = csv_path):
        self.mode = mode
        with open(csv_path, "r", encoding = "utf8") as file:

            data = pd.read_csv(file) 


        self.samples = data["eng_review"].to_list()
        self.labels = self.convert_binary(data["label"].to_list())
        print(sum(self.labels))
        test_list = []
        train_list = []
        val_list = []
        for i in range(len(self.samples)):
            if (i%25==0):
                test_list.append([self.samples[i], self.labels[i]])
            elif(i%33 == 0):
                val_list.append([self.samples[i], self.labels[i]])
            else:
                train_list.append([self.samples[i], self.labels[i]])


        if(self.mode == "train"):
            self.list = train_list
        elif(self.mode =="test"):
            self.list = test_list
        else:
            self.list = val_list

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        
        return 
        

    def convert_binary(self, label):
        for i in range(len(label)):
            if(label[i] == "negative"):
                label[i] = 0
            else:
                label[i] = 1
        return label




t = review_data("train")

        