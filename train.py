from dataset.read_csv import review_data
from transformers import BertForSequenceClassification
from models.sentiment_classifier import SentimentClassifier
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import f1_score_func


import os
import torch
import tqdm
import random
import numpy as np
import torch.nn as nn

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

num_of_classes = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, " is available")



def eval(model, dataloader_val, loss, device, writer, epoch):
    model.eval()
    val_iterator = enumerate(dataloader_val)
    loss_total = 0
    tq = tqdm.tqdm(total = len(dataloader_val))
    tq.set_description('Validation ')
    correct_predictions = 0
    f1= 0
    with torch.no_grad():
        for _, batch in val_iterator:
                input_ids, attention_masks, label = batch

                outputs = model(input_ids= input_ids.to(device),
                                attention_mask = attention_masks.to(device))
                
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == label.to(device))
                #f1+= f1_score_func(preds.cpu(), label.cpu())
                
                loss_i = loss(outputs, label.to(device))
                loss_total+= loss_i.item()
                tq.set_postfix(loss='%.6f' % loss_i.item())
                tq.update()

    tq.close()
    writer.add_scalar("Loss val", loss_total/len(dataloader_val), epoch)
    writer.add_scalar("acc val", correct_predictions/len(dataloader_val), epoch)
    writer.add_scalar("f1 val", f1/len(dataloader_val), epoch)
    return




def train(model, dataloader_train, dataloader_val, loss, optimizer, device, scheduler, max_epoch):
    

    writer  = SummaryWriter(comment="Sentiment_Azerbaijani_full")
    
    for epoch in range(max_epoch):
        model.train()
        train_iterator = enumerate(dataloader_train)

        loss_total = 0
        tq = tqdm.tqdm(total = len(dataloader_train))
        tq.set_description('epoch %d' % (epoch))
        correct_predictions = 0
        f1= 0
        
        for _, batch in train_iterator:
            input_ids, attention_masks, label = batch

            #input_ids = input_ids.squeeze(0)
            #attention_masks = attention_masks.squeeze(0)
            outputs = model(input_ids= input_ids.to(device),
                            attention_mask = attention_masks.to(device))
            
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == label.to(device))
            #f1+= f1_score_func(preds.cpu(), label.cpu())
            
            loss_i = loss(outputs, label.to(device))
            loss_total+= loss_i.item()
            loss_i.backward()
        
            # Gradient Descent
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            tq.set_postfix(loss='%.6f' % loss_i.item())
            tq.update()

        tq.close()
        writer.add_scalar("Loss train", loss_total/len(dataloader_train), epoch)
        writer.add_scalar("acc train", correct_predictions/len(dataloader_train), epoch)
        writer.add_scalar("f1 train", f1/len(dataloader_train), epoch)


        eval(model, dataloader_val, loss, device, writer, epoch)

        checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

        torch.save(checkpoint, os.path.join("models/", 'latest_model_aze_full.pth'))
        print("saving the model " )

        
        







train_data = review_data("train")
val_data = review_data("val")

dataloader_train = DataLoader(
    train_data,
    shuffle=True,
    batch_size=4,
    drop_last=True,
    num_workers=4
)

dataloader_val = DataLoader(
    val_data,
    batch_size=1
)



#print(BertForSequenceClassification.__doc__)

# model = BertForSequenceClassification.from_pretrained(
#                                       'bert-base-uncased', 
#                                       num_labels = 2,
#                                       output_attentions = False,
#                                       output_hidden_states = False
#                                      ).to(device)

model= SentimentClassifier(2).to(device)


optimizer = AdamW(model.parameters(), lr = 1e-5, eps = 1e-8)

epochs = 10

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps = len(dataloader_train)*epochs
)


loss_fn = nn.CrossEntropyLoss().to(device)

train(model, dataloader_train, dataloader_val, loss_fn, optimizer, device, scheduler, epochs )

