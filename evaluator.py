import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

def print_metrics(metrics, label_builder):
    print("Accuracy =",metrics["accuracy"])
    labels = []
    for i, (p, r, f, s) in enumerate(zip(metrics["precision"].tolist(),metrics["recall"].tolist(), metrics["fscore"].tolist(), metrics["support"].tolist())):
        label = label_builder.get_label(i)
        print("for class {}, precision = {}, recall = {}, fscore = {}, support = {}".format(label,p,r,f,s))
        labels.append(label)
    cm = metrics["confusion_matrix"]
    n_rows, n_cols = cm.shape
    for r in range(n_rows):
        print(f"{labels[r]}", end=': ')
        for c in range(n_cols):
           print(f"{labels[c]}: {cm[r,c]}", end='; ')
        print()        
    #disp = ConfusionMatrixDisplay(cm, display_labels=labels)
 
def compute_metrics(true_label_list, pred_list):
   acc = accuracy_score(true_label_list, pred_list)
   precision, recall, fscore, support = precision_recall_fscore_support(true_label_list, pred_list, average=None) 
   cm = confusion_matrix(true_label_list, pred_list)
   
   return {
            "accuracy" : acc,
            "confusion_matrix": cm,
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
            "support": support
            }
   
   
def evaluate(valid_dataloader, model, loss_fn=None, mode="classifier", device="cpu"):
   pred_list = []
   true_label_list = []  
   loss = 0
   model.eval()
   for step, (x,label) in enumerate(valid_dataloader):
      with torch.no_grad():
         
         x = x.to(device)
         x = torch.unsqueeze(x, dim = 1) #need to unsqeeze for classifier training
         label = label.to(device)
         if mode == "classifier":
            logits = model(x)
         elif mode == "supcon":
            logits = model(x,contrastive=False)
         preds = torch.argmax(logits, dim=1)
         true_labels = torch.argmax(label, dim=1)
         pred_list += preds.tolist()
         true_label_list += true_labels.tolist()
                 
         loss += loss_fn(logits, label)
         
   
   metrics = compute_metrics(true_label_list, pred_list)  
   avg_loss = loss/(step+1)
   
   return avg_loss, metrics
      
def compute_and_display(true_labels, pred_labels, label_builder):
    metrics = compute_metrics(true_labels, pred_labels)
    print_metrics(metrics, label_builder)
