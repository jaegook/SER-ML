import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

def print_evaluation_metrics(acc, cm, precision, recall, fscore, support, label_builder):
    print("Accuracy =",acc)
    labels = []
    for i, (p, r, f, s) in enumerate(zip(precision.tolist(),recall.tolist(), fscore.tolist(), support.tolist())):
        label = label_builder.get_label(i)
        print("for class {}, precision = {}, recall = {}, fscore = {}, support = {}".format(label,p,r,f,s))
        labels.append(label)
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
   #returns tuple->(precision,recall,fscore, support) -> p,r,f,s are numpy arrays with value for each class 0-5, ..
   return acc, cm, precision, recall, fscore, support
    
def evaluate(valid_dataloader, model, loss_fn=None, device="cpu"):
   pred_list = []
   true_label_list = []  
   loss = 0
   model.eval()
   for step, (x,label) in enumerate(valid_dataloader):
      with torch.no_grad():
         #print("step =", step)
         x = x.to(device)
         label = label.to(device)
         
         logits = model(x)
         preds = torch.argmax(logits, dim=1)
         true_labels = torch.argmax(label, dim=1)
         pred_list += preds.tolist()
         true_label_list += true_labels.tolist()
         #print("preds")
         
         loss += loss_fn(logits, label)
         #print("loss")
   
   
   acc, precision, recall, fscore, support, cm = compute_metrics(true_label_list, pred_list)
   
   #precision_recall_fscore_support(true_label_list, pred_list, average=None) 
   #returns tuple-> (precision, recall, fscore, support) precision = numpy array with precision for each class 0-6, ..
   #print("computing loss")
   
   avg_loss = loss/(step+1)
   #print("done computing")
   
   #fscore uses precision + recall and gives one number, support gives number of examples
   return avg_loss, acc, precision, recall, fscore, support         
      
def evaluate_and_display(true_labels, pred_labels, label_builder):
    acc, cm, precision, recall, fscore, support = compute_metrics(true_labels, pred_labels)
    print_evaluation_metrics(acc, cm, precision, recall, fscore, support, label_builder)
    