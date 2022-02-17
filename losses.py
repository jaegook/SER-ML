import torch
import torch.nn as nn
class ContrastiveLoss(nn.Module):
   def __init__(self):
      super().__init__()
      pass
   def forward(self, projection_list, label_list, temperature):
      total_loss = 0.0
      for projection, label in zip(projection_list, label_list):        #projection: shape-> [6,2,128], label: shape->[6,]
         labels = label.contiguous().view(-1,1)                         #labels: shape-> [6,1]
         mask = torch.eq(labels, labels.T).float()                      #labels.T: shape->[1,6], labels:shape->[6,1], mask: shape->[6,6]
         n_views = projection.size(1)                                   #n_views = 2
         x = torch.unbind(projection, dim=1)                            #x: shape-> ([6,128], [6,128]) unbind splits across dim=1 and erases that dim
         features = torch.cat(x, dim=0)                                 #features: shape-> [12,128]
         features_dot_prod = torch.matmul(features, features.T)         #features_dot_prod: shape-> [12,12] because [12,128] * [128,12]
         features_dot_prod = torch.div(features_dot_prod, temperature)  #divide features_dot_prod by temperature
         max_dot, _ = torch.max(features_dot_prod, dim=1, keepdim=True)
         features_dot_prod -= max_dot.detach()
         
         mask = mask.repeat(n_views, n_views)                           #repeat is also called tiling, n_views = 2, mask shape is now [12,12]
         ones = torch.ones_like(mask)                                   #ones shape is same as mask: [12,12] but filled with all 1's
         r = torch.arange(projection.size(0)*n_views).view(-1,1)        #r: shape-> [12,1]         
         mask_logits = torch.scatter(ones, 1, r, 0)
         mask = mask*mask_logits
         exp_logits = torch.exp(features_dot_prod)*mask_logits
         log_prob = features_dot_prod - torch.log(exp_logits.sum(1,keepdim=True))
         avg_log_prob_pos = (mask*log_prob).sum(1)/mask.sum(1)
         loss = -temperature*avg_log_prob_pos 
         loss = loss.view(n_views, projection.size(0)).mean()
         total_loss += loss
      
      return total_loss/len(projection_list)
         
      
   
   
class MultiClassLoss(nn.Module):
   def __init__(self):
      super().__init__()
   def forward(self, logits, labels):
      pass