import json

# how to use labelbuilder in dataset, when and where to save()
class LabelBuilder:
   def __init__(self):
      self.label_to_id = {}
      self.id_to_label = {}      

   #return label given label_id
   def get_label(self, label_id):
      if label_id in self.id_to_label:
         return self.id_to_label[label_id]
      else:
         return None

   def get_label_id(self, label):
      if label in self.label_to_id:
         return self.label_to_id[label]
      else:
         return None

   #save dict as json file
   def save(self, filepath):
      with open(filepath, "w") as outfile:
         json.dump(self.label_to_id, outfile)
   
   #load json file and convert to python dict
   def load(self, filepath):
      with open(filepath) as json_file:
         self.label_to_id = json.load(json_file)
      for label, id in self.label_to_id.items():
         self.id_to_label[id] = label

   
   # add label and return label id
   def add_label(self, label_name):
      if label_name in self.label_to_id:
         return self.label_to_id[label_name]
      else:
         label_id = len(self.label_to_id)
         self.label_to_id[label_name] = label_id
         self.id_to_label[label_id] = label_name
         return self.label_to_id[label_name]
   
   def count(self):
      return len(self.label_to_id)