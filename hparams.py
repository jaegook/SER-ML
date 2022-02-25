
class Hyperparameters: 
# parameters
   mode = "classifier"

# parameters for audio processing
   sampling_rate = 16000
   frame_size = 512   		# 32 ms
   hop_length = 160   		# 10 ms
   n_mels = 32			    #
   sample_size = 16000*4	# 1 second because it is smaller than our minimum length in our data_list
   trim_silence = True
   num_frames = 101

# parameters for classifier training
   batch_size = 32
   num_workers = 2
   num_epochs = 50
   init_lr = 1e-3
   end_lr = 5e-5	
   display_step = 50
   validation_step = 1000
   
# parameters for contrastive training   
   num_contrastive_samples = 2
   contrastive_batch_size = 2

   temperature = 0.1
   num_neg_examples = 2
   num_contrastive_epochs = 20
   contrastive_validation_step = 200
   contrastive_display_step = 2
   
# directories to make
   labels_dir = "labels"	#directory to save label_to_id dictionaries
   model_dir = "models"		#directory to save models

   