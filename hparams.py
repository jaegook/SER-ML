
class Hyperparameters: 
# parameters for audio processing
   sampling_rate = 16000
   frame_size = 512   		# 32 ms
   hop_length = 160   		# 10 ms
   n_mels = 32			    #
   sample_size = 16000		# 1 second because it is smaller than our minimum length in our data_list
   trim_silence = True


# parameters for training
   batch_size = 32
   num_workers = 2
   num_epochs = 50
   num_contrastive_epochs = 20
   init_lr = 1e-3
   end_lr = 5e-5	

# parameters for contrastive training   
   num_contrastive_samples = 4
   contrastive_batch_size = 2
   num_frames = 101
   temperature = 0.1
   
   contrastive_validation_step = 200
   contrastive_display_step = 10
# directories to make
   labels_dir = "labels"	#directory to save label_to_id dictionaries
   model_dir = "models"		#directory to save models

# parameters for validation
   display_step = 50
   