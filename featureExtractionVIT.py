import os
import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel

# Load the feature extractor and pre-trained ViT model from Hugging Face
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')

# Define a function to load and preprocess images
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# Extract features from images in a folder
def extract_vit_features(model, feature_extractor, folder_path, device='cpu'):
    model = model.to(device)
    model.eval()
    
    feature_list = []
    
    # Iterate through each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            
            # Preprocess the image
            inputs = load_and_preprocess_image(file_path)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to device
            
            # Extract features using ViT
            with torch.no_grad():
                outputs = model(**inputs)
                features = outputs.last_hidden_state  # Extract last hidden state features
                
                # We will use the mean of the feature vectors as the image feature
                image_features = features.mean(dim=1)  # Shape: (1, feature_dim)
                feature_list.append(image_features.squeeze(0))  # Remove the batch dimension
    
    # Stack all features into a single tensor
    all_features = torch.stack(feature_list)  # Shape: (num_frames, feature_dim
    torch.cuda.empty_cache() 
    return all_features

# Process all subfolders in the main directory
def process_subfolders(main_folder_path, save_folder, log_file_path, device='cpu'):
    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)
    
    #checkpoint = 0
    
    # Open the log file to record processed folder names
    with open(log_file_path, 'a') as log_file:
        # Iterate through each subfolder in the main directory
        for folder_name in os.listdir(main_folder_path):
            folder_path = os.path.join(main_folder_path, folder_name)
            
            # Skip non-directories
            if not os.path.isdir(folder_path):
                continue
            
            print(f"Processing subfolder: {folder_name}")
          
            # Extract features
            frame_features = extract_vit_features(model, feature_extractor, folder_path, device=device)
            
            # Move features to CPU before saving
            # frame_features_cpu = frame_features.to('cpu')
            
            # Save the extracted features
            save_path = os.path.join(save_folder, f'{folder_name}_features_vit.pt')
            torch.save(frame_features, save_path)
            
            print(f"Features for {folder_name} saved to {save_path}")
            
            # Log the processed folder name
            log_file.write(f"{folder_name}\n")
            log_file.flush()  # Ensure the name is immediately written to disk
            
            #if checkpoint >= 4:
            #    break
            #checkpoint += 1


# Example usage
if __name__ == '__main__':
    # Path to your main directory containing subfolders
    # main_folder_path = '/home/jiayu/ParkVehicle/SLR_back/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train'
    # main_folder_path = '/home/jiayu/ParkVehicle/SLR_back/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test'
    main_folder_path = '/home/streetparking/SLR/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev'
   
    # Specify folder to save the features
    #save_folder = '/home/jiayu/ParkVehicle/SLR_back/trainingVideoFeaturesGPU'
    #save_folder = '/home/jiayu/ParkVehicle/SLR_back/testingVideoFeaturesGPU'
    save_folder = '/home/streetparking/SLR/devingVideoFeaturesGPU'
    #save_folder = '/home/jiayu/ParkVehicle/SLR_back/trainingVideoFeatures'
    #save_folder = '/home/jiayu/ParkVehicle/SLR_back/testingVideoFeatures'

    # Path to log file where processed folder names will be recorded
    #log_file_path = '/home/jiayu/ParkVehicle/SLR_back/processed_folders_trainingGPU.txt'
    #log_file_path = '/home/jiayu/ParkVehicle/SLR_back/processed_folders_testingGPU.txt'
    log_file_path = '/home/streetparking/SLR/processed_folders_devingGPU.txt'
    #log_file_path = '/home/jiayu/ParkVehicle/SLR_back/processed_folders.txt'
    #log_file_path = '/home/jiayu/ParkVehicle/SLR_back/processed_folders_testing.txt'
    
    # Choose the device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process all subfolders and log the processed folder names
    process_subfolders(main_folder_path, save_folder, log_file_path, device=device)

    print("Feature extraction for all subfolders completed!")
