# Install Ultralytics for YOLO
!pip install ultralytics

# Import libraries
import zipfile
import os
from google.colab import drive

# Step 1: Mount Google Drive
drive.mount('/content/drive')

# Step 2: Extract dataset from Google Drive
zip_file_path = '/content/drive/MyDrive/combined_project.zip'  # Update with your actual file path
extract_folder = '/content/data/combined_project'  # Folder to extract files

if os.path.exists(zip_file_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    print("Dataset extracted successfully.")
else:
    raise FileNotFoundError("Zip file not found. Check the path in Google Drive.")

# Step 3: Check for data.yaml
data_yaml_path = os.path.join(extract_folder, 'data.yaml')
if os.path.exists(data_yaml_path):
    print(f"Found data.yaml at {data_yaml_path}")
else:
    raise FileNotFoundError("data.yaml not found in the extracted folder.")

# Step 4: Train YOLO model
model_name = 'yolo11n.pt'  # Replace with the model you want to use (e.g., yolov8n.pt, yolov5s.pt)
epochs_to_train = 100
backup_folder = '/content/drive/MyDrive/yolo_training'  # Target folder in Google Drive
os.makedirs(backup_folder, exist_ok=True)

# Train the model
!yolo task=detect mode=train model={model_name} data={data_yaml_path} epochs={epochs_to_train} imgsz=512 batch=16 patience=50 project={backup_folder} name=yolo_training_session1
print("Training complete for this session.")

# Step 5: Save the model and results
final_model_path = os.path.join(backup_folder, 'yolo_training_session1', 'weights', 'best.pt')
if os.path.exists(final_model_path):
    print(f"Model checkpoint saved to {final_model_path}")
else:
    print("Model checkpoint not found. Check training logs for errors.")

