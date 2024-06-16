from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
import nibabel as nib
import numpy as np
import os
import shutil
import pickle
import pandas as pd
from tempfile import NamedTemporaryFile

# Function to load the model from the pickle file
def load_model():
    with open('svm_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

# Load the trained model
model = load_model()

# Function to load image data from a filepath
def get_image_data(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return data

# Function to create a vector from a region by time matrix from an image using the atlas
def image_to_vector(image_data, atlas_data):
    time_dim = image_data.shape[-1]
    column_names = [f'time_{i}' for i in range(time_dim)]
    region_names = [f'region_{region}' for region in np.unique(atlas_data)]

    reshaped_image_data = image_data.reshape(-1, time_dim)
    df_times = pd.DataFrame(reshaped_image_data, columns=column_names)
    reshaped_atlas_data = atlas_data.reshape(-1)
    df_full = pd.concat([pd.Series(reshaped_atlas_data, name='atlas_region'), df_times], axis=1)
    regions_x_time = df_full.groupby('atlas_region').mean()
    regions_x_time.index = region_names
    regions_x_time_vector = regions_x_time.to_numpy().reshape(-1)
    return regions_x_time_vector

# Function to preprocess the input image and extract features
def preprocess_and_extract_features(nifti_data, atlas_data):
    features = image_to_vector(nifti_data, atlas_data)
    num_required_features = 116
    if features.size < num_required_features:
        features = np.pad(features, (0, num_required_features - features.size), 'constant')
    else:
        features = features[:num_required_features]
    return features.reshape(1, -1)

# FastAPI app setup
app = FastAPI()

@app.post("/predict", response_class=PlainTextResponse)
async def predict_region(file: UploadFile = File(...)):
    temp_file_path = None
    try:
        temp_file_path = NamedTemporaryFile(suffix=".nii.gz", delete=False).name
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        img = nib.load(temp_file_path)
        data = img.get_fdata()
        
        atlas_filepath = 'aal_mask_pad.nii.gz'
        if not os.path.exists(atlas_filepath):
            raise FileNotFoundError(f"Atlas file not found at: {atlas_filepath}")
        
        atlas_data = get_image_data(atlas_filepath)
        features = preprocess_and_extract_features(data, atlas_data)
        prediction = model.predict(features)
        return str(prediction[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
