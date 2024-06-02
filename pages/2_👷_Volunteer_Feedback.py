import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import Dataset
from safetensors.torch import load_model
from pathlib import Path
import hashlib

st.set_page_config(page_title='Volunteer Feedback', layout='wide', page_icon='👷')

st.title("Volunteer Feedback")

# Initialize session state for file hashes
if 'volunteer_file_hash' not in st.session_state:
    st.session_state.volunteer_file_hash = None
if 'beneficiary_file_hash' not in st.session_state:
    st.session_state.beneficiary_file_hash = None

# Streamlit sharing is CPU only
device = torch.device('cpu')

# Model location
cloud_model_location = "1-8Od2aCrZ2vGMHA5wScJapXrgJcvYO_C"

def install_model():
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    
    f_checkpoint = Path("model/model.safetensors")
        
    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive(cloud_model_location, f_checkpoint)
            print("Download complete!")
    
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5, ignore_mismatched_sizes=True)
    load_model(model, "model/model.safetensors")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded...")
    return tokenizer, model

def prepare_dataset(dataframe, tokenizer, batch_size=8):
    dataset = Dataset.from_pandas(dataframe)
    dataset = dataset.map(tokenizer, input_columns="Review", fn_kwargs={"padding": "max_length", "truncation": True, "max_length": 512})
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def make_predictions(model, dataloader, device):
    model.to(device)
    model.eval()
    predictions = []
    progress_text = st.empty()

    with st.spinner("NLP is NLPing... this may take awhile! \n Don't stop it!"):
        with torch.no_grad():
            for batch_num, batch in enumerate(dataloader, start=1):
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                logits = outputs.logits
                predictions.extend(torch.argmax(logits, dim=-1).tolist())
                progress_text.text(f'Processed batch {batch_num}/{len(dataloader)}')

    return predictions

# Load the model and tokenizer
tokenizer, model = install_model()

# Helper function to compute file hash
def compute_file_hash(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()

# Check if the volunteer dataframe exists and process it
if not st.session_state.volunteer_df.empty:
    # Compute hash of the current volunteer dataframe
    current_volunteer_file_hash = compute_file_hash(st.session_state.volunteer_df)

    # Check if new predictions are needed for volunteer data
    if st.session_state.volunteer_file_hash != current_volunteer_file_hash:
        st.session_state.volunteer_file_hash = current_volunteer_file_hash
        st.session_state.volunteer_results_df = pd.DataFrame()  # Reset results if new file is uploaded
        
        # Prepare the dataset and make predictions
        dataloader = prepare_dataset(st.session_state.volunteer_df, tokenizer)
        predictions = make_predictions(model, dataloader, device)
        results_df = pd.DataFrame({
            'Review': st.session_state.volunteer_df['Review'],
            'Predicted Sentiment': [pred + 1 for pred in predictions]  # Adjust label scale if necessary
        })
        st.session_state.volunteer_results_df = results_df

# Display the results for volunteer data if available
if not st.session_state.volunteer_results_df.empty:
    st.header("Volunteer Feedback Results")
    st.dataframe(st.session_state.volunteer_results_df)

# Need to transfer the below one to the beneficiary feedback page

# # Check if the beneficiary dataframe exists and process it
# if not st.session_state.beneficiary_df.empty:
#     # Compute hash of the current beneficiary dataframe
#     current_beneficiary_file_hash = compute_file_hash(st.session_state.beneficiary_df)

#     # Check if new predictions are needed for beneficiary data
#     if st.session_state.beneficiary_file_hash != current_beneficiary_file_hash:
#         st.session_state.beneficiary_file_hash = current_beneficiary_file_hash
#         st.session_state.beneficiary_results_df = pd.DataFrame()  # Reset results if new file is uploaded
        
#         # Prepare the dataset and make predictions
#         dataloader = prepare_dataset(st.session_state.beneficiary_df, tokenizer)
#         predictions = make_predictions(model, dataloader, device)
#         results_df = pd.DataFrame({
#             'Review': st.session_state.beneficiary_df['Review'],
#             'Predicted Sentiment': [pred + 1 for pred in predictions]  # Adjust label scale if necessary
#         })
#         st.session_state.beneficiary_results_df = results_df

# # Display the results for beneficiary data if available
# if not st.session_state.beneficiary_results_df.empty:
#     st.header("Beneficiary Feedback Results")
#     st.dataframe(st.session_state.beneficiary_results_df)
