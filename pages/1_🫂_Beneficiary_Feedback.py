import streamlit as st
import pandas as pd
import numpy as np
import hydralit_components as hc
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import Dataset
from safetensors.torch import load_model

st.set_page_config(page_title = 'Beneficiary Feedback', 
    layout='wide',
    page_icon='🫂')

if st.session_state.beneficiary_df.empty:
    st.write("Please upload a file to get started.")

st.title("Beneficiary Feedback")

# Streamlit sharing is CPU only
device = torch.device('cpu')

# To connect sentiment analysis model
# https://drive.google.com/file/d/1-8Od2aCrZ2vGMHA5wScJapXrgJcvYO_C/view?usp=sharing
cloud_model_location = "1-8Od2aCrZ2vGMHA5wScJapXrgJcvYO_C"

def install_model():
    # # Load the original model with 3 labels
    # original_model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    # original_model = AutoModelForSequenceClassification.from_pretrained(original_model_name, num_labels=3)

    # # Modify the model to have 5 labels
    # model = AutoModelForSequenceClassification.from_pretrained(original_model_name, num_labels=5)

    # # Copy the weights from the original model, ignoring the classifier weights
    # for name, param in original_model.named_parameters():
    #     if "classifier" not in name:
    #         model.state_dict()[name].copy_(param)

    # # Initialize the new classifier weights
    # model.classifier.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    # model.classifier.bias.data.zero_()

    # # Save the modified model
    # modified_model_path = "modified_model"
    # model.save_pretrained(modified_model_path)
    # tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    # tokenizer.save_pretrained(modified_model_path)

    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    
    f_checkpoint = Path("model/model.safetensors")
        
    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive(cloud_model_location, f_checkpoint)
            print("Download complete!")
    
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    # model_name = "path/to/modified_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5, ignore_mismatched_sizes=True)

    load_model(model, "model/model.safetensors")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("model loaded...")
    return tokenizer, model

# Define a function to prepare the dataset
def prepare_dataset(dataframe, tokenizer, batch_size=8):
    dataset = Dataset.from_pandas(dataframe)
    dataset = dataset.map(tokenizer, input_columns="Review", fn_kwargs={"padding": "max_length", "truncation": True, "max_length": 512})
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

# Define a function to make predictions
def make_predictions(model, dataloader, device):
    model.to(device)
    model.eval()
    predictions = []
    # Create a placeholder for the batch progress
    progress_text = st.empty()

    with st.spinner("NLP is NLPing... this may take awhile! \n Don't stop it!"):
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                logits = outputs.logits
                predictions.extend(torch.argmax(logits, dim=-1).tolist())
                # Update the placeholder text
                progress_text.text(f'{len(predictions)}\{len(df)}')


    return predictions

# Load the model and tokenizer
tokenizer, model = install_model()

df = st.session_state.beneficiary_df

df = prepare_dataset(df, tokenizer)

predictions = make_predictions(model, df, device)

results_df = pd.DataFrame({
    'Review': st.session_state.beneficiary_df['Review'],
    'Predicted Sentiment': [pred + 1 for pred in predictions]  # Adjust label scale if necessary
})
st.session_state.beneficiary_results_df = results_df
st.dataframe(results_df)





#can apply customisation to almost all the properties of the card, including the progress bar
theme_bad = {'bgcolor': '#FFF0F0','title_color': 'red','content_color': 'red','icon_color': 'red', 'icon': 'fa fa-thumbs-down', 'progress_color': 'red'}
theme_neutral = {'bgcolor': '#FFF4EF','title_color': 'orange','content_color': 'orange','icon_color': 'orange', 'icon': 'fa fa-question-circle', 'progress_color': 'orange'}
theme_good = {'bgcolor': '#EFF8F7','title_color': 'green','content_color': 'green','icon_color': 'green', 'icon': 'fa fa-thumbs-up', 'progress_color': 'green'}
theme_review = {'bgcolor': '#FFFFFF','title_color': 'black','content_color': 'black','icon_color': 'black', 'icon': 'fa fa-envelope-open-text', 'progress_color': 'black'}

### top row 
first_kpi, second_kpi, third_kpi, fourth_kpi = st.columns(4)

with first_kpi:
    number1 = st.session_state.beneficiary_df['Review'].count()
    hc.info_card(title='Number of Reviews', 
                 content=number1.__str__(), 
                 bar_value=number1.__str__(),
                 theme_override=theme_review,
                 title_text_size='20px',
                 icon_size='30px')

with second_kpi:
    number2 = 222 
    sum2 = number2/number1 * 100
    hc.info_card(title='Number of Positive Reviews', 
                 content=number2.__str__(), 
                 bar_value= sum2,
                 theme_override=theme_good,
                 title_text_size='20px',
                 icon_size='30px',)

with third_kpi:
    number3 = 333
    sum3 = number3/number1 * 100
    hc.info_card(title='Number of Neutral Reviews', 
                content=number3.__str__(), 
                bar_value=sum3,
                theme_override=theme_neutral,
                title_text_size='20px',
                icon_size='30px',)

with fourth_kpi:
    number3 = 333
    sum3 = number3/number1 * 100
    hc.info_card(title='Number of Negative Reviews', 
                content=number3.__str__(), 
                bar_value=sum3,
                theme_override=theme_bad,
                title_text_size='20px',
                icon_size='30px',)

st.markdown("<hr/>", unsafe_allow_html=True)


st.markdown("## Chart Section: 1")

first_chart, second_chart = st.columns(2)


with first_chart:
    chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
    st.line_chart(chart_data)

with second_chart:
    chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
    st.line_chart(chart_data)


st.markdown("## Chart Section: 2")

first_chart, second_chart = st.columns(2)


with first_chart:
    chart_data = pd.DataFrame(np.random.randn(100, 3),columns=['a', 'b', 'c'])
    st.line_chart(chart_data)

with second_chart:
    chart_data = pd.DataFrame(np.random.randn(2000, 3),columns=['a', 'b', 'c'])
    st.line_chart(chart_data)