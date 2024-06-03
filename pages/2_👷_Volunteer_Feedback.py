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
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import matplotlib.pyplot as plt
import re
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

st.set_page_config(page_title = 'Volunteer Feedback', 
    layout='wide',
    page_icon='👷')

# Customize the sidebar
markdown = """
This is a Streamlit app for analyzing feedback data from beneficiaries and volunteers.

&nbsp;
&nbsp;

Source code:
https://github.com/MarCYK/NLP-Group-Project

&nbsp;
&nbsp;

In collaboration with:
SEKRETARIAT SUKARELAWAN UNIVERSITI MALAYA (SEKRUM)
"""

st.sidebar.title("About")
logo = "images\SEKRUM Logo.jpg"
st.sidebar.info(markdown)
st.sidebar.image(logo)

st.title("Volunteer Feedback")

# Initialize session state
if "volunteer_results_df" not in st.session_state:
    st.session_state.volunteer_results_df = pd.DataFrame()
if not st.session_state.volunteer_results_df.empty:
    st.write("volunteer results loaded")

if st.session_state.volunteer_df.empty:
    st.write("Please upload a file to get started.")

# Training the model stuff
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
                progress_text.text(f'{len(predictions)}\{st.session_state.volunteer_df.shape[0]} reviews processed')


    return predictions


volunteer_results_df = st.session_state.volunteer_results_df

f_checkpoint = Path("model/model.safetensors")
        
if not f_checkpoint.exists(): #To avoid breaking the code when changing page while installing
    # Load the model and tokenizer
    tokenizer, model = install_model()
elif volunteer_results_df.empty:
    tokenizer, model = install_model()

    df = st.session_state.volunteer_df

    df = prepare_dataset(df, tokenizer)

    predictions = make_predictions(model, df, device)

    volunteer_results_df = pd.DataFrame({
        'Review': st.session_state.volunteer_df['Review'],
        'Predicted Sentiment': [pred + 1 for pred in predictions]  # Adjust label scale if necessary
    })
    st.session_state.volunteer_results_df = volunteer_results_df


# Data visualization
#can apply customisation to almost all the properties of the card, including the progress bar
theme_bad = {'bgcolor': '#FFF0F0','title_color': 'red','content_color': 'red','icon_color': 'red', 'icon': 'fa fa-thumbs-down', 'progress_color': 'red'}
theme_neutral = {'bgcolor': '#FFF4EF','title_color': 'orange','content_color': 'orange','icon_color': 'orange', 'icon': 'fa fa-question-circle', 'progress_color': 'orange'}
theme_good = {'bgcolor': '#EFF8F7','title_color': 'green','content_color': 'green','icon_color': 'green', 'icon': 'fa fa-thumbs-up', 'progress_color': 'green'}
theme_review = {'bgcolor': '#FFFFFF','title_color': 'black','content_color': 'black','icon_color': 'black', 'icon': 'fa fa-envelope-open-text', 'progress_color': 'black'}

# Review Category
Positive = volunteer_results_df[volunteer_results_df['Predicted Sentiment'] >= 4]
Neutral = volunteer_results_df[volunteer_results_df['Predicted Sentiment'] == 3]
Negative = volunteer_results_df[volunteer_results_df['Predicted Sentiment'] <= 2]

### top row 
first_kpi, second_kpi, third_kpi, fourth_kpi = st.columns(4)

with first_kpi:
    number1 = st.session_state.volunteer_df['Review'].count()
    hc.info_card(title='Number of Reviews', 
                 content=number1.__str__(), 
                 bar_value=number1.__str__(),
                 theme_override=theme_review,
                 title_text_size='20px',
                 icon_size='30px')

with second_kpi:
    number2 = Positive.shape[0]
    sum2 = number2/number1 * 100
    hc.info_card(title='Number of Positive Reviews', 
                 content=number2.__str__(), 
                 bar_value= sum2,
                 theme_override=theme_good,
                 title_text_size='20px',
                 icon_size='30px',)

with third_kpi:
    number3 = Neutral.shape[0]
    sum3 = number3/number1 * 100
    hc.info_card(title='Number of Neutral Reviews', 
                content=number3.__str__(), 
                bar_value=sum3,
                theme_override=theme_neutral,
                title_text_size='20px',
                icon_size='30px',)

with fourth_kpi:
    number3 = Negative.shape[0]
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


# Word Cloud
st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown("## Word Cloud")
st.markdown("What the beneficiaries are saying about the program...")

text = ' '.join(volunteer_results_df['Review'].astype(str))
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(["Smart Tutor Program", "program", "Kompleks Perdana Siswa", "Dato Kamaruddin Mosque", "Universiti Malaya", "Dato Kamaruddin", "Smart Tutor", "Smart", "Tutor", "overall", "Soft"])
text = re.sub(r'\b\d+\b', '', text)

tokens = word_tokenize(text)
tagged_tokens = pos_tag(tokens)
adjectives = [word for word, pos in tagged_tokens if pos in ['JJ', 'JJR', 'JJS'] and word.lower() not in custom_stopwords]
filtered_text = ' '.join(adjectives)

wordcloud = WordCloud(stopwords=custom_stopwords, background_color='white', width=800, height=400).generate(filtered_text)
fig = plt.figure(figsize=(10, 5))
ax = plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
st.pyplot(fig)