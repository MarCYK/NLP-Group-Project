import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title = 'Summarization', 
    layout='wide',
    page_icon='📝')

# Initialize session state
if st.session_state.beneficiary_df.empty:
    st.write("Please upload a file to get started.")
else:
    st.dataframe(st.session_state.beneficiary_df)
if st.session_state.volunteer_df.empty:
    st.write("Please upload a file to get started.")
else:
    st.dataframe(st.session_state.volunteer_df)


# Load the model and tokenizer
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_text(text, max_length=130, min_length=30):
    print("summarizing...")
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def preprocess_dataframe(df):
    # Define a mapping dictionary for label conversion
    label_mapping = {
        "Excellent (5/5)": 4,
        "Good (4/5)": 3,
        "Neutral (3/5)": 2,
        "Poor (2/5)": 1,
        "Very poor (1/5)": 0
    }

    # Apply label conversion to the dataframe
    df['labels'] = df['Sentiment'].map(label_mapping)
    return df

# Streamlit app
st.title("Feedback Summarizer")
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

uploaded_file = st.session_state.uploaded_file

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = preprocess_dataframe(df)
    
    if 'labels' not in df.columns:
        st.error("The 'labels' column is missing from the DataFrame after preprocessing.")
    else:
        st.write(df.head())

        sentiment = st.selectbox("Select sentiment", ("Positive", "Neutral", "Negative"))
        sentiment_mapping = {
            "Positive": [4, 3],
            "Neutral": [2],
            "Negative": [1, 0]
        }

        filtered_df = df[df['labels'].isin(sentiment_mapping[sentiment])]
        reviews = filtered_df['Review'].tolist()
        combined_reviews = " ".join(reviews)

        summaries = summarize_text(combined_reviews)
        st.write(summaries)
