import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title='Summarization', 
                   layout='wide',
                   page_icon='📝')

# Initialize session state for dataframes if not already initialized
if 'beneficiary_df' not in st.session_state:
    st.session_state.beneficiary_df = pd.DataFrame()
if 'volunteer_df' not in st.session_state:
    st.session_state.volunteer_df = pd.DataFrame()

# Load the model and tokenizer
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_text(text, max_length=130, min_length=30):
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



feedbackID=st.radio("Choose the Feedback type",
                    ("Beneficiary","Volunteer"), 
                    horizontal=True)
# print(feedbackID)

if feedbackID == "Beneficiary":
    print("beneficiary")
    if not st.session_state.beneficiary_df.empty:
        st.write("Beneficiary Feedback DataFrame:")
        st.write(st.session_state.beneficiary_file_name)
        df = st.session_state.beneficiary_df
        df = preprocess_dataframe(df)
    else:
        st.write("Please upload a beneficiary feedback file on the main page to get started.")
        df = None
elif feedbackID == "Volunteer":
    print("volunteer")
    if not st.session_state.volunteer_df.empty:
        st.write("Volunteer Feedback DataFrame:")
        st.write(st.session_state.volunteer_file_name)
        df = st.session_state.volunteer_df
        df = preprocess_dataframe(df)
    else:
        st.write("Please upload a volunteer feedback file on the main page to get started.")
        df = None

if df is not None:
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
