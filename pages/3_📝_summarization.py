import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from openai import OpenAI

st.set_page_config(page_title='Summarization', 
                   layout='wide',
                   page_icon='📝')

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

# Initialize session state for dataframes if not already initialized
if 'beneficiary_df' not in st.session_state:
    st.session_state.beneficiary_df = pd.DataFrame()
if 'volunteer_df' not in st.session_state:
    st.session_state.volunteer_df = pd.DataFrame()

# st.write(st.session_state.beneficiary_df)
# st.write(len(st.session_state.beneficiary_df.columns))

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


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
        df = st.session_state.beneficiary_results_df
        # df = preprocess_dataframe(df)
    else:
        st.write("Please upload a beneficiary feedback file on the main page to get started.")
        df = None
elif feedbackID == "Volunteer":
    print("volunteer")
    if not st.session_state.volunteer_df.empty:
        st.write("Volunteer Feedback DataFrame:")
        st.write(st.session_state.volunteer_file_name)
        df = st.session_state.volunteer_results_df
        # df = preprocess_dataframe(df)
    else:
        st.write("Please upload a volunteer feedback file on the main page to get started.")
        df = None

if df is not None:
    if 'Predicted Sentiment' not in df.columns:
        st.error("The 'Predicted Sentiment' column is missing from the DataFrame after preprocessing.")
    else:
        st.dataframe(filter_dataframe(df))

        # sentiment = st.selectbox("Select sentiment", ("Positive", "Neutral", "Negative"))
        # sentiment_mapping = {
        #     "Positive": [4, 3],
        #     "Neutral": [2],
        #     "Negative": [1, 0]
        # }

        # The new model seems to predict negative feedback as "Neutral", so even when choosing
        # "Neutral" sentiment to be summarized, it will still say like "The program suck facking ass" (rather than before it's more like "It's decent ig")
        # The lazy fix is just make two sentiment to be summarized; "Positive", "Negative"
        sentiment = st.selectbox("Select sentiment", ("Positive", "Negative"))
        sentiment_mapping = {
            "Positive": [4, 3],
            "Negative": [2, 1, 0]
        }

        filtered_df = df[df['Predicted Sentiment'].isin(sentiment_mapping[sentiment])]
        reviews = filtered_df['Review'].tolist()
        combined_reviews = " ".join(reviews)

        with st.spinner("Reading Reviews... Generating Summary... This may take awhile!"):
            summaries = summarize_text(combined_reviews)
        st.write(summaries)


# Suggestion - Open-AI Witchcraft
st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown("## Suggestion")
st.markdown("How can you improve the program from the feedback received...")

client = OpenAI(
    # Disable when not in use cause we poor asf
    # api_key = "sk-proj-NxBoJTkA7Aiq21YMHuJ4T3BlbkFJiGAKvqiWgCSkIBnL603B"
)

with st.spinner("Generating Suggestion... This may take awhile!"):
    # Define the prompt for suggestions
    prompt = f"Here are some negative reviews:\n\n{summaries}\n\nBased on these reviews, how can we improve the project?" 

    completion = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages=[
            {"role" : "system", "content" : "You are a program advisor that looks for improvements for programs"},
            {"role" : "user", "content" : prompt}

        ]
    )
    suggestions = completion.choices[0].message.content
st.write(suggestions)