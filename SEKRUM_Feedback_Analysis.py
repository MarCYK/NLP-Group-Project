import streamlit as st # type: ignore
import pandas as pd # type: ignore

# To find emoji: https://emojipedia.org

st.set_page_config(page_title="SEKRUM Feedback Analysis",
                   page_icon=":earth_asia:", layout="wide")

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

# Initialize session state
if 'beneficiary_df' not in st.session_state:
    st.session_state.beneficiary_df = pd.DataFrame()
if 'volunteer_df' not in st.session_state:
    st.session_state.volunteer_df = pd.DataFrame()
if "beneficiary_results_df" not in st.session_state:
    st.session_state.beneficiary_results_df = pd.DataFrame()
if "volunteer_results_df" not in st.session_state:
    st.session_state.volunteer_results_df = pd.DataFrame()

# Customize page title
st.title("Upload Feedback Data")
# st.markdown(
#     """
#     Please upload in CSV format.
#     """
# )

feedbackID = st.radio("Choose the Feedback type",
                      ("Beneficiary", "Volunteer"),
                      horizontal=True)

if feedbackID == "Beneficiary":
    st.header("Beneficiary Feedback")
    beneficiary_file = st.file_uploader("Choose a CSV file",
                                        type="csv",
                                        key="1")
    if beneficiary_file:
        try:
            beneficiary_temp = pd.read_csv(beneficiary_file)
            beneficiary_file_name = beneficiary_file.name
            st.success("File uploaded successfully.")
        except:
            st.error("File upload failed. Please upload a CSV file.")

        beneficiary_df = beneficiary_temp[(beneficiary_temp['Review'].notnull()) & (
            beneficiary_temp['Review'] != "")].reset_index(drop=True)
        st.session_state.beneficiary_df = beneficiary_df
        st.session_state.beneficiary_file_name = beneficiary_file_name
        st.session_state.beneficiary_results_df = pd.DataFrame()  # Reset Prediction Results

    # To-do: Drop Rating & Category column, they are for training only

    # Display the uploaded file
    if st.session_state.beneficiary_df.empty:
        st.write("Please upload a file to get started.")
    else:
        st.header("Data Preview")
        st.write(st.session_state.beneficiary_file_name)
        st.dataframe(st.session_state.beneficiary_df)
        st.write("Number of reviews: ",
                 st.session_state.beneficiary_df.shape[0])

elif feedbackID == "Volunteer":
    st.header("Volunteer Feedback")
    volunteer_file = st.file_uploader("Choose a CSV file",
                                      type="csv",
                                      key="2")
    if volunteer_file:
        try:
            volunteer_temp = pd.read_csv(volunteer_file)
            volunteer_file_name = volunteer_file.name
            st.success("File uploaded successfully.")
        except:
            st.error("File upload failed. Please upload a CSV file.")

        volunteer_df = volunteer_temp[volunteer_temp['Review'].notnull()].reset_index(
            drop=True)
        st.session_state.volunteer_df = volunteer_df
        st.session_state.volunteer_file_name = volunteer_file_name
        st.session_state.volunteer_results_df = pd.DataFrame()  # Reset Prediction Results

    # Display the uploaded file
    if st.session_state.volunteer_df.empty:
        st.write("Please upload a file to get started.")
    else:
        st.header("Data Preview")
        st.write(st.session_state.volunteer_file_name)
        st.dataframe(st.session_state.volunteer_df)
        st.write("Number of reviews: ", st.session_state.volunteer_df.shape[0])
