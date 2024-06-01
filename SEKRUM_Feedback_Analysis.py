import streamlit as st
import pandas as pd

# To find emoji: https://emojipedia.org

st.set_page_config(page_title="SEKRUM Feedback Analysis", page_icon=":earth_asia:", layout="wide")

# Customize the sidebar
markdown = """
Insert Project Description Here
"""

st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "images\SEKRUM Logo.jpg"
st.sidebar.image(logo)

# Initialize session state
if 'beneficiary_df' not in st.session_state:
    st.session_state.beneficiary_df = pd.DataFrame()
if 'volunteer_df' not in st.session_state:
    st.session_state.volunteer_df = pd.DataFrame()

# Customize page title
st.title("Upload Feedback Data")
# st.markdown(
#     """
#     Please upload in CSV format.
#     """
# )

feedbackID=st.radio("Choose the Feedback type",
                    ("Beneficiary","Volunteer"), 
                    horizontal=True)

if feedbackID=="Beneficiary":
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
        
        beneficiary_df = beneficiary_temp[(beneficiary_temp['Review'].notnull()) & (beneficiary_temp['Review'] != "")].reset_index(drop=True)
        st.session_state.beneficiary_df = beneficiary_df
        st.session_state.beneficiary_file_name = beneficiary_file_name

    # To-do: Drop Rating & Category column, they are for training only

    # Display the uploaded file
    if st.session_state.beneficiary_df.empty:
        st.write("Please upload a file to get started.")
    else:
        st.header("Data Preview")
        st.write(st.session_state.beneficiary_file_name)
        st.dataframe(st.session_state.beneficiary_df, hide_index=True)
        st.write("Number of reviews: ", st.session_state.beneficiary_df.shape[0])
           
elif feedbackID=="Volunteer":
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
        
        volunteer_df = volunteer_temp[volunteer_temp['Review'].notnull()].reset_index(drop=True)    
        st.session_state.volunteer_df = volunteer_df
        st.session_state.volunteer_file_name = volunteer_file_name

    # Display the uploaded file
    if st.session_state.volunteer_df.empty:
        st.write("Please upload a file to get started.")
    else:
        st.header("Data Preview")
        st.write(st.session_state.volunteer_file_name)
        st.dataframe(st.session_state.volunteer_df, hide_index=True)
        st.write("Number of reviews: ", st.session_state.volunteer_df.shape[0])


    




# To-Dos
# st.title("Summary Page")

# st.markdown(
#     """
#     Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
#     """
# )

# st.header("#To-do")

# markdown = """
# 1. Create the summary page (Figma Design)
# 2. Create the file upload page?
# 3. Create the review summarizer page n word cloud page?

# To Add a new app to the `pages/` directory with an emoji in the file name, e.g., `1_🚀_Chart.py`.

# """

# st.markdown(markdown)