import streamlit as st
import pandas as pd

st.set_page_config(page_title = 'Volunteer Feedback', 
    layout='wide',
    page_icon='👷')

st.title("Volunter Feedback")

if st.session_state.volunteer_df.empty:
    st.write("Please upload a file to get started.")
else:
    st.dataframe(st.session_state.volunteer_df)