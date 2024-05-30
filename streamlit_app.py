import streamlit as st
import leafmap.foliumap as leafmap

# To find emoji: https://emojipedia.org

st.set_page_config(page_title="SEKRUM Review Analysis", page_icon=":earth_asia:", layout="wide")

# Customize the sidebar
markdown = """
Insert Project Description Here
"""

st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "images\SEKRUM Logo.jpg"
st.sidebar.image(logo)

# Customize page title
st.title("Summary Page")

st.markdown(
    """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    """
)

st.header("#To-do")

markdown = """
1. Create the summary page (Figma Design)
2. Create the file upload page?
3. Create the review summarizer page n word cloud page?

To Add a new app to the `pages/` directory with an emoji in the file name, e.g., `1_🚀_Chart.py`.

"""

st.markdown(markdown)