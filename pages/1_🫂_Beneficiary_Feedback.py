import streamlit as st
import pandas as pd
import numpy as np
from streamlit_card import card
import hydralit_components as hc

# st.dataframe(st.session_state.beneficiary_df)

st.set_page_config(page_title = 'Beneficiary Feedback', 
    layout='wide',
    page_icon='🫂')

st.title("Beneficiary Feedback")

# To connect sentiment analysis model


#can apply customisation to almost all the properties of the card, including the progress bar
theme_bad = {'bgcolor': '#FFF0F0','title_color': 'red','content_color': 'red','icon_color': 'red', 'icon': 'fa fa-times-circle'}
theme_neutral = {'bgcolor': '#f9f9f9','title_color': 'orange','content_color': 'orange','icon_color': 'orange', 'icon': 'fa fa-question-circle'}
theme_good = {'bgcolor': '#EFF8F7','title_color': 'green','content_color': 'green','icon_color': 'green', 'icon': 'fa fa-check-circle'}

### top row 
first_kpi, second_kpi, third_kpi = st.columns(3)

with first_kpi:
    st.markdown("**Number of Reviews**")
    number1 = st.session_state.beneficiary_df['Review'].count()
    st.markdown(f"<h1 style='text-align: center; color: black;'>{number1}</h1>", unsafe_allow_html=True)
    hc.info_card(title='Number of Reviews', 
                 content="test", 
                 bar_value=77,
                 theme_override=theme_neutral)

with second_kpi:
    st.markdown("**Number of Positive Reviews**")
    number2 = 222 
    st.markdown(f"<h1 style='text-align: center; color: green;'>{number2}</h1>", unsafe_allow_html=True)

with third_kpi:
    st.markdown("**Number of Negative Reviews**")
    number3 = 333 
    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)


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