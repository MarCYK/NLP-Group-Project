import streamlit as st
import pandas as pd
import numpy as np
import hydralit_components as hc

st.set_page_config(page_title = 'Beneficiary Feedback', 
    layout='wide',
    page_icon='🫂')

if st.session_state.beneficiary_df.empty:
    st.write("Please upload a file to get started.")
else:
    st.dataframe(st.session_state.beneficiary_df)

st.title("Beneficiary Feedback")

# To connect sentiment analysis model


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