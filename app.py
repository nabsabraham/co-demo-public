import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from utils import streamlit_header_and_footer_setup, top_menu

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    initial_sidebar_state='collapsed',
    #layout="wide"

)

streamlit_header_and_footer_setup()
menu = top_menu(0)

if menu == 'Import':
    switch_page('Import Data')
elif menu == 'Cluster':
    switch_page('Cluster')
elif menu == 'Charts':
    switch_page('Charts')


# App title
st.markdown("<h1 style='text-align: center; color: black;'>Clustering</h1>", unsafe_allow_html=True)

# App description
st.markdown(
    """
    This application allows visualize clusters from the output of co.cluster()  
    
    Upload a dataset using the **Import** tab in the menu to get started.

    ### Want to learn more?
    - Check out our documentation [here](https://docs.cohere.ai) 
    - Ask questions in our discord community [here](https://discord.com/invite/co-mmunity) 
"""
)

