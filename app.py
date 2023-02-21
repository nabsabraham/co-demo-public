import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from utils import streamlit_header_and_footer_setup, top_menu, switch_menu
from config import config

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    initial_sidebar_state=config["sidebar"],
    layout=config["layout"],
)

streamlit_header_and_footer_setup()
menu = top_menu(0)
st.session_state["current_page"] = "Home"
if menu:
    switch_menu(menu)


# App title
st.markdown(
    "<h1 style='text-align: center; color: black;'>Clustering</h1>",
    unsafe_allow_html=True,
)

# App description
st.markdown(
    """
    This application allows you to submit cluster jobs and visualize their outputs!
    
    The steps to cluster your data include the following: 
    1. Embed your dataset - get Cohere embeddings for your text data! See example notebook [here](https://colab.research.google.com/drive/1X5TILcXq4pkmUv-K7wC1MqHxDwE_KVfg?usp=sharing#scrollTo=w51srx0n6XJz)
    2. Navigate to **Cluster** and submit a cluster job. Download your results. 
    3. Visualize your clustered data! 

    To get started, navigate to **Cluster**. 
    If you have a precomputed embeddings file from and precomputed clusters, navigate to **Visualize**. 
    ### Want to learn more?
    - Check out our documentation [here](https://docs.cohere.ai) 
    - Ask questions in our discord community [here](https://discord.com/invite/co-mmunity) 
"""
)
