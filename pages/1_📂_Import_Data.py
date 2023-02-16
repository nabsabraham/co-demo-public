import ast
import hashlib

import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from utils import streamlit_header_and_footer_setup, top_menu

st.set_page_config(
    page_title="Import Data",
    page_icon="📈",
    initial_sidebar_state='collapsed',
    #layout="wide"
)
# Render header/footer
streamlit_header_and_footer_setup()
# Set menu index 
menu = top_menu(1)
# Menu routing
if menu == 'Cluster':
    switch_page('Cluster')
elif menu == 'Home':
    switch_page('app')
elif menu == 'Charts':
    switch_page('Charts')

#st.title('Import Data')
#uploaded_file = st.file_uploader("Choose a file", type=['csv', 'jsonl'])
#st.write("Or, alternatively:")
path_to_gcs = st.text_input("Insert a JSONL path from the output of `co.cluster()`.")
co_cluster_output = False

uploaded_file = None
#if uploaded_file is not None:
    # JSONL file support - This would be an output of co.cluster
#    if '.jsonl' in uploaded_file.name:
#        co_cluster_output = True
        # Can be used wherever a "file-like" object is accepted:
#        df = pd.read_json(uploaded_file, lines=True)
#        df = df[['description', 'keywords', 'elements', 'cluster_id']]    

#        df['elements'] = df['elements'].apply(lambda x: ast.literal_eval(str(x)))

#        df = (df.explode("elements")
#            .rename(columns={"elements": "text", "description": "Cluster Name", "cluster_id": "Cluster"})
#            .reset_index(drop = True)
#            )
#        st.session_state['processed_df'] = df
#        st.session_state['text_column'] = 'text'
#    else:
#        df = pd.read_csv(uploaded_file)
#    file_name = uploaded_file.name
#    file_id = uploaded_file.id
#    file_size = uploaded_file.size

import_fail = False
if path_to_gcs:
    try:
        if '.jsonl' in path_to_gcs:
            co_cluster_output = True
            # Can be used wherever a "file-like" object is accepted:
            df = pd.read_json(path_to_gcs, lines=True)
            df = df[['description', 'keywords', 'elements', 'cluster_id']]    

            df['elements'] = df['elements'].apply(lambda x: ast.literal_eval(str(x)))

            df = (df.explode("elements")
                .rename(columns={"elements": "text", "description": "Cluster Name", "cluster_id": "Cluster"})
                .reset_index(drop = True)
                )
            st.session_state['processed_df'] = df
            st.session_state['text_column'] = 'text'
        else:
            df = pd.read_csv(path_to_gcs)
        file_name = path_to_gcs
        file_id = int(hashlib.sha256(file_name.encode('utf-8')).hexdigest(), 16) % 10**8
        file_size = len(df)
    except Exception as e:
        st.error("Error")
        import_fail = True
        st.write("Please make sure the provided file path is a JSONL output from co.cluster()")


if uploaded_file is not None or path_to_gcs and not import_fail:
    # Create a file UUID to be used for file specific caching
    file_uuid = f'{file_name.split(".")[0]}_{file_id}_{file_size}'
    next = st.button('Next')
    st.success("Success")
    st.write(df)
    st.session_state['df'] = df
    st.session_state['import_done'] = True
    st.session_state['co_cluster_output'] = co_cluster_output
    st.session_state['file_uuid'] = file_uuid
    # Delete state of text column on each new dataset import
    if 'text_column' in st.session_state and not co_cluster_output: 
        del st.session_state['text_column']
    if next:
        if co_cluster_output:
            switch_page('Charts')
        else:
            switch_page('Cluster')