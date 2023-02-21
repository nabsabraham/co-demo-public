import ast
import hashlib

import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

import plotly.express as px
import plotly.graph_objects as go

from wordcloud import WordCloud

from utils import (
    streamlit_header_and_footer_setup,
    top_menu,
    switch_menu,
    process_clusterdf,
    rename_clusters,
    show_wordcloud,
    compute_pca,
    join_pca_cluster_dfs,
    get_topk_clusters,
)
from config import config

CURRENT_PAGE = "Import"
st.set_page_config(
    page_title="Import/compute Cohere clusters",
    page_icon="📈",
    initial_sidebar_state=config["sidebar"],
    layout=config["layout"],
)
# Render header/footer
streamlit_header_and_footer_setup()
# Set menu index
menu = top_menu(1)
st.session_state["current_page"] = CURRENT_PAGE
if menu:
    switch_menu(menu)


st.title("Import cluster data")
st.write(
    """
    You've precomputed your text embeddings and clusters and want to visualize them!
    Select how you'd like to upload your data and then hit **Next** to visualize the output.
    """
)
uploaded_file, path_to_gcs = None, None
text_column = "text"
st.session_state["processed_df"] = None
st.session_state["cluster_df"] = None
st.session_state["co_cluster_output"] = None
st.session_state["embed_df"] = None
# st.session_state["pca_df"] = None

CLUSTER_COLUMNS = ["description", "keywords", "elements", "cluster_id", "text_ids"]

# give user the choice to upload from filesystem or a bucket
upload_options = ["file-system", "cloud-bucket"]
upload_choice = st.radio("Upload type", upload_options)
if upload_choice == "file-system":
    embed_file = st.file_uploader("Choose an embeddings output file", type=["jsonl"])
    if embed_file is not None:
        embed_df = pd.read_json(embed_file, lines=True)
        st.session_state["embed_df"] = embed_df

    uploaded_file = st.file_uploader(
        "Choose a cluster output file", type=["csv", "jsonl"]
    )
    co_cluster_output = False
    if uploaded_file is not None:
        # JSONL file support - This would be an output of co.cluster
        if ".jsonl" in uploaded_file.name:
            co_cluster_output = True
            # Can be used wherever a "file-like" object is accepted:
            df = pd.read_json(uploaded_file, lines=True)
            assert set(CLUSTER_COLUMNS).issubset(
                set(df.columns)
            ), f"Expected file to contain columns: {CLUSTER_COLUMNS}"
            df = df[CLUSTER_COLUMNS]
            df = process_clusterdf(df)

            st.session_state["processed_df"] = df
            st.session_state["text_column"] = text_column
        else:
            df = pd.read_csv(uploaded_file)
        file_name = uploaded_file.name
        file_id = uploaded_file.id
        file_size = uploaded_file.size

else:
    path_to_gcs = st.text_input("Insert a path to the output of `co.cluster()`.")
    if path_to_gcs:
        if ".jsonl" in path_to_gcs:
            co_cluster_output = True
            # Can be used wherever a "file-like" object is accepted:
            df = pd.read_json(path_to_gcs, lines=True)
            assert set(CLUSTER_COLUMNS).issubset(
                set(df.columns)
            ), f"Expected file to contain columns: {CLUSTER_COLUMNS}"
            df = df[CLUSTER_COLUMNS]
            df = process_clusterdf(df)
            st.session_state["processed_df"] = df
            st.session_state["text_column"] = "text"
        else:
            df = pd.read_csv(path_to_gcs)
        file_name = path_to_gcs
        file_id = (
            int(hashlib.sha256(file_name.encode("utf-8")).hexdigest(), 16) % 10**8
        )
        file_size = len(df)

# if uploaded_file is not None or path_to_gcs is not None:
if st.session_state["processed_df"] is not None:
    # Create a file UUID to be used for file specific caching
    file_uuid = f'{file_name.split(".")[0]}_{file_id}_{file_size}'

    st.session_state["df"] = df
    st.session_state["import_done"] = True
    # st.session_state["co_cluster_output"] = co_cluster_output
    # st.session_state["file_uuid"] = file_uuid
    if "pca_df" not in st.session_state:
        with st.spinner("Reducing dimensionality of embeddings"):
            pca_df = compute_pca(embed_df)
            st.session_state["pca_df"] = pca_df
            st.success("Success")
    st.write(df)
    next = st.button("Next")
    # Delete state of text column on each new dataset import
    # if "text_column" in st.session_state and not co_cluster_output:
    #     del st.session_state["text_column"]
    if next:
        switch_page("Visualize")
    # else:
    #     switch_page("Cluster")
