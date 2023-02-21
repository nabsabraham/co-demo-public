import streamlit as st
from streamlit_toggle import st_toggle_switch

import os
import torch
from sentence_transformers import util
from topically import Topically
import plotly.express as px
import umap, hdbscan
from streamlit_extras.switch_page_button import switch_page
from utils import streamlit_header_and_footer_setup, top_menu, switch_menu
from cluster import coCluster, coClusterTest

from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_PAGE = "Cluster"
st.set_page_config(
    page_title="Import/compute Cohere clusters",
    page_icon="📈",
    initial_sidebar_state=config["sidebar"],
    layout=config["layout"],
)
# Render header/footer
streamlit_header_and_footer_setup()
# Set menu index
menu = top_menu(2)
st.session_state["current_page"] = CURRENT_PAGE
if menu:
    switch_menu(menu)
st.title("Cluster Settings")
# cluster_fail = False

# # Get dataframe from state
# try:
#     import_done = st.session_state["import_done"]
#     dataframe = st.session_state["df"]
#     columns = list(dataframe.columns)

#     api_key_env = os.environ.get("cohere_api_key")
#     api_key_env = init_ss("cohere_api_key", api_key_env)
#     # FIXME comment for deploy
#     api_key = api_key_env
api_key = st.text_input("Cohere API Key", type="password")
embeddings_file = st.text_input("signed URL to embeddings file")

min_cluster_size = st.slider("Min cluster size", 10, 100, 10)
similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.85)
cluster_algo = st.selectbox("Clustering Algorithm", ("hdbscan", "fastcluster"))

# assert embeddings_file.startswith("gs://"), "must be a path to a cloud bucket"
st.session_state["cohere_api_key"] = api_key

cluster_button = st.button("Submit")
if cluster_button:
    print(f"submitted")
    cluster_engine = coClusterTest(api_key=api_key)

    with st.spinner("Waiting for clustering completion"):
        response = cluster_engine.submit_job(
            embeddings_file, similarity_threshold, min_cluster_size
        )


#     # Initialize session states
#     text_column = init_ss("text_column", columns[0])
#     # Default option or get previous user saved state for algo
#     text_column_idx = columns.index(text_column)

#     text_column = st.selectbox(
#         "Text Column",
#         columns,
#         text_column_idx,
#     )
#     st.session_state["text_column"] = text_column

#     model = st.selectbox("Cohere Model", ("large", "small"))

#     clustering_algos = ["Fast Cluster", "HDBSCAN"]

#     # Initialize session states
#     clustering_algo = init_ss("clustering_algo", "Fast Cluster")
#     # Fast Cluster params
#     fc_threshold = init_ss("fc_threshold", 0.7)
#     fc_min_community_size = init_ss("fc_min_community_size", 10)
#     # BERTopic params
#     bt_n_neighbors = init_ss("bt_n_neighbors", 15)
#     bt_n_components = init_ss("bt_n_components", 5)
#     bt_min_cluster_size = init_ss("bt_min_cluster_size", 10)
#     # Default option or get previous user saved state for algo
#     clustering_idx = clustering_algos.index(clustering_algo)

#     clustering_algo = st.selectbox(
#         "Clustering Algorithm", clustering_algos, index=clustering_idx
#     )

#     st.session_state["clustering_algo"] = clustering_algo

#     if clustering_algo == "Fast Cluster":
#         fc_threshold = st.slider(
#             "Threshold",
#             min_value=0.3,
#             max_value=0.9,
#             value=fc_threshold,
#             on_change=on_value_change("fc_threshold", fc_threshold),
#         )
#         fc_min_community_size = st.text_input(
#             "Minimum Community Size", fc_min_community_size
#         )
#         st.session_state["fc_min_community_size"] = fc_min_community_size

#     elif clustering_algo == "HDBSCAN":
#         bt_n_neighbors = st.text_input("Number of neighbors", bt_n_neighbors)
#         bt_n_components = st.text_input("Number of components", bt_n_components)
#         bt_min_cluster_size = st.text_input("Minimum cluster size", bt_min_cluster_size)
#         st.session_state["bt_n_neighbors"] = bt_n_neighbors
#         st.session_state["bt_n_components"] = bt_n_components
#         st.session_state["bt_min_cluster_size"] = bt_min_cluster_size

#     enable_topically_value = init_ss("enable_topically_value", True)
#     enable_topically = st_toggle_switch(
#         label="Enable Topically?",
#         key="enable_topically",
#         default_value=enable_topically_value,
#         label_after=True,
#         inactive_color="#D3D3D3",  # optional
#         active_color="#11567f",  # optional
#         track_color="#29B5E8",  # optional
#     )
#     cluster_button = st.button("Cluster")
#     if cluster_button:
#         try:
#             with st.spinner("Getting Embeddings"):
#                 pickle_filename = f"{st.session_state.file_uuid}.pkl"
#                 if os.path.isfile(pickle_filename):
#                     embeddings = pickle.load(open(pickle_filename, "rb"))
#                 else:
#                     dataframe[text_column] = dataframe[text_column].astype(str)
#                     text = list(dataframe[text_column].values)
#                     embeddings = get_embeddings(model, text, api_key)
#                     pickle.dump(embeddings, open(pickle_filename, "wb"))

#             with st.spinner("Running Clustering"):
#                 if clustering_algo == "Fast Cluster":
#                     umap_embeddings = generate_umap_embeddings(
#                         embeddings=embeddings, n_components=2
#                     )
#                     # umap_embeddings = get_pca_embeddings(embeddings=embeddings)
#                     embeddings = torch.tensor(embeddings).to(device)
#                     # This returns nested list of indices of clusters i.e. [[cluster 0 indices],[cluster 1 indices]...]
#                     fast_clusters = community_detection(
#                         embeddings,
#                         min_community_size=int(fc_min_community_size),
#                         threshold=float(fc_threshold),
#                     )
#                     cluster_labels = [-1] * embeddings.shape[0]
#                     for i in range(len(fast_clusters)):
#                         for j in fast_clusters[i]:
#                             cluster_labels[j] = i
#                 else:
#                     umap_embeddings, clusters = generate_clusters_berttopic(
#                         embeddings,
#                         n_neighbors=int(bt_n_neighbors),
#                         n_components=int(bt_n_components),
#                         min_cluster_size=int(bt_min_cluster_size),
#                         random_state=42,
#                     )

#                     cluster_labels = clusters.labels_

#                 dataframe["Cluster"] = cluster_labels
#         except Exception as e:
#             st.write(f"Error: {e}")
#             cluster_fail = True

#         if enable_topically:
#             st.session_state["enable_topically_value"] = True
#             try:
#                 with st.spinner("Running Topically"):
#                     texts = dataframe[text_column].values
#                     max_length = 200
#                     texts = [
#                         text[:max_length] if len(text) > max_length else text
#                         for text in texts
#                     ]
#                     app = Topically(api_key)
#                     cluster_names = app.name_topics((texts, dataframe["Cluster"]))[0]
#                     dataframe["Cluster Name"] = cluster_names
#                     dataframe.loc[
#                         dataframe["Cluster"] == -1, "Cluster Name"
#                     ] = "Unknown"
#             except Exception as e:
#                 st.write(f"Error: {e}")
#                 cluster_fail = True
#         else:
#             dataframe["Cluster Name"] = dataframe["Cluster"]

#         df = dataframe[[text_column, "Cluster", "Cluster Name"]]

#         if not cluster_fail:
#             df["x"] = umap_embeddings[:, 0]
#             df["y"] = umap_embeddings[:, 1]
#             st.session_state["processed_df"] = df
#             switch_page("Charts")

# except Exception as e:
#     if not cluster_fail:
#         st.write(f"Please import a dataset first")
#         import_data = st.button("Import Data")
#         if import_data:
#             switch_page("Import Data")
