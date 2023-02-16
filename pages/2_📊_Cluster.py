import streamlit as st
from streamlit_toggle import st_toggle_switch
import time
import cohere
import os
import torch
from sentence_transformers import util
from topically import Topically
import plotly.express as px
import umap, hdbscan
from streamlit_extras.switch_page_button import switch_page
from utils import streamlit_header_and_footer_setup, top_menu, init_ss, community_detection
import pickle
from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_embeddings(model, text, api_key):
    co = cohere.Client(api_key)
    response = co.embed(
        model=model,
        texts=text
    )
    embeddings = response.embeddings
    return embeddings

def generate_umap_embeddings(embeddings,
                      n_neighbors=15,
                      n_components=5,
                      random_state=None):

    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors,
                                 n_components=n_components,
                                 metric='cosine',
                                 random_state=random_state)
                       .fit_transform(embeddings))


    return umap_embeddings

def get_pca_embeddings(embeddings=None, n_components=2):
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings)
    return pca_embeddings

def generate_clusters_berttopic(message_embeddings,
                      n_neighbors,
                      n_components,
                      min_cluster_size,
                      random_state=None):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    """

    umap_embeddings = generate_umap_embeddings(message_embeddings, n_neighbors, n_components)

    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               metric='euclidean',
                               cluster_selection_method='eom').fit(umap_embeddings)

    return umap_embeddings, clusters

def on_value_change(value_str, new_value):
    st.session_state[value_str]= new_value

st.set_page_config(
    page_title="Cluster",
    page_icon="📈",
    initial_sidebar_state='collapsed'
    )

# Render header/footer
streamlit_header_and_footer_setup()
# Set menu index 
menu = top_menu(2)
# Menu routing
if menu == 'Import':
    switch_page('Import Data')
elif menu == 'Home':
    switch_page('app')
elif menu == 'Charts':
    switch_page('Charts')

st.title('Cluster Settings')
cluster_fail = False

# Get dataframe from state
try:
    import_done = st.session_state['import_done']
    dataframe = st.session_state['df']
    columns = list(dataframe.columns)

    api_key_env = os.environ.get('cohere_api_key')
    api_key_env = init_ss('cohere_api_key', api_key_env)
    # FIXME comment for deploy 
    api_key = api_key_env
    #api_key = st.text_input('Cohere API Key', type='password', value=api_key_env)
    #st.session_state['cohere_api_key'] = api_key

    # Initialize session states
    text_column = init_ss('text_column', columns[0])
    # Default option or get previous user saved state for algo
    text_column_idx = columns.index(text_column)

    text_column = st.selectbox(
        'Text Column',
        columns, 
        text_column_idx,
    )
    st.session_state['text_column'] = text_column

    model = st.selectbox(
        'Cohere Model',
        ('large','small')
    )

    clustering_algos = ['Fast Cluster', 'HDBSCAN']

    # Initialize session states
    clustering_algo = init_ss('clustering_algo', 'Fast Cluster')
    # Fast Cluster params
    fc_threshold = init_ss('fc_threshold', 0.7)
    fc_min_community_size = init_ss('fc_min_community_size', 10)
    # BERTopic params
    bt_n_neighbors = init_ss('bt_n_neighbors', 15)
    bt_n_components = init_ss('bt_n_components', 5)
    bt_min_cluster_size= init_ss('bt_min_cluster_size', 10)
    # Default option or get previous user saved state for algo
    clustering_idx = clustering_algos.index(clustering_algo)

    clustering_algo = st.selectbox(
        'Clustering Algorithm',
        clustering_algos,
        index = clustering_idx
    )

    st.session_state['clustering_algo'] = clustering_algo

    if clustering_algo == 'Fast Cluster':       
        fc_threshold = st.slider(
            'Threshold',
             min_value=0.3, max_value=0.9,
             value=fc_threshold,
             on_change=on_value_change('fc_threshold', fc_threshold)
        )
        fc_min_community_size = st.text_input('Minimum Community Size', fc_min_community_size)
        st.session_state['fc_min_community_size'] = fc_min_community_size

    elif clustering_algo == 'HDBSCAN':
        bt_n_neighbors = st.text_input('Number of neighbors', bt_n_neighbors)
        bt_n_components = st.text_input('Number of components', bt_n_components)
        bt_min_cluster_size = st.text_input('Minimum cluster size', bt_min_cluster_size)
        st.session_state['bt_n_neighbors'] = bt_n_neighbors
        st.session_state['bt_n_components'] = bt_n_components
        st.session_state['bt_min_cluster_size'] = bt_min_cluster_size

    enable_topically_value = init_ss('enable_topically_value', True)
    enable_topically = st_toggle_switch(
        label="Enable Topically?",
        key="enable_topically",
        default_value=enable_topically_value,
        label_after=True,
        inactive_color="#D3D3D3",  # optional
        active_color="#11567f",  # optional
        track_color="#29B5E8",  # optional
    )
    cluster_button = st.button('Cluster')
    if cluster_button:
        try:
            with st.spinner("Getting Embeddings"):  
                pickle_filename = f'{st.session_state.file_uuid}.pkl'
                if os.path.isfile(pickle_filename):
                    embeddings = pickle.load(open(pickle_filename, "rb" ))
                else:
                    dataframe[text_column] = dataframe[text_column].astype(str)
                    text = list(dataframe[text_column].values)
                    embeddings = get_embeddings(model, text, api_key)
                    pickle.dump( embeddings, open(pickle_filename, "wb" ) )

            with st.spinner("Running Clustering"):  
                if clustering_algo == 'Fast Cluster':
                    umap_embeddings = generate_umap_embeddings(embeddings=embeddings, n_components=2)
                    #umap_embeddings = get_pca_embeddings(embeddings=embeddings)
                    embeddings = torch.tensor(embeddings).to(device)
                    # This returns nested list of indices of clusters i.e. [[cluster 0 indices],[cluster 1 indices]...]
                    fast_clusters = community_detection(embeddings, min_community_size=int(fc_min_community_size), threshold=float(fc_threshold))
                    cluster_labels = [-1] * embeddings.shape[0]
                    for i in range(len(fast_clusters)):
                        for j in fast_clusters[i]:
                            cluster_labels[j] = i
                else:
                    umap_embeddings, clusters = generate_clusters_berttopic(
                        embeddings,
                        n_neighbors=int(bt_n_neighbors),
                        n_components=int(bt_n_components),
                        min_cluster_size=int(bt_min_cluster_size),
                        random_state=42
                        )

                    cluster_labels = clusters.labels_
                    
                dataframe['Cluster'] = cluster_labels
        except Exception as e:
            st.write(f'Error: {e}')
            cluster_fail = True

        if enable_topically:
            st.session_state['enable_topically_value'] = True
            try:
                with st.spinner("Running Topically"):  
                    texts = dataframe[text_column].values
                    max_length = 200
                    texts = [text[:max_length] if len(text) > max_length else text for text in texts]
                    app = Topically(api_key)
                    cluster_names = app.name_topics((texts, dataframe['Cluster']))[0]
                    dataframe['Cluster Name'] = cluster_names
                    dataframe.loc[dataframe['Cluster'] == -1, 'Cluster Name'] = 'Unknown'
            except Exception as e:
                st.write(f'Error: {e}')
                cluster_fail = True       
        else:
            dataframe['Cluster Name'] = dataframe['Cluster']

        df = dataframe[[text_column, 'Cluster', 'Cluster Name']]

        if not cluster_fail:
            df['x'] = umap_embeddings[:, 0]
            df['y'] = umap_embeddings[:, 1]      
            st.session_state['processed_df'] = df
            switch_page('Charts')
            
except Exception as e:
    if not cluster_fail:
        st.write(f'Please import a dataset first')
        import_data = st.button('Import Data')
        if import_data:
            switch_page('Import Data')
