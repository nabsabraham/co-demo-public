import ast

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
import torch
from torch import Tensor
from streamlit_extras.switch_page_button import switch_page
import torch
from sentence_transformers import util
import umap, hdbscan
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def init_ss(ss_key, ss_value):
    """
    Initialize session state variable with user provided default
    If present already, just return session state
    """
    if ss_key not in st.session_state:
        st.session_state[ss_key] = ss_value
    return st.session_state[ss_key]


def switch_menu(menu):
    current_page = st.session_state["current_page"]
    if current_page == menu:
        pass
    else:
        print(f"switching from {current_page} to {menu}")
        st.session_state["current_page"] = menu
        # Menu routing
        if menu == "Home":
            switch_page("app")
        elif menu == "Import":
            switch_page("Import")
        # elif menu == "Embed":
        #     switch_page("Embed")
        elif menu == "Cluster":
            switch_page("Cluster")
        elif menu == "Visualize":
            switch_page("Visualize")


def top_menu(index):
    """
    Nav Menu
    """
    menu = option_menu(
        None,
        # ["Home", "Import", "Embed", "Cluster", "Visualize"],
        ["Home", "Import", "Cluster", "Visualize"],
        icons=[
            "house",
            # "card-list",
            "cloud-upload",
            # "explicit",
            "collection",
            "graph-up",
        ],
        menu_icon="cast",
        default_index=index,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "black"},
        },
    )
    return menu


def streamlit_header_and_footer_setup():
    """
    Cohere banner
    """
    st.markdown(
        '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .navbar-nav{
            display: flex;
            flex-direction: column;
        }
        @media only screen and (max-width: 525px) {
            nav{
                flex-direction: column !important;
                gap: 15px !important;
            }    
        }
        </style>
        <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #000; justify-content:space-between;">
            <a href="https://cohere.ai" target="_blank"><svg class="navbar-brand" height="50" viewBox="0 0 1552 446" fill="none" xmlns="http://www.w3.org/2000/svg" class="block w-auto align-top h-11 false">
                <path d="M1377.78 250.565C1390.73 200.116 1424.14 181.954 1451.08 181.954C1480.74 181.954 1509.72 199.107 1513.47 250.565H1377.78ZM1463.69 444.627C1502.9 444.627 1529.84 431.174 1549.95 415.703L1543.82 406.958C1525.07 420.748 1502.22 430.502 1472.9 430.502C1419.71 430.502 1372.66 398.55 1372.66 304.042C1372.66 285.88 1374.02 269.736 1375.39 263.346H1552C1540.75 195.071 1497.45 170.183 1451.08 170.183C1397.21 170.183 1327.66 217.269 1327.66 305.723C1327.66 399.896 1396.53 444.627 1463.69 444.627ZM1151.73 440.255H1244.12V430.838C1218.55 424.784 1217.87 416.712 1217.87 348.437V252.92L1246.85 221.641C1256.4 210.878 1260.83 207.515 1269.35 207.515C1281.29 207.515 1289.81 218.278 1304.13 218.278C1317.43 218.278 1324.93 207.515 1324.93 192.38C1324.93 180.609 1316.4 172.537 1301.4 172.537C1281.63 172.537 1270.04 183.299 1246.85 208.188L1217.87 239.13V170.183L1151.04 205.497V215.251C1174.91 217.269 1174.91 218.95 1174.91 292.27V348.437C1174.91 416.712 1174.23 424.784 1151.73 430.838V440.255ZM954.657 250.565C967.613 200.116 1001.03 181.954 1027.96 181.954C1057.62 181.954 1086.6 199.107 1090.35 250.565H954.657ZM1040.58 444.627C1079.79 444.627 1106.72 431.174 1126.84 415.703L1120.7 406.958C1101.95 420.748 1079.1 430.502 1049.78 430.502C996.594 430.502 949.543 398.55 949.543 304.042C949.543 285.88 950.907 269.736 952.271 263.346H1128.88C1117.63 195.071 1074.33 170.183 1027.96 170.183C974.091 170.183 904.538 217.269 904.538 305.723C904.538 399.896 973.409 444.627 1040.58 444.627ZM554.724 245.184C570.749 245.184 583.023 233.076 583.023 217.269C583.023 201.798 570.749 189.69 554.724 189.69C539.04 189.69 527.107 201.798 527.107 217.269C527.107 233.076 539.04 245.184 554.724 245.184ZM554.724 445.636C570.749 445.636 583.023 434.201 583.023 418.394C583.023 402.586 570.749 390.815 554.724 390.815C539.04 390.815 527.107 402.586 527.107 418.394C527.107 434.201 539.04 445.636 554.724 445.636ZM365.156 433.865C321.856 433.865 283.67 400.232 283.67 309.087C283.67 218.278 321.856 181.618 365.156 181.618C409.139 181.618 447.666 218.278 447.666 309.087C447.666 400.232 409.139 433.865 365.156 433.865ZM365.156 444.964C422.436 444.964 493.353 396.869 493.353 309.087C493.353 221.305 422.436 170.183 365.156 170.183C308.559 170.183 237.641 221.305 237.641 309.087C237.641 396.869 308.559 444.964 365.156 444.964ZM132.629 443.955C172.861 443.955 201.842 428.82 219.571 406.622L213.775 399.559C197.069 417.721 172.861 429.829 141.835 429.829C84.2144 429.829 44.6643 394.178 44.6643 303.369C44.6643 215.587 86.942 182.627 134.334 182.627C155.473 182.627 169.452 190.362 172.861 203.816C174.907 212.56 178.316 229.04 195.705 229.04C209.684 229.04 218.207 219.959 218.207 205.834C218.207 181.954 176.271 170.855 133.652 170.855C71.5993 170.855 0 222.986 0 306.396C0 392.496 63.0756 443.955 132.629 443.955ZM620.186 440.255H711.22V430.838C686.671 424.784 685.989 416.712 685.989 348.437V245.52L705.083 228.368C739.177 197.425 759.634 188.008 778.046 188.008C801.23 188.008 815.209 201.461 815.209 237.449V348.437C815.209 416.712 814.186 424.784 789.638 430.838V440.255H881.012V430.838C858.169 424.784 858.169 416.712 858.169 348.437V239.803C858.169 191.035 832.938 171.528 790.661 171.528C763.726 171.528 737.473 184.981 705.083 214.914L685.989 232.067V0L619.845 33.6329V42.7138C643.03 43.7228 643.03 45.4045 643.03 111.661V348.437C643.03 416.712 642.348 424.784 620.186 430.838V440.255Z" fill="#FFF"></path>
            </svg></a>
            <div style="justify-content: flex-end;">
                <ul class="navbar-nav">
                    <li>
                        <a class="nav-link text-white" href="https://dashboard.cohere.ai/welcome/register" style="background: #917EF3; border-radius: 100px; padding-left: 15px; padding-right: 15px;">Sign up for free</a>
                    </li>
                    <li>
                        <a class="nav-link text-white" href="https://docs.cohere.ai" target="_blank">Read our docs →</a>
                    </li>
                </ul>
            </div>
        </nav>
    """,
        unsafe_allow_html=True,
    )

    # Hide Streamlit style in top right
    hide_st_style = """
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'Cohere Inc';
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    st.markdown(
        f"""
            <style>
                .sidebar .sidebar-content {{
                    width: 0px;
                }}
            </style>
        """,
        unsafe_allow_html=True,
    )


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    Assumes a and b are normalized.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


def community_detection(
    embeddings, threshold=0.75, min_community_size=10, batch_size=1024
):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order.
    """
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)  # (num_samples, embed_dim)

    # normalize all at once to same repeated compute
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    threshold = torch.tensor(threshold, device=embeddings.device)

    extracted_communities = []

    # Maximum size for community
    min_community_size = min(min_community_size, len(embeddings))

    for start_idx in range(0, len(embeddings), batch_size):
        # Compute cosine similarity scores
        cos_scores = cos_sim(embeddings[start_idx : start_idx + batch_size], embeddings)

        # Evren Note:
        # this is fast since it is parallel over the batch dim
        # there may be a way to replace this call (and the torch.where below)
        # with just 1 call to torch.where

        # Minimum size for a community
        top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

        # Filter for rows >= min_threshold
        for i in range(len(top_k_values)):
            if top_k_values[i][-1] >= threshold:
                # avoid the repeated calls to topk and/or sort_max_size
                # growing too big - this should be O(N)
                top_idx_large = torch.where(cos_scores[i] >= threshold)[0]
                extracted_communities.append(top_idx_large.tolist())
        del cos_scores

    # Largest cluster first
    extracted_communities = sorted(
        extracted_communities, key=lambda x: len(x), reverse=True
    )

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for community in extracted_communities:
        community = sorted(community)
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)

    return unique_communities


def on_value_change(value_str, new_value):
    st.session_state[value_str] = new_value


def get_embeddings(model, text, api_key):
    co = cohere.Client(api_key)
    response = co.embed(model=model, texts=text)
    embeddings = response.embeddings
    return embeddings


def generate_umap_embeddings(
    embeddings, n_neighbors=15, n_components=5, random_state=None
):
    umap_embeddings = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=random_state,
    ).fit_transform(embeddings)

    return umap_embeddings


def get_pca_embeddings(embeddings=None, n_components=2):
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings)
    return pca_embeddings


def generate_clusters_berttopic(
    message_embeddings, n_neighbors, n_components, min_cluster_size, random_state=None
):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    """

    umap_embeddings = generate_umap_embeddings(
        message_embeddings, n_neighbors, n_components
    )

    clusters = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit(umap_embeddings)

    return umap_embeddings, clusters


def process_clusterdf(df):
    df["elements"] = df["elements"].apply(lambda x: ast.literal_eval(str(x)))
    df = (
        df.explode(["elements", "text_ids"])
        .rename(
            columns={
                "elements": "text",
                "text_ids": "text_id",
                "description": "Cluster Name",
                "cluster_id": "Cluster",
            }
        )
        .reset_index(drop=True)
    )
    return df


def show_wordcloud(dataframe, column_name):
    """
    Show wordcloud based on user selection
    """
    text = " ".join(review for review in dataframe[column_name])
    wordcloud = WordCloud(
        max_font_size=50, max_words=100, background_color="white"
    ).generate(text)
    plt.figure()
    fig = plt.figure(figsize=(7, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    fig.savefig("figure_name.png")
    from PIL import Image

    image = Image.open("figure_name.png")
    st.image(image)


def rename_clusters(change_fc):
    """
    Rename a cluster based on user selections
    """
    if st.session_state.change_full_cluster:
        df.loc[
            df["Cluster Name"] == st.session_state.old_cluster_label, "Cluster Name"
        ] = st.session_state.new_cluster_label
        st.session_state.change_full_cluster = False
    else:
        df.at[
            st.session_state.row_idx, "Cluster Name"
        ] = st.session_state.new_cluster_label
    st.session_state["processed_df"] = df


def join_pca_cluster_dfs(cluster_df, pca_df):
    merged = cluster_df.merge(right=pca_df, how="inner", on="text_id")
    return merged


@st.experimental_memo
def compute_pca(embed_df) -> pd.DataFrame:
    embed_df["embedding"] = embed_df["embedding"].apply(
        lambda x: np.array(x).astype(np.float16).reshape(1, -1)
    )
    X = np.concatenate(embed_df["embedding"])
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    res = pd.DataFrame(result)
    res = res.reset_index().rename(columns={"index": "text_id", 0: "x", 1: "y"})
    return res


def get_topk_clusters(df, k):
    return df["Cluster"].value_counts()[:k].reset_index()["index"].tolist()
