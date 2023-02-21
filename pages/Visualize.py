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

CURRENT_PAGE = "Visualize"
st.set_page_config(
    page_title="Visualize Cohere clusters",
    page_icon="📈",
    initial_sidebar_state=config["sidebar"],
    layout=config["layout"],
)
# Render header/footer
streamlit_header_and_footer_setup()
# Set menu index
menu = top_menu(3)
st.session_state["current_page"] = CURRENT_PAGE
if menu:
    switch_menu(menu)

text_column = "text"
st.title("Visualize cluster data")

# if uploaded_file is not None or path_to_gcs is not None:
if "import_done" in st.session_state and st.session_state["import_done"] == True:
    # Create a file UUID to be used for file specific caching
    # file_uuid = f'{file_name.split(".")[0]}_{file_id}_{file_size}'
    df = st.session_state["processed_df"]
    # st.success("Success")
    # st.write(df)
    st.session_state["df"] = df
    # st.session_state["import_done"] = True
    # st.session_state["co_cluster_output"] = co_cluster_output
    # st.session_state["file_uuid"] = file_uuid
    # next = st.button("Next")
    # Delete state of text column on each new dataset import
    # if "text_column" in st.session_state and not co_cluster_output:
    #     del st.session_state["text_column"]
    # if next:
    #     if co_cluster_output:
    #         switch_page("Charts")
    #     else:
    #         switch_page("Cluster")

    ###################
    # Count of clusters
    ###################
    with st.container():
        k = st.slider("Select top-k clusters", min_value=5, max_value=20, value=10)
        cluster_counts = (
            df["Cluster Name"]
            .value_counts()
            .reset_index()
            .rename(columns={"Cluster Name": "Count", "index": "Cluster Name"})[:k]
            .sort_values(by="Count", ascending=True)
        )
        # Show top N clusters in chart
        st.markdown(
            f"<h2 style='text-align: center; color: black;'>View top-{k} clusters</h2>",
            unsafe_allow_html=True,
        )
        fig = go.Figure(
            [
                go.Bar(
                    y=cluster_counts["Cluster Name"].values,
                    x=cluster_counts["Count"].values,
                    text=cluster_counts["Count"].values,
                    textposition="outside",
                    orientation="h",
                )
            ]
        )
        fig.update_xaxes(title="Count")
        fig.update_yaxes(title="Cluster", tickangle=0, automargin=True)
        fig.update_layout(
            autosize=False,
            width=800,
            height=700,
        )
        config = {"displayModeBar": True, "showlegend": False}
        st.plotly_chart(fig, use_container_width=True, config=config)

        ##################
        # View embeddings
        ##################
        st.markdown(
            f"<h2 style='text-align: center; color: black;'>View top-{k} cluster embeddings </h1>",
            unsafe_allow_html=True,
        )
        if st.session_state["embed_df"] is not None:
            pca_df = st.session_state["pca_df"]
            final_df = join_pca_cluster_dfs(df, pca_df)
            cluster_names = get_topk_clusters(final_df, k)
            tmp = final_df[final_df["Cluster"].isin(set(cluster_names))]

            fig = px.scatter(
                tmp,
                x="x",
                y="y",
                color="Cluster Name",
                hover_name="Cluster Name",
                hover_data={
                    "text": True,
                    "text_id": True,
                    "Cluster Name": False,
                    "x": False,
                    "y": False,
                },
                log_x=True,
                size_max=60,
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(autosize=True)
            config = {"displayModeBar": True, "showlegend": False}
            st.plotly_chart(fig, use_container_width=False, config=config)

    #########################
    # Wordcloud
    # #########################
    # Display 2 columns for a table and wordcloud
    st.title("Inspect/change cluster names")
    with st.container():
        table, plot = st.columns([2, 2])
        with table:
            st.subheader("Table")
            st.write("Click on a row to select a cluster")
            gd = GridOptionsBuilder.from_dataframe(df)
            gd.configure_pagination(enabled=True)
            gd.configure_default_column(editable=False, groupable=True)
            gd.configure_selection(selection_mode="single", use_checkbox=False)
            gridoptions = gd.build()
            grid_table = AgGrid(
                df,
                gridOptions=gridoptions,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=400,
                allow_unsafe_jscode=True,
                theme="streamlit",
            )
            sel_row = grid_table["selected_rows"]
            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                "cluster_results.csv",
                "text/csv",
                key="download-csv",
            )
        with plot:
            st.subheader("Word Cloud")
            if sel_row and "Cluster Name" in sel_row[0]:
                selected_cluster = sel_row[0]["Cluster Name"]
                selected_text = sel_row[0][text_column]
                # Show Wordcloud if row selected
                if selected_text:
                    row_idx = int(sel_row[0]["_selectedRowNodeInfo"]["nodeId"])
                    st.write(f"Selected Text: {selected_text}")
                    # User selected cluster
                    st.session_state.old_cluster_label = selected_cluster
                    # New cluster label if user changes it
                    st.session_state.new_cluster_label = st.text_input(
                        f"Selected Cluster. Edit to change.", selected_cluster
                    )
                    # Option to change full cluster label
                    st.session_state.change_full_cluster = st.checkbox(
                        "Apply to entire cluster"
                    )
                    # Row index to change a specific row
                    st.session_state.row_idx = row_idx
                    # Arg is needed to avoid constant trigger. Not sure why.
                    st.button(
                        "Change Cluster Label",
                        on_click=rename_clusters,
                        args=[st.session_state.change_full_cluster],
                    )
                    # Filter dataframe based on selected cluster
                    filt_df = df[df["Cluster Name"] == selected_cluster]
                    show_wordcloud(filt_df, text_column)
else:
    switch_page("Import")
