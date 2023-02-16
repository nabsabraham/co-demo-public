import streamlit as st
from streamlit_toggle import st_toggle_switch
from sentence_transformers import util
import plotly.express as px
from streamlit_extras.switch_page_button import switch_page
from utils import streamlit_header_and_footer_setup, top_menu
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import random

st.set_page_config(
    page_title="Charts",
    page_icon="📈",
    initial_sidebar_state='collapsed',
    layout="wide"
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
elif menu == 'Cluster':
    switch_page('Cluster')


def rename_clusters(change_fc):
    """
    Rename a cluster based on user selections
    """
    if st.session_state.change_full_cluster:
        df.loc[df['Cluster Name'] == st.session_state.old_cluster_label, 'Cluster Name'] = st.session_state.new_cluster_label
        st.session_state.change_full_cluster = False
    else:
        df.at[st.session_state.row_idx,'Cluster Name']= st.session_state.new_cluster_label
    st.session_state['processed_df'] = df 

def show_wordcloud(dataframe, column_name):
    """
    Show wordcloud based on user selection
    """
    text = " ".join(review for review in dataframe[column_name])
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    plt.figure()
    fig = plt.figure(figsize=(7,4))  
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    fig.savefig("figure_name.png") 
    from PIL import Image
    image = Image.open('figure_name.png')
    st.image(image)

try:
    df = st.session_state['processed_df']
    text_column = st.session_state['text_column']
    num_clusters = df['Cluster'].nunique()
    if st.session_state['co_cluster_output']:
        cluster_df = df.groupby('Cluster Name')['keywords'].apply(list)

    # Count of clusters
    top_n = 20
    cluster_counts = df['Cluster Name'].value_counts().reset_index().rename(columns={'Cluster Name': 'Count', 'index': 'Cluster Name'})[:top_n].sort_values(by='Count', ascending=True)
   
    # Show top N clusters in chart
    st.markdown("<h2 style='text-align: center; color: black;'>Top 20 Clusters</h1>", unsafe_allow_html=True)    
    fig = go.Figure([
        go.Bar(
            y=cluster_counts['Cluster Name'].values,
            x=cluster_counts['Count'].values,
            text = cluster_counts['Count'].values,
            textposition='outside',
            orientation='h'
            )
        ]
    )
    fig.update_xaxes(title='Count')
    # Automargin allows clipping of text
    fig.update_yaxes(title='Cluster', tickangle = 0,automargin = True)
    fig.update_layout(
        autosize=False,
        width=800,
        height=700,
    )
    #fig = px.pie(cluster_counts,values='Count', names='Cluster Name')    
    config = {'displayModeBar': True, 'showlegend': False}
    st.plotly_chart(fig, use_container_width=True, config=config)

    # If not co.cluster output, we have embeddings as well. Show embeddings scatter plot
    if not st.session_state['co_cluster_output']:
        clustering_algo = st.session_state['clustering_algo'] 
        # Show embedding plot only for BERTopic flow
        if clustering_algo == 'HDBSCAN' or clustering_algo == 'Fast Cluster':
            df_filt = df[df['Cluster'] != -1]
            df_filt['Cluster'] = df_filt['Cluster'].astype(str)
            st.markdown("<h2 style='text-align: center; color: black;'>Clustering Plot</h1>", unsafe_allow_html=True)
            fig = px.scatter(
                df_filt, x='x', y="y", 
                color='Cluster', 
                hover_data=[text_column, 'Cluster Name'],
                )
            fig.update_yaxes(showticklabels=False, title='', zeroline=False, showgrid=False)
            fig.update_xaxes(showticklabels=False, title='', zeroline=False, showgrid=False)
            config = {'displayModeBar': True}
            st.plotly_chart(fig, use_container_width=True, config=config)

    if st.session_state['co_cluster_output']:
        df = df[['Cluster Name', 'keywords', 'Cluster', text_column]]
    else:
        df = df[['Cluster Name', 'Cluster', text_column]]

    # Display 2 columns for a table and wordcloud
    data_container = st.container()
    with data_container:
        table, plot = st.columns([2,2])
        with table:
            st.subheader("Table")
            st.write("Click on a row to select a cluster")
            gd = GridOptionsBuilder.from_dataframe(df)
            gd.configure_pagination(enabled=True)
            gd.configure_default_column(editable=False,groupable=True)
            gd.configure_selection(selection_mode='single',use_checkbox=False)
            gridoptions = gd.build()
            grid_table = AgGrid(df,gridOptions=gridoptions,
                                update_mode= GridUpdateMode.SELECTION_CHANGED,
                                height = 400,
                                allow_unsafe_jscode=True,
                                theme = 'streamlit')
            sel_row = grid_table["selected_rows"]
            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                "cluster_results.csv",
                "text/csv",
                key='download-csv'
            )
        with plot:
            st.subheader("Word Cloud")
            if sel_row and 'Cluster Name' in sel_row[0]:
                selected_cluster = sel_row[0]['Cluster Name']
                selected_text = sel_row[0][text_column]                
                # Show Wordcloud if row selected               
                if selected_text:
                    row_idx = int(sel_row[0]['_selectedRowNodeInfo']['nodeId'])
                    st.write(f'Selected Text: {selected_text}')
                    # User selected cluster
                    st.session_state.old_cluster_label = selected_cluster
                    # New cluster label if user changes it
                    st.session_state.new_cluster_label = st.text_input(f'Selected Cluster. Edit to change.', selected_cluster)
                    # Option to change full cluster label
                    st.session_state.change_full_cluster = st.checkbox('Apply to entire cluster')
                    # Row index to change a specific row
                    st.session_state.row_idx = row_idx
                    # Arg is needed to avoid constant trigger. Not sure why.
                    st.button('Change Cluster Label', on_click = rename_clusters, args=[st.session_state.change_full_cluster])                    
                    # Filter dataframe based on selected cluster
                    filt_df = df[df['Cluster Name'] == selected_cluster]
                    show_wordcloud(filt_df, text_column)

except Exception as e:
    st.write(f'Please import a dataset first')
    import_data = st.button('Import Data')
    if import_data:
        switch_page('Import Data')

