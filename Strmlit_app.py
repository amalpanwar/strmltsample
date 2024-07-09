#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
#from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import getpass
import os
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_mistralai import ChatMistralAI
#import pandas as pd
import json
import bs4
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# In[3]:


#pip install streamlit


# In[4]:


import streamlit as st


df = pd.read_excel("CM_Elgin.xlsx")

# Pivot the dataframe
pivot_df = df.pivot(index='Player', columns='Attribute', values='Value')


def create_radar_chart(df, players, id_column, title=None, max_values=None, padding=1.25):
    df_selected = df.loc[players]
    categories = df_selected.columns.tolist()
    data = df_selected.to_dict(orient='list')
    ids = df_selected.index.tolist()
    
    # Check and handle zero division or NaNs in max_values
    if max_values is None:
        max_values = {key: padding * max(value) for key, value in data.items()}
    else:
        for key, max_val in max_values.items():
            if max_val == 0 or np.isnan(max_val):
                max_values[key] = padding * max(data[key])
                
    # Normalize the data
    normalized_data = {}
    for key, value in data.items():
        if max_values[key] != 0:  # Avoid division by zero
            normalized_data[key] = np.array(value) / max_values[key]
        else:
            normalized_data[key] = np.zeros(len(value))  # Handle zero division case
    
    num_vars = len(data.keys())
    ticks = list(data.keys())
    ticks += ticks[:1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    
    # Plotting radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('black')  # Set figure background to black
    ax.set_facecolor('grey') 
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]
        values += values[:1]  # Close the plot for a better look
        ax.plot(angles, values, label=model_name)
        ax.fill(angles, values, alpha=0.15)
        for angle, value, actual_value in zip(angles, values, actual_values):
            ax.text(angle, value, f'{actual_value:.1f}', ha='center', va='bottom', fontsize=10, color='black')
            
    ax.fill(angles, np.ones(num_vars + 1), alpha=0.05)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(ticks, color='white',fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), facecolor='black', edgecolor='white', labelcolor='white')

    if title is not None:
        plt.suptitle(title,color='white',fontsize=14)
    
    return fig

# def create_radar_chart(df, players, id_column, title=None, max_values=None, padding=1.25):
#     df_selected = df.loc[players]
#     categories = df_selected.columns.tolist()
#     data = df_selected.to_dict(orient='list')
#     ids = df_selected.index.tolist()
    
#     if max_values is None:
#         max_values = {key: padding * max(value) for key, value in data.items()}
#     else:
#         for key, max_val in max_values.items():
#             if max_val == 0 or np.isnan(max_val):
#                 max_values[key] = padding * max(data[key])
                
#     normalized_data = {}
#     for key, value in data.items():
#         if max_values[key] != 0:
#             normalized_data[key] = np.array(value) / max_values[key]
#         else:
#             normalized_data[key] = np.zeros(len(value))
    
#     fig = go.Figure()

#     for i, model_name in enumerate(ids):
#         values = [normalized_data[key][i] for key in data.keys()]
#         actual_values = [data[key][i] for key in data.keys()]
#         values += values[:1]
#         fig.add_trace(go.Scatterpolar(
#             r=values,
#             theta=categories + [categories[0]],
#             fill='toself',
#             name=model_name
#         ))

#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 1]
#             )),
#         showlegend=True,
#         title=title
#        )
    
#     return fig

def create_pizza_plot(df, players, categories, title):
    N = len(categories)
    angles = np.linspace(0, 2 * pi, N, endpoint=False).tolist()
    angles_mids = np.linspace(0, 2 * pi, N, endpoint=False) + (angles[1] / 2)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    fig.patch.set_facecolor('black')  # Set figure background to black
    ax.set_facecolor('grey') 
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles_mids)
    ax.set_xticklabels(categories,color='white',fontsize=10)
    ax.xaxis.set_minor_locator(plt.FixedLocator(angles))

    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="black", size=8)
    ax.set_ylim(0, 100)

    width = angles[1] - angles[0]

    for player in players:
        values = df.loc[player, categories].values.flatten().tolist()
        ax.bar(angles_mids, values, width=width, alpha=0.5, edgecolor='k', linewidth=1, label=player)

    ax.grid(True, axis='x', which='minor')
    ax.grid(False, axis='x', which='major')
    ax.grid(True, axis='y', which='major')
    ax.legend(loc='upper left', bbox_to_anchor=(0.9, 1.1),facecolor='black', edgecolor='white', labelcolor='white')
    plt.title(title,color='white',fontsize=14)

    return fig


# RAG Pipeline for Chatting
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["MISTRAL_API_KEY"] = 'Eercwo3eC59ZehvjZXf3GSs57vR8B8sE'
llm = ChatMistralAI(model="mistral-large-latest")

# Loading document through loader
loader = UnstructuredFileLoader(file_path="CM_ElginFC.xlsx")
data = loader.load()

#formatting data to ready for LLM model
combined_text = '\n\n\n'.join(doc.page_content.strip() for doc in data)
player_data_blocks = combined_text.split('\n\n\n')
header = player_data_blocks[0]

class Document:
    def __init__(self, metadata, page_content):
        self.metadata = metadata
        self.page_content = page_content

# Create a list of Document objects
# Create a list of Document objects, ensuring the header is included in each block's content
documents = [
    Document(metadata={'source': 'CM_ElginFC.xlsx', 'header': header}, page_content=block)
    for block in player_data_blocks[1:]  # Skip the first block as it's the header
]

vectorstore = Chroma.from_documents(documents=documents,embedding=HuggingFaceHubEmbeddings(huggingfacehub_api_token='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'))
retriever = vectorstore.as_retriever(search_type="mmr",
    search_kwargs={'k': 20, 'fetch_k':50})

# Preparing Prompt for Q/A
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



# Streamlit app
st.title('Player Performance Dashboard')
user_prompt = st.sidebar.text_input("Enter your query:")
if user_prompt:
    # Get response from RAG chain
    response = rag_chain.invoke({"input": user_prompt})
    st.write(response["answer"])
default_position_index = ["GK","FB","CB","CM","CAM","Winger","CF"].index('CM')
position = st.sidebar.selectbox('Select position:', options=["GK","FB","CB","CM","CAM","Winger","CF"],index=default_position_index)

# Initialize df_position and default player list
# Initialize df_position and default player list
df_position = None

# Determine the dataframe to use based on selected position
if position == 'CM':
    df_position = pivot_df
# elif position == 'GK':
#     df_position = df_GK
# Add other positions here with elif statements

# Ensure df_position is selected
if df_position is not None:
    # Dropdown menu for player selection based on position
    players = st.sidebar.multiselect('Select players:', options=df_position.index.tolist(), default=['League Two Average'])
    df_filtered = df_position.loc[players].reset_index()

    # Create point facet graph
    fig = px.scatter(df_filtered, x='Passes per 90', y=['Progressive passes per 90', 'Passes to final third per 90'], facet_col='variable',
                     color='Player',text='Player', title=f'{position} passing threats')
    fig.update_layout(
        autosize=True,
        width=1000,
        height=600,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        font=dict(size=8)
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig)
    
    #st.plotly_chart(fig)
    # Ensure 'League Two Average' is included in the list of selected players
    # if 'League Two Average' not in players:
    #     players.append('League Two Average')

    pizza_fig=create_pizza_plot(df_position, players, categories=['Accurate forward passes, %', 'Accurate passes to final third, %', 'Accurate passes, %',
                        'Accurate progressive passes, %','Aerial duels won, %',], title='Pizza Plot for Selected Players')

    # Create radar chart for selected players
    radar_fig =create_radar_chart(df_position, players, id_column='Player', title=f'Radar Chart for Selected {position} Players and League Average')

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(radar_fig)
    with col2:
        st.pyplot(pizza_fig)

    fig2 = px.scatter(df_filtered, x='Successful defensive actions per 90', y='Fouls per 90',
                     color='Player',text='Player', title=f'{position} Defensive ability')
  
    fig2.update_traces(textposition='top center')
    

    df_filtered['Aerial duels won per 90'] = df_filtered['Aerial duels per 90'] * (df_filtered['Aerial duels won, %'] / 100)

    df_filtered2 = df_filtered.sort_values(by='Aerial duels won per 90', ascending=False)

    # Melt the dataframe to long format for stacking
    df_melted = df_filtered2.melt(id_vars='Player', value_vars=['Aerial duels per 90', 'Aerial duels won per 90'], var_name='Metric', value_name='Value')

    # Create stacked bar chart
    fig3 = px.bar(df_melted, x='Player', y='Value', color='Metric', title=f'{position} Aerial ability (Stacked)')
    

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.plotly_chart(fig2)
    with col2:
        st.plotly_chart(fig3)
    # Input field for user prompt
    

# players = st.selectbox('Select a player:', options=pivot_df.index.tolist())

# # Filter data for selected player
# #selected_data = pivot_df.loc[[player_selected]]
# st.subheader('Radar Chart for Selected Player and League Average')
# # Create radar chart for selected player
# create_radar_chart(pivot_df, [players, 'League Two Average'], id_column='Player', title=f'Radar Chart for {players} and League Average')



