#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pandas as pd
import plotly.express as px


# In[3]:


#pip install streamlit


# In[4]:


import streamlit as st


df = pd.read_excel("CM_Elgin.xlsx")

# Pivot the dataframe
pivot_df = df.pivot(index='Player', columns='Attribute', values='Value')


# def create_radar_chart(df, players, id_column, title=None, max_values=None, padding=1.25):
#     df_selected = df.loc[players]
#     categories = df_selected.columns.tolist()
#     data = df_selected.to_dict(orient='list')
#     ids = df_selected.index.tolist()
    
#     # Check and handle zero division or NaNs in max_values
#     if max_values is None:
#         max_values = {key: padding * max(value) for key, value in data.items()}
#     else:
#         for key, max_val in max_values.items():
#             if max_val == 0 or np.isnan(max_val):
#                 max_values[key] = padding * max(data[key])
                
#     # Normalize the data
#     normalized_data = {}
#     for key, value in data.items():
#         if max_values[key] != 0:  # Avoid division by zero
#             normalized_data[key] = np.array(value) / max_values[key]
#         else:
#             normalized_data[key] = np.zeros(len(value))  # Handle zero division case
    
#     num_vars = len(data.keys())
#     ticks = list(data.keys())
#     ticks += ticks[:1]
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    
#     # Plotting radar chart
#     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
#     for i, model_name in enumerate(ids):
#         values = [normalized_data[key][i] for key in data.keys()]
#         actual_values = [data[key][i] for key in data.keys()]
#         values += values[:1]  # Close the plot for a better look
#         ax.plot(angles, values, label=model_name)
#         ax.fill(angles, values, alpha=0.15)
#         for angle, value, actual_value in zip(angles, values, actual_values):
#             ax.text(angle, value, f'{actual_value:.1f}', ha='center', va='bottom', fontsize=10, color='gray')
            
#     ax.fill(angles, np.ones(num_vars + 1), alpha=0.05)
    
#     ax.set_yticklabels([])
#     ax.set_xticks(angles)
#     ax.set_xticklabels(ticks)
#     ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
#     if title is not None:
#         plt.suptitle(title)
    
#     return fig

def create_radar_chart(df, players, id_column, title=None, max_values=None, padding=1.25):
    df_selected = df.loc[players]
    categories = df_selected.columns.tolist()
    data = df_selected.to_dict(orient='list')
    ids = df_selected.index.tolist()
    
    if max_values is None:
        max_values = {key: padding * max(value) for key, value in data.items()}
    else:
        for key, max_val in max_values.items():
            if max_val == 0 or np.isnan(max_val):
                max_values[key] = padding * max(data[key])
                
    normalized_data = {}
    for key, value in data.items():
        if max_values[key] != 0:
            normalized_data[key] = np.array(value) / max_values[key]
        else:
            normalized_data[key] = np.zeros(len(value))
    
    fig = go.Figure()

    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]
        values += values[:1]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=model_name
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=title
       )
    
    return fig

# def create_pizza_plot(df, players, categories, title):
#     N = len(categories)
#     angles = np.linspace(0, 2 * pi, N, endpoint=False).tolist()
#     angles_mids = np.linspace(0, 2 * pi, N, endpoint=False) + (angles[1] / 2)

#     fig = plt.figure(figsize=(8, 8))
#     ax = plt.subplot(111, polar=True)
#     ax.set_theta_offset(pi / 2)
#     ax.set_theta_direction(-1)
#     ax.set_xticks(angles_mids)
#     ax.set_xticklabels(categories)
#     ax.xaxis.set_minor_locator(plt.FixedLocator(angles))

#     # Draw ylabels
#     ax.set_rlabel_position(0)
#     ax.set_yticks([20, 40, 60, 80, 100])
#     ax.set_yticklabels(["20", "40", "60", "80", "100"], color="black", size=8)
#     ax.set_ylim(0, 100)

#     width = angles[1] - angles[0]

#     for player in players:
#         values = df.loc[player, categories].values.flatten().tolist()
#         ax.bar(angles_mids, values, width=width, alpha=0.5, edgecolor='k', linewidth=1, label=player)

#     ax.grid(True, axis='x', which='minor')
#     ax.grid(False, axis='x', which='major')
#     ax.grid(True, axis='y', which='major')
#     ax.legend(loc='upper left', bbox_to_anchor=(0.9, 1.1))
#     plt.title(title)

#     return fig

# def create_pizza_plot(df, players, categories, title):
#     fig = go.Figure()

#     for player in players:
#         values = df.loc[player, categories].values.flatten().tolist()
#         fig.add_trace(go.Barpolar(
#             r=values,
#             theta=categories,
#             width=[360 / len(categories)] * len(categories),
#             name=player,
#             opacity=0.5
#         ))

#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 100]
#             )),
#         showlegend=True,
#         title=title
#        )
    
#     return fig


# Streamlit app
st.title('Player Performance Dashboard')
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

    # pizza_fig=create_pizza_plot(df_position, players, categories=['Accurate forward passes, %', 'Accurate passes to final third, %', 'Accurate passes, %',
                        # 'Accurate progressive passes, %','Aerial duels won, %',], title='Pizza Plot for Selected Players')

    # Create radar chart for selected players
    radar_fig =create_radar_chart(df_position, players, id_column='Player', title=f'Radar Chart for Selected {position} Players and League Average')

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(radar_fig)
    # with col2:
    #     st.plotly_chart(pizza_fig)

    fig2 = px.scatter(df_filtered, x='Successful defensive actions per 90', y='Fouls per 90',
                     color='Player',text='Player', title=f'{position} Defensive ability')
  
    fig2.update_traces(textposition='top center')
    

    df_filtered2 = df_filtered.sort_values(by='Aerial duels won, %', ascending=False)

    # Melt the dataframe to long format for stacking
    df_melted = df_filtered2.melt(id_vars='Player', value_vars=['Aerial duels per 90', 'Aerial duels won, %'], var_name='Metric', value_name='Value')

    # Create stacked bar chart
    fig3 = px.bar(df_melted, x='Player', y='Value', color='Metric', title=f'{position} Aerial Duels and Aerial Duels Won % (Stacked)')
    

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.plotly_chart(fig2)
    with col2:
        st.plotly_chart(fig3)

# players = st.selectbox('Select a player:', options=pivot_df.index.tolist())

# # Filter data for selected player
# #selected_data = pivot_df.loc[[player_selected]]
# st.subheader('Radar Chart for Selected Player and League Average')
# # Create radar chart for selected player
# create_radar_chart(pivot_df, [players, 'League Two Average'], id_column='Player', title=f'Radar Chart for {players} and League Average')



