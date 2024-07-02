#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pandas as pd


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
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]
        values += values[:1]  # Close the plot for a better look
        ax.plot(angles, values, label=model_name)
        ax.fill(angles, values, alpha=0.15)
        for angle, value, actual_value in zip(angles, values, actual_values):
            ax.text(angle, value, f'{actual_value:.1f}', ha='center', va='bottom', fontsize=10, color='gray')
            
    ax.fill(angles, np.ones(num_vars + 1), alpha=0.05)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(ticks)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    if title is not None:
        plt.suptitle(title)
    
    st.pyplot(fig)
# Streamlit app
st.title('Player Performance Radar Chart')
default_position_index = ["GK","FB","CB","CM","CAM","Winger","CF"].index('CM')
position = st.sidebar.selectbox('Select position:', options=["GK","FB","CB","CM","CAM","Winger","CF"],index=default_position_index)

if position == 'CM':
    # Dropdown menu for player selection
    players_CM = st.sidebar.multiselect('Select players:', options=pivot_df.index.tolist(), default=['League Two Average'])

# Filter data for selected player
#selected_data = pivot_df.loc[[player_selected]]

# Create radar chart for selected player
    create_radar_chart(pivot_df, [players_CM, 'League Two Average'], id_column='Player', title='Radar Chart for Selected Players and League Average')

# players = st.selectbox('Select a player:', options=pivot_df.index.tolist())

# # Filter data for selected player
# #selected_data = pivot_df.loc[[player_selected]]
# st.subheader('Radar Chart for Selected Player and League Average')
# # Create radar chart for selected player
# create_radar_chart(pivot_df, [players, 'League Two Average'], id_column='Player', title=f'Radar Chart for {players} and League Average')



