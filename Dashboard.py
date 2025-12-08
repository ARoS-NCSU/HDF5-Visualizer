import streamlit as st
import h5py
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import helperFunctions as hf
import pandas as pd

st.title("H5 Data Viewer")

# File uploader
uploaded_file = st.file_uploader("Upload HDF5 file", type=['hdf5', 'h5'])

if uploaded_file is not None:
    # Load HDF5 file
    with h5py.File(uploaded_file, 'r') as f:
        # Get all dataset names
        def get_datasets(group, prefix=''):
            datasets = []
            for key in group.keys():
                if isinstance(group[key], h5py.Dataset):
                    datasets.append(prefix + key)
                elif isinstance(group[key], h5py.Group):
                    datasets.extend(get_datasets(group[key], prefix + key + '/'))
            return datasets

        dataset_names = sorted(list(set(f.keys()).difference({'Annotations'})))
        
        if dataset_names:

            # Timeline bar plot
            st.subheader("Timeline Bar Plot")
            fig, ax = plt.subplots(figsize=(10, 4))
            hf.makeTimePlotH5(ax, f)
            st.pyplot(fig)

            # Dropdown to select dataset
            selected_dataset = st.selectbox("Select a dataset", dataset_names)
            
            # Load selected dataset
            t = f[selected_dataset]['time'][()]
            data = f[selected_dataset]['data'][()]            
            #st.write(f"Dataset shape: {data.shape}")
            #st.write(f"Dataset dtype: {data.dtype}")

            # Create checkboxes for each column if data is 2D
            cols = f[selected_dataset]['data'].attrs.get('column_descriptions',None)
            if data.ndim > 1:
                st.write("Select channels to display:")
                selected_channels = {}
                for i in range(data.shape[1]):
                    selected_channels[i] = st.checkbox(cols[i], value=False)
            else:
                selected_channels = {0: True}
            
            # Flatten data if multi-dimensional
            #if data.ndim > 1:
            #    data = data.flatten()
            
            # Extracting the data
            df = pd.DataFrame(data[:,list(selected_channels.values())],
                              columns=[cols[i] for i in selected_channels.keys() if selected_channels[i]])

            # Plotly interactive plot
            st.subheader("Interactive Plot (Plotly)")
            fig_plotly = go.Figure()
            hf.plotData(fig_plotly,t,df, downsample = int(np.ceil(len(data)/1e6))) # Downsampling for performance
            st.plotly_chart(fig_plotly, width="stretch")
        else:
            st.warning("No datasets found in the HDF5 file")
else:
    st.info("Please upload an HDF5 file to begin")