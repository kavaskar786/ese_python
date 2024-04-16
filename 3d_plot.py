import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("WomensClothingE-CommerceReviews.csv")  # Replace "your_dataset.csv" with the path to your CSV file
    return df

df = load_data()

# Sidebar for filtering data
st.sidebar.header("Filter Data")
division_names = df['Division Name'].unique()
selected_division = st.sidebar.selectbox("Select Division", division_names)

# Filter data based on selected division
filtered_df = df[df['Division Name'] == selected_division]

# Display the filtered dataset
st.subheader("Filtered Dataset")
st.write(filtered_df)

# Visualization: 3D plot
st.subheader("3D Plot: Age vs Rating vs Positive Feedback Count")
fig = px.scatter_3d(filtered_df, x='Age', y='Rating', z='Positive Feedback Count', color='Rating')
st.plotly_chart(fig, use_container_width=True)
