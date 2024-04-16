import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache
def load_data():
    return pd.read_csv("WomensClothingE-CommerceReviews.csv")  # Replace "your_dataset.csv" with the path to your CSV file

def text_similarity_analysis(df):
    filtered_df = df[df['Division Name'].isin(['General Petite', 'Intimates'])]
    tokenized_texts = filtered_df['Review Text'].apply(lambda x: ' '.join(x))
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_texts)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    similar_reviews = {}
    for i in range(len(cosine_sim)):
        similar_reviews[i] = [j for j, score in enumerate(cosine_sim[i]) if score > 0.7 and j != i]
    return similar_reviews

def process_image(image, techniques):
    if 'Resize' in techniques:
        width, height = st.slider("Select resize dimensions", 1, 1000, (image.width, image.height))
        image = image.resize((width, height))
    if 'Grayscale Conversion' in techniques:
        image = image.convert('L')
    if 'Image Cropping' in techniques:
        left, top, right, bottom = st.slider("Select crop region", 0, image.width, (0, image.width))
        image = image.crop((left, top, right, bottom))
    if 'Image Rotation' in techniques:
        angle = st.slider("Select rotation angle", -180, 180, 0)
        image = image.rotate(angle)
    return image


def main():
    st.title("Interactive Web Application")

    df = load_data()

    selected_tab = st.sidebar.radio("Select Tab", ("3D Plot Visualization", "Image Processing", "Text Similarity Analysis"))

    if selected_tab == "3D Plot Visualization":
        st.header("3D Plot Visualization")
        division_names = df['Division Name'].unique()
        selected_division = st.selectbox("Select Division", division_names)
        filtered_df = df[df['Division Name'] == selected_division]
        fig = px.scatter_3d(filtered_df, x='Age', y='Rating', z='Positive Feedback Count', color='Rating')
        st.plotly_chart(fig, use_container_width=True)

    elif selected_tab == "Image Processing":
        st.header("Image Processing")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            original_image = Image.open(uploaded_image)
            st.subheader("Original Image")
            st.image(original_image, caption="Original Image", use_column_width=True)
            selected_techniques = st.multiselect("Select image processing techniques",
                                                 ['Resize', 'Grayscale Conversion', 'Image Cropping', 'Image Rotation'])
            if st.button("Process Image"):
                processed_image = process_image(original_image, selected_techniques)
                st.subheader("Processed Image")
           
    elif selected_tab == "Text Similarity Analysis":
        st.header("Text Similarity Analysis")
        similar_reviews = text_similarity_analysis(df)
        for i, similar_list in similar_reviews.items():
            if similar_list:
                st.write(f"Review {i}: Similar to {similar_list}")

if __name__ == "__main__":
    main()
