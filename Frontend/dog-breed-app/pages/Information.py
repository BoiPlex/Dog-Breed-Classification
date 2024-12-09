import streamlit as st

st.set_page_config(page_title="Information")

st.header("Project Information")
st.subheader("Overview")

st.markdown("""
This project focuses on **dog breed classification** using the **VGG16 model**, aiming to accurately identify dog breeds from images. We fine-tuned a VGG16 model and trained it with the [**Stanford Dog Breed dataset**](http://vision.stanford.edu/aditya86/ImageNetDogs/) to make it suitable for classifying specific breeds.

Additionally, we compared the performance of three optimizers: **Adam**, **AdamW**, and **SGD**, on this large-scale classification task. By analyzing the results, we gained insights into how different optimization strategies affect the **performance and efficiency** of the model in actual image classification tasks.
""")


st.image("Frontend/dog-breed-app/src/adamw_metrics.png", use_column_width=True)
st.image("Frontend/dog-breed-app/src/adam_metrics.png", use_column_width=True)
st.image("Frontend/dog-breed-app/src/sgd_metrics.png", use_column_width=True)
st.image("Frontend/dog-breed-app/src/convergence_comparison.png", use_column_width=True)
st.header("Contribution")
st.image("Frontend/dog-breed-app/src/Temp_Contribution.png", use_column_width=True)
