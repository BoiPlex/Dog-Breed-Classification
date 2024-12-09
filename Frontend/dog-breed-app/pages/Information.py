import streamlit as st

st.set_page_config(page_title="Information")

st.header("Project Information")
st.markdown("""
## Overview
""")

st.markdown("""
This project focuses on **dog breed classification** using the **VGG16 model**, aiming to accurately identify dog breeds from images. We fine-tuned a VGG16 model and trained it with the [**Stanford Dog Breed dataset**](http://vision.stanford.edu/aditya86/ImageNetDogs/) to make it suitable for classifying specific breeds.

Additionally, we compared the performance of three optimizers: **Adam**, **AdamW**, and **SGD**, on this large-scale classification task. By analyzing the results, we gained insights into how different optimization strategies affect the **performance and efficiency** of the model in actual image classification tasks.
""")

st.markdown("""
## AI methodologies and techniques
#### Model:

  

Use VGG16 as the base model, preload the weights to ImageNet, and remove the original classification layer. We make modifications to the original model.

First 15 layers freeze: Fix the weights of early feature extraction layers to avoid overfitting.

Layer 15 after unfreezing: allows deep weight updates to improve the model's adaptability to the current data set.

  

#### Train strategy:

  

Data Augmentation:

  

Use ImageDataGenerator to enhance training data online. Methods include:

  

```python  
  
datagen = ImageDataGenerator(

rotation_range=20,

width_shift_range=0.2,

height_shift_range=0.2,

zoom_range=0.2,

horizontal_flip=True,

fill_mode='nearest'

)

  
```

  
  

### Learning Rate Scheduler:

  

ReduceLROnPlateau:

  

When the validation loss shows no improvement in 10 epochs, cut the learning rate in half. The minimum learning rate is set to 1e-6.

  

LearningRateLogger:

Record and print the learning rate changes for each epoch.

  

#### Early Stopping:

  

Monitor validation loss (val_loss).

If the validation loss does not improve after 30 epochs, terminate training and restore the optimal weights.

  

#### Training model:

  

Batch size: 128

Maximum number of Epochs: 500

We use more than expected Epochs as set up because we purchased colab pro+ and there still is a limit for compute usage.
""")

st.markdown("""
## Challenges faced and their resolutions.

Challenge: Limited dataset size

Problem: The data set used to train the dog breed classifier was relatively small, increasing the risk of overfitting and reducing the model's ability to generalize.

Solution: Apply data augmentation to expand the data set. Use techniques such as rotation, flipping, scaling, and moving to create a diverse set of training images to improve model robustness.

---

Challenge: Overfitting

Problem: Due to the complexity of the VGG16 architecture and the small dataset, the model tends to overfit the training data.

Solution: Implement early stopping to monitor validation loss and stop training when improvements stop. Additionally, a Dropout layer is included in the model to introduce regularization.

---

Challenge: Computational Constraints

Problem: Training a large model like VGG16 using augmentation and multiple experiments is computationally intensive.

Solution: We purchased Colab Pro+ for the A100 GPU. The model is trained on an A100 GPU to accelerate operations. Additionally, a number of sizes and learning rates are optimized to ensure efficient use of available resources.

""")

st.markdown("""
## Result Figures
""")
st.image("Frontend/dog-breed-app/src/adamw_metrics.png", use_column_width=True)
st.image("Frontend/dog-breed-app/src/adam_metrics.png", use_column_width=True)
st.image("Frontend/dog-breed-app/src/sgd_metrics.png", use_column_width=True)
st.image("Frontend/dog-breed-app/src/convergence_comparison.png", use_column_width=True)
st.header("Contribution")
st.image("Frontend/dog-breed-app/src/Temp_Contribution.png", use_column_width=True)
