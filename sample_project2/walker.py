# app.py
import streamlit as st
import pandas as pd

st.title('Streamlit App with Sketch')

# Create a sample Pandas DataFrame
df = pd.read_csv("./data_bundle.csv")

# Display DataFrame
st.dataframe(df)

# Use Sketch functionalities
st.write("Ask Sketch Question:")
question = st.text_input("Enter your question:")
st.write(df.sketch.ask(question))
