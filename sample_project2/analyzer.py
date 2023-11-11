import streamlit as st
import pandas as pd
from recommendation_model import RecommendationModel
import pandas as pd
import pygwalker as pyg
from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html
import streamlit.components.v1 as components
import sketch
from io import BytesIO



st.set_page_config(
    page_title="Recommendation System App",
    layout="wide"
)

# Load the pre-trained model
model = RecommendationModel.load_model("bundle_recommendation_model.pkl")

# Streamlit app
st.title("Recommendation System App")

# Initialize pygwalker communication
init_streamlit_comm()

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Get bundle name from user input
bundle_name = st.text_input("Enter bundle name:")

# Section break
st.write("---")

# Button to trigger recommendation
if st.button("Recommend"):
    if uploaded_file is not None:
        # Load data and make recommendations
        model.load_data(uploaded_file)
        recommendations = model.get_users_to_activate_bundles(bundle_name, N=3)

        # Display the recommendations
        st.write("Recommendations:")
        st.write(recommendations)
    else:
        st.warning("Please upload a CSV file.")

# Section break
st.write("---")

# Button to trigger dataset analysis
if st.button("Analyze Dataset"):
    if uploaded_file is not None:
        # Load data
        #data = pd.read_csv(uploaded_file)
        # When using `use_kernel_calc=True`, you should cache your pygwalker html, if you don't want your memory to explode
        @st.cache_resource
        def get_pyg_html(df: pd.DataFrame) -> str:
            # When you need to publish your application, you need set `debug=False`,prevent other users to write your config file.
            # If you want to use feature of saving chart config, set `debug=True`
            html = get_streamlit_html(df, spec="./gw0.json", use_kernel_calc=True, debug=False)
            return html
        
        @st.cache_data
        def get_df() -> pd.DataFrame:
            return pd.read_csv(uploaded_file)
        
        data = get_df()
        
        components.html(get_pyg_html(data), width=1300, height=1024, scrolling=True)
        #Display the dataset using st.dataframe()

        #pyg.walk(data,dark='light',env='Streamlit')

    else:
        st.warning("Please upload a CSV file for dataset analysis.")


# Section break
st.write("---")
  
if st.button("Talk to file"):
    if uploaded_file is not None:
        # # Read the contents of the file
        # file_contents = uploaded_file.read()

        # # Use BytesIO to create a file-like object from the file contents
        # file_buffer = BytesIO(file_contents)

        # # Load the CSV file into a pandas DataFrame
        # df = pd.read_csv(file_buffer)

        # Display loaded data
        #st.dataframe(df)
        df = pd.DataFrame(uploaded_file)
        # getting a question
        quest = st.text_input("Ask a question from Data set!")

        # Load data and make recommendations
        
        df.sketch.ask(quest)

        # Display the recommendations
        st.write("Answer:")
        st.write(df)
    else:
        st.warning("Please upload a CSV file for dataset analysis.")


st.write('Predicting customers who are ending their subscription in the specified time interval')
time_interval = st.number_input("input the max interval number  starting today.")

# Dictionary to map bundle types to their respective values
bundle_type_mapping = {"Monthly": 30 , "bi-Weekly": 14 , "Weekly": 7 , "bi-Monthly": 60 }

# Radio button for selecting bundle type
# getting the valid value of bundle type
day = bundle_type_mapping[st.radio("Select Bundle Type", list(bundle_type_mapping.keys()))]

if st.button("Predicting Subscription validity"):
    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)

        #changing the activation data type
        data["ACTIVATION_DATE"] = pd.to_datetime(data["ACTIVATION_DATE"])
        
        #st.write(data["ACTIVATION_DATE"])
        #st.write(data["ACTIVATION_DATE"])

        # Filter customers ending subscription based on the specified time interval
        today = pd.to_datetime('today')
        end_date = pd.to_datetime('today') + pd.DateOffset(days=time_interval)
        data["DEACTIVATION_DATE"] = pd.to_datetime(data["ACTIVATION_DATE"]) + pd.DateOffset(days=day)
        st.write("Today:", today)
        st.write("End Date:", end_date)
        #st.write("Deat date:", deact_date)
        #st.write("cond greater",deact_date < today)
        #st.write("same bundles ; " , data[data['BUNDLE_NAME'] == bundle_name])

        # Assuming 'ACTIVATION_DATE' is the column containing the activation date
        ending_customers = data[
            (data['BUNDLE_NAME'] == bundle_name) & 
            (pd.to_datetime(data["ACTIVATION_DATE"]) + pd.DateOffset(days=30) > today) & 
            (pd.to_datetime(data["ACTIVATION_DATE"]) < end_date)
        ]

        # Display the customers
        st.write(f"Customers Ending Subscription in the Next {time_interval} Days:")
        st.write(ending_customers[["SUBSCRIPTION_ID","BUNDLE_NAME","ACTIVATION_DATE","DEACTIVATION_DATE"]]) 
    else:
        st.warning("Please upload a CSV file.")