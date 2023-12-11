import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.error import URLError
import components.algorithm as algorithm

import cv2
import math
import datetime
import numpy as np
from datetime import datetime
from PIL import Image

import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth




st.set_page_config(page_title="ExploreFile", page_icon=" ", layout="wide")
st.title("Explore File")
st.markdown("_________________________________________________")


# Check authentication when user lands on the page.
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)
    
authenticator = stauth.Authenticate(
config['credentials'],
config['cookie']['name'],
config['cookie']['key'],
config['cookie']['expiry_days'],
config['preauthorized']
)

if st.session_state["authentication_status"]:
    st.sidebar.write(f'UserID : **{str.upper(st.session_state["username"])}**') #name
    authenticator.logout('Logout', 'sidebar')
    # st.title('Some content')
    # elif st.session_state["authentication_status"] == False:
    #     st.error('Username/password is incorrect')
    # elif st.session_state["authentication_status"] == None:
    #     st.warning('Please enter your username and password')
 
  
  
  

  
    # -----------------------------------------
    ## Record after judgment
    #------------------------------------------
                     
    df1 = pd.read_csv("./result_judgement.csv")
    current_today = datetime.now()
    current_file = str(current_today).split(" ")[0]
    
    def clear_text():
        st.session_state["text"] = ""
    
    ## Open CSV file
    cols001, cols002 = st.columns([1 ,4])
    with cols001:
        
        # reset_btn =  st.button("Rerun", type="primary")
        data_type = st.radio(
            ":gray[**Select type that you want to download**]", #👇
            ["All data", "Only today", "Fillter day"])

    with cols002:

        if data_type == "All data":
            
            ## load CSV All file 
            csv = pd.DataFrame(df1).to_csv(index=False)
            
            st.download_button(
            label="Download",
            data=csv,
            file_name="result_judgement_" + current_file + ".csv",
            mime='text/csv',
            type="primary"
            )
            st.write(df1)
            
        elif data_type == "Only today":
            
            ## load CSV All file
            filtered_df = df1[df1["updatedate"].isin([str(current_file)])] 
            csv = pd.DataFrame(filtered_df).to_csv(index=False)
            
            st.download_button(
            label="Download",
            data=csv,
            file_name="result_judgement_" + current_file + ".csv",
            mime='text/csv',
            type="primary"
            )
            st.write('Today is:', current_file)
            st.write(filtered_df)
        
        elif data_type == "Fillter day":

            date_filter = st.date_input("Date you want to choose", key = "text")
            st.write('Today is:', date_filter)
            
            filtered_df = df1[df1["updatedate"].isin([str(date_filter)])] 
            csv = pd.DataFrame(filtered_df).to_csv(index=False)
            
            st.download_button(
            label="Download",
            data=csv,
            file_name="result_judgement_" + current_file + ".csv",
            mime='text/csv',
            type="primary"
            )
            st.write(filtered_df)

                


    
    st.markdown("_________________________________________________")

 

elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')