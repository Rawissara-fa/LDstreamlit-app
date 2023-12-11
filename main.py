import yaml
import streamlit as st
import pandas as pd

import requests
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader 

st.set_page_config(page_title="Home", page_icon="👋",layout="wide")
st.write("# H e LL O! 👋")


st.markdown("_________________________________________________")


# st.session_state.clear()    
st.session_state["count"] = 0
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
config['credentials'],
config['cookie']['name'],
config['cookie']['key'],
config['cookie']['expiry_days']
# config['preauthorized']
)
        
        
cols1, cols2 = st.columns([1 ,1])

with cols1:

    
    if st.session_state["count"] == 0:
        
        # st.session_state.clear() 
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Sign in")
        
        
        if login_btn:
            # send api to fft security system
            SECURITY_API = "https://fitelsmart-api.azurewebsites.net/api/security-center/SignIn"
            login_data_json = {
                "USER_NAME": username,
                "USER_PASSWORD": password
                }
    
            # if ((username == 'Admin') &(password == '12345')):
            #     st.session_state["authentication_status"] = 'True'  
            #     login_response = 'None'      
            #     # st.sidebar.write(f'User : **{str.upper(st.session_state["username"])}**') #name

            # else:
            login_response = requests.post(url = SECURITY_API, json = login_data_json).json()
            print(login_response)
            
            st.session_state.login_user = login_response["ResultOnDb"][0]
            st.session_state["authentication_status"] = login_response["Status"]
            st.sidebar.write("Welcome: "+st.session_state.login_user["TITLE_OF_COURTESY_NAME"] +st.session_state.login_user["FIRST_NAME"]+ \
                    "\n"+ "Department: " + st.session_state.login_user["SECTION_NAME"])
                
            
            if st.session_state["authentication_status"] is True:
                
                # st.sidebar.write("USER: "+st.session_state.login_user["TITLE_OF_COURTESY_NAME"] +st.session_state.login_user["FIRST_NAME"]+ \
                #         " in " + st.session_state.login_user["SECTION_NAME"])
                
                authenticator.logout('Logout', 'sidebar', key='unique_key')
                st.success('User Login successfully')
                st.session_state["count"] = 1
                
                # st.sidebar.title('Some content')
            elif st.session_state["authentication_status"] is False:
                st.error('Username/password is incorrect')
            elif st.session_state["authentication_status"] is None:
                st.warning('Please enter your username and password')  
            else:
                st.error(login_response["Message"])
    
    elif st.session_state["count"] == 1:
        
        if st.session_state["authentication_status"] is True:
            st.sidebar.write("Welcome: "+st.session_state.login_user["TITLE_OF_COURTESY_NAME"] +st.session_state.login_user["FIRST_NAME"]+ \
                            "\n"+ "Department: " + st.session_state.login_user["SECTION_NAME"])
            authenticator.logout('Logout', 'sidebar', key='unique_key')
            
    else:  
        st.session_state['logout'] = True
        st.session_state["count"] = 0
        st.session_state.clear() 
