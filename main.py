import yaml
import streamlit as st
import pandas as pd

import streamlit_authenticator as stauth
from yaml.loader import SafeLoader


st.set_page_config(page_title="Home", page_icon="👋",layout="wide")
st.write("# H e LL O! 👋")


st.markdown("_________________________________________________")





with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
config['credentials'],
config['cookie']['name'],
config['cookie']['key'],
config['cookie']['expiry_days'],
config['preauthorized']
)

cols1, cols2 = st.columns([1 ,1])

with cols1:
    name, authentication_status, username = authenticator.login('Login', 'main')
    # st.write(stauth.Hasher(['12345', '54321']).generate())

    if st.session_state["authentication_status"]:
        # authenticator.logout('Logout', 'main', key='unique_key')
        st.sidebar.write(f'User : **{str.upper(st.session_state["username"])}**') #name
        authenticator.logout('Logout', 'sidebar', key='unique_key')
        st.success('User Login successfully')
        
        # st.sidebar.title('Some content')
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')


st.markdown("_________________________________________________")
# st.markdown(":black[ -> **more option**]")

                    
regis_page_bth = st.toggle("Reset password OR Register new user")
if regis_page_bth:
    
    cols1, cols2 = st.columns([0.2,1])
    with cols1:
        index = st.radio(" ", ["Reset password", "Register new user"])

    with cols2:
        
        if index == "Reset password" :
            
            if st.session_state["authentication_status"]:

                st.text('** Please enter your new password **')

                cols1, cols2, cols3 = st.columns([1 ,0.1 ,1])
                with cols1:
                    try:
                        if authenticator.reset_password(st.session_state["username"], 'Reset password'):
                            st.success('Password modified successfully')
                    except Exception as e:
                        st.error(e)
                    
                    with open('config.yaml', 'w') as file:
                        yaml.dump(config, file, default_flow_style=False)
                    
            else:
                cols3, cols4 = st.columns([1.5 ,2])
                with cols3:
                    st.text("""
                            ** If fortgot password for log in, Please enter your username **
                            """)
                    try:
                        status_reset = None
                        username_of_forgotten_password, email_forgot_password, random_password = authenticator.forgot_password('Forgot password')
                        if username_of_forgotten_password:
                            status_reset = True
                            # st.success('Next to reset password')
                            st.success('New password: '+ random_password)
                            # Random password should be transferred to user securely
                            
                            with open('config.yaml', 'w') as file:
                                yaml.dump(config, file, default_flow_style=False)
                                
                        else:
                            # st.error('Username not found')
                            st.info('Could you fill username (again)')
                    except Exception as e:
                        st.error(e)
                        

        elif index == "Register new user" :
            # fogot_bth == False, reset_bth == False
            st.text('** Please fill data into box **')
            cols1, cols2, cols3 = st.columns([1 ,0.1 ,1])
            with cols1:
                try:
                    if authenticator.register_user('Register user', preauthorization=False):
                        st.success('User registered successfully')
                except Exception as e:
                    st.error(e)
                
                with open('config.yaml', 'w') as file:
                        yaml.dump(config, file, default_flow_style=False)
            
            with cols3:
                    st.text("... some detail in config file...")
                    st.text ("""\
                       (userID):
                          email: (name-mailoutlook)@furukawaelectric.com
                          name: Name Surename
                          password: # Your password #
                        
                        """)
      
                    # st.text("Username: "+ str.upper(st.session_state["username"]))
                    # fillusername = str.lower(st.session_state["username"])
                    # st.write(config['credentials']['usernames'][fillusername])








# -----------------------------------------
  ## OPen streamlit app
#------------------------------------------

# C:/Users/rawissara.bua/AppData/Local/anaconda3/Scripts/activate
# PS conda activate fastapiwork
# PS streamlit run main.py
# PS(0.159) python -m streamlit run main.py
# http://localhost:8501 // http://192.168.0.16:8501