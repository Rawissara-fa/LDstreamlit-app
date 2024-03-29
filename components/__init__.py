import yaml
import streamlit as st
from yaml.loader import SafeLoader
import streamlit.components.v1 as components

from .hasher import Hasher
from .authenticate import Authenticate


### SAVE File yaml
_RELEASE = True

if not _RELEASE:
    # hashed_passwords = Hasher(['abc', 'def']).generate()

    # Loading config file
    with open('../config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    # Creating the authenticator object
    authenticator = Authenticate(
        config['credentials'],
        config['cookie']['name'], 
        config['cookie']['key'], 
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    # creating a login widget
    authenticator.login('Login', 'main')
    if st.session_state["authentication_status"]:
        authenticator.logout('Logout', 'main')
        st.write(f'Welcome *{st.session_state["name"]}*')
        st.title('Some content')
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')

    # Creating a password reset widget
    if st.session_state["authentication_status"]:
        try:
            if authenticator.reset_password(st.session_state["username"], 'Reset password'):
                # st.success('Password modified successfully')
                st.success('Next to reset password')
        except Exception as e:
            st.error(e)
    elif st.session_state["authentication_status"]:
        try:
            if authenticator.reset_unknown(st.session_state["username"], 'Reset password'):
                # st.success('Password modified successfully')
                st.success('Next to reset password')
        except Exception as e:
            st.error(e)
            

    # Creating a new user registration widget
    try:
        if authenticator.register_user('Register user', preauthorization=False):
            st.success('User registered successfully')
    except Exception as e:
        st.error(e)

    # Creating an update user details widget
    if st.session_state["authentication_status"]:
        try:
            if authenticator.update_user_details(st.session_state["username"], 'Update user details'):
                st.success('Entries updated successfully')
        except Exception as e:
            st.error(e)

    # Saving config file
    with open('../config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
        