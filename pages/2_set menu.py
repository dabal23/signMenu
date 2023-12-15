import streamlit as st
import pandas as pd
import pickle

# Load data from pickle file
data = pickle.load(open('data_df.pickle', 'rb'))

# Initialize session state
if 'edit_button_clicked' not in st.session_state:
    st.session_state.edit_button_clicked = False

if 'menu_name' not in st.session_state:
    st.session_state.menu_name = ''

if 'menu_desc' not in st.session_state:
    st.session_state.menu_desc = ''

# Load or create the DataFrame for displaying menu
if 'menu_df' not in st.session_state:
    st.session_state.menu_df = data.copy()

# Create a new DataFrame to store edited values
if 'edited_menu_df' not in st.session_state:
    st.session_state.edited_menu_df = data.copy()

st.title('Menu description')
st.write(st.session_state.menu_df)

# Edit button
edit_button = st.button("Edit")

# Check if the "Edit" button is clicked
if edit_button:
    st.session_state.edit_button_clicked = True

# Display the text inputs if the Edit button is clicked
if st.session_state.edit_button_clicked:
    opt = st.selectbox('Choose sign:', data['label'])

    col1, col2 = st.columns(2)

    # Add text input to the first column
    with col1:
        st.header("Menu name")
        st.session_state.menu_name = st.text_input(
            "Enter text for Menu name", value=st.session_state.menu_name)

    # Add text input to the second column
    with col2:
        st.header("Menu description")
        st.session_state.menu_desc = st.text_input(
            "Enter text for Menu description", value=st.session_state.menu_desc)

    # OK button
    ok = st.button('OK')
    if ok:
        # Update the original DataFrame with entered values
        data.loc[data['label'] == opt,
                 'Menu Name'] = st.session_state.menu_name
        data.loc[data['label'] == opt,
                 'Description'] = st.session_state.menu_desc
        data.sort_values(by='label', inplace=True)
        pickle.dump(data, open('data_df.pickle', 'wb'))
        st.success('Edited data exported to data_df.pickle')

# Display the updated DataFrame
st.write(data)

# # Export the edited DataFrame to the original pickle file
# if st.button('Export Edited Data'):
#     pickle.dump(data, open('data_df.pickle', 'wb'))
#     st.success('Edited data exported to data_df.pickle')
