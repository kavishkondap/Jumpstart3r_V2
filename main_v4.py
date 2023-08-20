import streamlit as st
from helper_v4 import data_processing, successBreakdown, load_model, add_bg_from_local, title_section, numbers_section, about_section, mission_section, ml_section, data_section, footer_section
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import base64

st.set_page_config(page_title="Jumpstart3r - Kickstarter Success Predictor", layout="wide", page_icon = 'logo5.webp')
hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.markdown("""
<style>div[data-testid="stToolbar"] { display: none;}</style>
""", unsafe_allow_html=True)

add_bg_from_local('background.png')

font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 24px;
}
</style>
"""

st.write(font_css, unsafe_allow_html=True)
hide_full_screen = '''
<style>
.element-container:nth-child(3) .overlayBtn {visibility: hidden;}
.element-container:nth-child(12) .overlayBtn {visibility: hidden;}
</style>
'''

st.markdown(hide_full_screen, unsafe_allow_html=True) 
title_section ()
numbers_section ()
ml_section()
about_section()
mission_section()
# data_section()
footer_section()
