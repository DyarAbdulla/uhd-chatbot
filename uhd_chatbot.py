import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import base64

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="UHD AI Chatbot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
    /* Main container styling - Campus Background Theme with Enhanced Overlay */
    .main {
        padding-top: 1rem;
        background: radial-gradient(circle at 50% 20%, rgba(0,0,0,0.4) 0%, rgba(0,0,0,0.7) 70%), 
                    linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.6)), 
                    url('https://images.unsplash.com/photo-1562774053-701939374585?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    /* Override Streamlit's default background */
    .stApp {
        background: radial-gradient(circle at 50% 20%, rgba(0,0,0,0.4) 0%, rgba(0,0,0,0.7) 70%), 
                    linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.6)), 
                    url('https://images.unsplash.com/photo-1562774053-701939374585?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Force all Streamlit elements to have proper styling */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 20%, rgba(0,0,0,0.4) 0%, rgba(0,0,0,0.7) 70%), 
                    linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.6)), 
                    url('https://images.unsplash.com/photo-1562774053-701939374585?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    [data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Header styling - Campus Building Colors */
    .header-container {
        background: linear-gradient(135deg, rgba(245, 222, 179, 0.9) 0%, rgba(160, 82, 45, 0.8) 50%, rgba(135, 206, 235, 0.8) 100%);
        backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 25px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.3);
        border: 2px solid rgba(245, 222, 179, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
        z-index: -1;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    
    .university-name {
        color: white;
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: 0.5px;
    }
    
    .subtitle {
        color: #f0f0f0;
        font-size: 1.3rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Tab styling - Dark Glassmorphism */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 18px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(245, 222, 179, 0.3);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 16px 30px;
        font-weight: 600;
        color: #2c3e50;
        border: 1px solid rgba(245, 222, 179, 0.4);
        transition: all 0.4s ease;
        font-size: 16px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(245, 222, 179, 0.5);
        border-color: #f5deb3;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(245, 222, 179, 0.4);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f5deb3 0%, #a0522d 50%, #87ceeb 100%);
        color: #2c3e50 !important;
        border-color: transparent;
        box-shadow: 0 10px 30px rgba(245, 222, 179, 0.5), 0 0 20px rgba(135, 206, 235, 0.3);
        transform: translateY(-2px);
        position: relative;
    }
    
    .stTabs [aria-selected="true"]::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #f5deb3, #a0522d, #87ceeb);
        border-radius: 0 0 15px 15px;
    }
    
    /* Content cards - Campus Background */
    .content-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.5rem;
        border: 2px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .content-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(245, 222, 179, 0.2) 0%, rgba(135, 206, 235, 0.2) 100%);
        border-radius: 20px;
        z-index: -1;
    }
    
    .content-card h3 {
        color: #ffffff !important;
        margin-bottom: 1rem;
        font-size: 1.5rem;
        font-weight: 700;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.8), 0 0 20px rgba(0, 0, 0, 0.5);
    }
    
    .content-card p, .content-card div {
        color: #ffffff !important;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.7);
        font-weight: 500;
    }
    
    /* Button styling - Campus Building Colors */
    .stButton button {
        background: linear-gradient(135deg, #f5deb3 0%, #a0522d 50%, #87ceeb 100%);
        color: #2c3e50;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(245, 222, 179, 0.3);
        font-size: 14px;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(245, 222, 179, 0.4);
        background: linear-gradient(135deg, #f0e68c 0%, #cd853f 50%, #b0e0e6 100%);
    }
    
    /* Input styling - Campus Background */
    .stTextInput input, .stSelectbox select {
        border-radius: 15px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        padding: 1rem 1.2rem;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.12) !important;
        backdrop-filter: blur(15px);
        color: #ffffff !important;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stTextInput input::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus {
        border-color: #f5deb3;
        box-shadow: 0 0 0 4px rgba(245, 222, 179, 0.4), 0 8px 25px rgba(0, 0, 0, 0.3);
        background: rgba(255, 255, 255, 0.15) !important;
        outline: none;
        transform: translateY(-1px);
    }
    
    .stTextInput label, .stSelectbox label {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 8px;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.8);
    }
    
    /* Fix selectbox dropdown - Dark Chatbot */
    .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px);
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Fix selectbox options - Dark Chatbot */
    [role="option"] {
        background: rgba(30, 30, 60, 0.95) !important;
        color: #ffffff !important;
    }
    
    [role="option"]:hover {
        background: rgba(245, 222, 179, 0.3) !important;
    }
    
    /* Success/Warning/Info boxes - Dark Chatbot */
    .stSuccess {
        background: rgba(40, 167, 69, 0.15) !important;
        backdrop-filter: blur(10px);
        border-left: 4px solid #28a745;
        border-radius: 12px;
        padding: 1rem;
        color: #90ee90 !important;
        border: 1px solid rgba(40, 167, 69, 0.3);
    }
    
    .stWarning {
        background: rgba(255, 193, 7, 0.15) !important;
        backdrop-filter: blur(10px);
        border-left: 4px solid #ffc107;
        border-radius: 12px;
        padding: 1rem;
        color: #ffd54f !important;
        border: 1px solid rgba(255, 193, 7, 0.3);
    }
    
    .stInfo {
        background: rgba(23, 162, 184, 0.15) !important;
        backdrop-filter: blur(10px);
        border-left: 4px solid #17a2b8;
        border-radius: 12px;
        padding: 1rem;
        color: #80deea !important;
        border: 1px solid rgba(23, 162, 184, 0.3);
    }
    
    /* Logo container */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .logo-img {
        border-radius: 15px;
        border: 4px solid white;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Class card styling - Dark Premium */
    .class-card {
        background: linear-gradient(135deg, rgba(245, 222, 179, 0.2) 0%, rgba(160, 82, 45, 0.2) 100%);
        backdrop-filter: blur(15px);
        border-left: 5px solid #f5deb3;
        border-radius: 20px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        transition: all 0.4s ease;
        border: 1px solid rgba(245, 222, 179, 0.3);
    }
    
    .class-card:hover {
        transform: translateX(10px) scale(1.02);
        box-shadow: 0 12px 40px rgba(245, 222, 179, 0.4);
        border-left-width: 8px;
        background: linear-gradient(135deg, rgba(245, 222, 179, 0.3) 0%, rgba(160, 82, 45, 0.3) 100%);
    }
    
    /* Status badge - Enhanced Clarity */
    .status-badge {
        background: rgba(245, 222, 179, 0.6);
        backdrop-filter: blur(15px);
        padding: 0.8rem 1.8rem;
        border-radius: 35px;
        display: inline-block;
        box-shadow: 0 8px 25px rgba(245, 222, 179, 0.4);
        font-size: 1rem;
        font-weight: 800;
        color: #1a1a1a;
        border: 2px solid rgba(245, 222, 179, 0.7);
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    
    .status-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(245, 222, 179, 0.6);
        background: rgba(245, 222, 179, 0.7);
    }
    
    /* Footer - Enhanced Clarity */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #ffffff;
        font-size: 0.875rem;
        font-weight: 500;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.7);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 2rem;
        background: rgba(255, 255, 255, 0.02);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Section headers - Enhanced Clarity */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
        text-shadow: 0 3px 12px rgba(0, 0, 0, 0.8), 0 0 25px rgba(0, 0, 0, 0.6);
    }
    
    /* Regular text - Enhanced Clarity */
    p, div, span, label, li {
        color: #ffffff !important;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.7);
        font-weight: 500;
    }
    
    /* Markdown text - Enhanced Clarity */
    .markdown-text-container {
        color: #ffffff !important;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.7);
        font-weight: 500;
    }
    
    /* Streamlit elements - Enhanced Clarity */
    .stMarkdown, .element-container {
        color: #ffffff !important;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.7);
        font-weight: 500;
    }
    
    /* Text areas - Dark Chatbot */
    .stTextArea textarea {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Expander styling - Dark Chatbot */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        font-weight: 600;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-top: none;
        color: #e2e8f0 !important;
    }
    
    /* Dataframe styling - Dark Chatbot */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stDataFrame table {
        color: #e2e8f0 !important;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4);
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(40, 167, 69, 0.6);
    }
    
    /* Remove all horizontal lines */
    hr {
        display: none !important;
    }
    
    /* Hide GitHub and Fork buttons */
    [data-testid="stHeader"] .stAppHeader {
        display: none !important;
    }
    
    /* Hide Streamlit's default header elements */
    .stApp > header {
        display: none !important;
    }
    
    /* Hide any GitHub/Fork related elements */
    [data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Fix developer name spacing */
    .developer-card h4 {
        margin: 0.5rem 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
    }
    
    /* Enhanced Clock Widget */
    .clock-widget {
        background: linear-gradient(135deg, #f5deb3 0%, #a0522d 100%) !important;
        text-align: center;
        min-width: 120px;
        color: white !important;
    }
    
    .clock-widget small {
        font-size: 0.8rem;
        opacity: 0.9;
        display: block;
        margin-top: 2px;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Enhanced Status Badges */
    .status-badge {
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .status-badge::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .status-badge:hover::before {
        left: 100%;
    }
    
    .status-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(245, 222, 179, 0.5);
    }
    
    /* Enhanced Header with Animation */
    .header-container {
        position: relative;
        overflow: hidden;
    }
    
    .header-container::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        transform: translateX(-100%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab"] {
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stTabs [data-baseweb="tab"]:hover::before {
        left: 100%;
    }
    
    /* Enhanced Button Animations */
    .stButton button {
        position: relative;
        overflow: hidden;
    }
    
    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton button:hover::before {
        left: 100%;
    }
    
    /* Enhanced Class Cards */
    .class-card {
        position: relative;
        overflow: hidden;
    }
    
    .class-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.8s;
    }
    
    .class-card:hover::before {
        left: 100%;
    }
    
    /* Enhanced Input Focus Effects */
    .stTextInput input:focus, .stSelectbox select:focus {
        transform: scale(1.02);
        box-shadow: 0 0 0 3px rgba(245, 222, 179, 0.3), 0 8px 25px rgba(245, 222, 179, 0.2);
    }
    
     /* Enhanced Content Cards with Unified Glass Style */
     .content-card {
         position: relative;
         overflow: hidden;
         background: rgba(255, 255, 255, 0.1) !important;
         backdrop-filter: blur(20px) !important;
         border: 1px solid rgba(255, 255, 255, 0.2) !important;
         border-radius: 20px !important;
         box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
         padding: 2rem !important;
     }
     
     /* Unified Glass Style for All Cards */
     .glass-card {
         background: rgba(255, 255, 255, 0.1) !important;
         backdrop-filter: blur(20px) !important;
         border: 1px solid rgba(255, 255, 255, 0.2) !important;
         border-radius: 20px !important;
         box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
         padding: 1.5rem !important;
     }
     
     /* Answer Box Glass Style */
     .answer-box {
         background: rgba(0, 0, 0, 0.4) !important;
         backdrop-filter: blur(20px) !important;
         border: 1px solid rgba(255, 255, 255, 0.2) !important;
         border-radius: 16px !important;
         padding: 1.5rem !important;
         color: white !important;
         line-height: 1.6 !important;
     }
     
     /* Professional Tab Styling */
     .stTabs [data-baseweb="tab-list"] {
         gap: 0.5rem !important;
         background: rgba(255, 255, 255, 0.1) !important;
         backdrop-filter: blur(20px) !important;
         border: 1px solid rgba(255, 255, 255, 0.2) !important;
         border-radius: 15px !important;
         padding: 0.5rem !important;
         margin: 1rem 0 !important;
     }
     
     .stTabs [data-baseweb="tab"] {
         background: rgba(255, 255, 255, 0.1) !important;
         backdrop-filter: blur(15px) !important;
         border: 1px solid rgba(255, 255, 255, 0.2) !important;
         border-radius: 12px !important;
         color: #ffffff !important;
         font-weight: 600 !important;
         padding: 0.75rem 1.5rem !important;
         margin: 0 !important;
         transition: all 0.3s ease !important;
         position: relative !important;
     }
     
     .stTabs [data-baseweb="tab"]:hover {
         background: rgba(255, 255, 255, 0.15) !important;
         backdrop-filter: blur(20px) !important;
         border: 1px solid rgba(255, 255, 255, 0.3) !important;
         transform: translateY(-1px) !important;
         box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
     }
     
     .stTabs [aria-selected="true"] {
         background: rgba(255, 255, 255, 0.2) !important;
         backdrop-filter: blur(25px) !important;
         border: 1px solid rgba(255, 255, 255, 0.4) !important;
         box-shadow: 0 0 20px rgba(255, 255, 255, 0.1) !important;
     }
     
     .stTabs [aria-selected="true"]::after {
         content: '' !important;
         position: absolute !important;
         bottom: -2px !important;
         left: 50% !important;
         transform: translateX(-50%) !important;
         width: 60% !important;
         height: 3px !important;
         background: linear-gradient(90deg, #ff6b6b, #ff8e8e) !important;
         border-radius: 2px !important;
         box-shadow: 0 0 10px rgba(255, 107, 107, 0.5) !important;
     }
    
    .content-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #f5deb3, #a0522d, #f5deb3);
        background-size: 200% 100%;
        animation: gradient-shift 3s ease infinite;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Enhanced Developer Cards */
    .developer-card {
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
    }
    
    .developer-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.6s;
    }
    
    .developer-card:hover::before {
        left: 100%;
    }
    
    .developer-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* Enhanced Footer */
    .footer {
        background: linear-gradient(135deg, rgba(245, 222, 179, 0.15) 0%, rgba(160, 82, 45, 0.15) 100%);
        backdrop-filter: blur(10px);
        border-top: 2px solid rgba(245, 222, 179, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #f5deb3, transparent);
        animation: footer-shine 4s infinite;
    }
    
    @keyframes footer-shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }

</style>
""", unsafe_allow_html=True)

# ================== LOGO HANDLING ==================
SCRIPT_DIR = Path(__file__).resolve().parent
LOGO_CANDIDATES = [
    SCRIPT_DIR / "uhd_logo.png",
    SCRIPT_DIR / "uhd_logo.jpg",
    SCRIPT_DIR / "logo.png",
    SCRIPT_DIR / "logo.jpg"
]


def get_logo():
    """Find and return the logo file path"""
    for logo_path in LOGO_CANDIDATES:
        if logo_path.exists():
            return logo_path
    return None


def get_base64_image(image_path):
    """Convert image to base64 for embedding"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None


# ================== HEADER ==================
LOGO_FILE = get_logo()

# Create header with logo
if LOGO_FILE:
    logo_b64 = get_base64_image(LOGO_FILE)
    if logo_b64:
        st.markdown(f"""
        <div class="header-container">
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_b64}" class="logo-img" width="100">
                <div>
                    <h1 class="university-name">UNIVERSITY OF HUMAN DEVELOPMENT</h1>
                    <p class="subtitle">🎓 AI FAQ & Class Schedule Chatbot</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="header-container">
            <h1 class="university-name">🎓 UNIVERSITY OF HUMAN DEVELOPMENT</h1>
            <p class="subtitle">AI FAQ & Class Schedule Chatbot</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="header-container">
        <h1 class="university-name">🎓 UNIVERSITY OF HUMAN DEVELOPMENT</h1>
        <p class="subtitle">AI FAQ & Class Schedule Chatbot</p>
    </div>
    <div class="uhd-watermark">UHD</div>
    """, unsafe_allow_html=True)

# ================== FILE PATHS ==================
FAQ_PATH = SCRIPT_DIR / "faq.csv"
SCHED_PATH = SCRIPT_DIR / "schedule.csv"

# ================== DEFAULT DATA ==================
DEFAULT_FAQ = pd.DataFrame([
    # General University Questions
    ["Where is the main library?",
     "The Main library under building  (C). Open 8:30–18:00 Sat–Thu.",
     "library;location;books;study"],
    ["What is the Wi-Fi network?",
     "Free WiFi without password is available in the cafeteria under the G building",
     "wifi;internet;it;network;connection"],
    ["Where is the cafeteria?",
     "We have two cafeterias, the first is a coffee shop to the left of the entrance and the second is on the ground floor of the building (C) . Breakfast 8:00–10:30, lunch 12:00–15:00.",
     "food;cafeteria;dining;meals;restaurant;eat"],
    ["Where can I print or photocopy?",
     "It is next to Building (A).",
     "printing;photocopy;services;printer;copy;documents"],
    ["How do I register for courses?",
     "Go to uhd website login and enter Email and password You can Register For courses. Registration opens one week before each semester.",
     "registration;courses;enrollment;classes;register;enroll"],

    # AI & Data Science Questions
    ["Where are the AI labs located?",
     "AI and Data Science labs (Lab 2, Lab 3, Lab 4, Lab 5) are in the Building (G). Lab schedules are posted outside each lab.",
     "ai;artificial intelligence;data science;labs;location;computer labs"],
    ["What software do I need for AI courses?",
     "For AI courses, you'll need Python, Jupyter Notebook, and various libraries (TensorFlow, PyTorch, scikit-learn). Install instructions are provided in the first class.",
     "ai;software;python;programming;tools;tensorflow;machine learning"],
    ["Who teaches Introduction to Data Science?",
     "Introduction to Data Science is taught by M. Hiwa in AI and DataScience Hall A14. Check the schedule for specific times.",
     "data science;instructor;teacher;lecturer;professor;hiwa"],
    ["Where is Hall A11 and A14?",
     "Hall A11 and Hall A14 are in Building A (Academic Building), first floor. Hall A11 is for lectures, A14 is the AI and Data Science specialized hall.",
     "hall;location;building;classroom;a11;a14;ai;data science"],
    ["What is Problem Solving and Algorithms about?",
     "Problem Solving and Algorithms teaches computational thinking, algorithm design, and programming fundamentals. It's taught by M. Shima with both lecture and lab sessions.",
     "algorithms;problem solving;programming;course;shima;ai;it"],
    ["Who teaches Advanced Mathematics?",
     "Advanced Mathematics is taught by M. Sana on Tuesday 12:00-14:00 in Hall A11 for AI-DS students.",
     "advanced mathematics;math;mathematics;teacher;instructor;sana;ai;data science"],
    ["What is Advanced Mathematics about?",
     "Advanced Mathematics covers advanced mathematical concepts including calculus, linear algebra, statistics, and mathematical foundations for AI and data science applications.",
     "advanced mathematics;math;mathematics;calculus;linear algebra;statistics;ai;data science"],

    # IT Department Questions
    ["Where are the IT labs?",
     "IT labs (Lab 2, Lab 3, Lab 4, Lab 5) are in the Building (G).",
     "it;information technology;labs;computer;location;web design"],
    ["What programming languages do IT students learn?",
     "IT students learn multiple languages including Python (Problem Solving), HTML/CSS/JavaScript (Web Design), and Object-Oriented Programming concepts.",
     "it;programming;languages;coding;web design;oop;python;javascript"],
    ["Who can take Web Design course?",
     "Web Design is specifically for IT students only. The course is taught by M. Karwan in Lab 1 and Lab 4 with hands-on practical sessions.",
     "web design;it;course;students;karwan;lab;website"],
    ["What is Object Oriented Programming?",
     "Object Oriented Programming Concepts (OOP) teaches programming using objects and classes. It's taught by Dr. Mazn in Lab 5 with separate sessions for Groups A1 and A2.",
     "oop;object oriented;programming;course;mazn;java;python;classes"],
    ["Where is the IT Helpdesk?",
     "The IT Helpdesk is in the Technology Building, ground floor. They assist with student email, Wi-Fi access, and technical support. Open 9:00–15:00 daily.",
     "it;helpdesk;support;technical;help;wifi;email;password"],

    # Nursing Department Questions
    ["Where do nursing students do hospital practice?",
     "Nursing students attend Adult Nursing and Lab sessions at the hospital. Sessions are on Saturday (Groups A1 and A2) and Wednesday (all groups).",
     "nursing;hospital;practice;clinical;adult nursing;lab;training"],
    ["Where is Hall G203?",
     "Hall G203 is in Building G (Health Sciences Building), second floor. It's used for Clinical Biochemistry, Adult Nursing lectures, and other nursing courses.",
     "hall;location;g203;nursing;building;classroom"],
    ["Who teaches Clinical Biochemistry?",
     "Clinical Biochemistry is taught by Dr. Shkar with lectures in Hall G203 and lab sessions in Lab F101 and Hall F114 for different groups.",
     "biochemistry;clinical;chemistry;instructor;shkar;nursing;lab"],
    ["What is Medical Microbiology?",
     "Medical Microbiology & Lab covers microorganisms, infections, and laboratory techniques. Taught by M. Gasha and M. Hawbash with both lecture and practical lab sessions.",
     "microbiology;medical;lab;course;nursing;bacteria;microorganisms"],
    ["Where is Lab F101?",
     "Lab F101 is in Building F (Health Sciences Lab Building), first floor. It's used for Clinical Biochemistry and Medical Microbiology lab sessions.",
     "lab;location;f101;nursing;biochemistry;microbiology"],
    ["What are the nursing lab requirements?",
     "For nursing labs, students must wear proper lab coats, closed-toe shoes, and bring required materials as specified by instructors. Lab safety rules are mandatory.",
     "nursing;lab;requirements;safety;equipment;dress code;uniform"],
    ["When is Ethics for Medical Students?",
     "Ethics for Medical Students is taught by M. Baxan on Monday 9:00-10:00 in Hall G203 for Group A1. It covers medical ethics and professional conduct.",
     "ethics;medical;nursing;course;monday;baxan;professional"],

    # Lab & Facility Questions
    ["How do I book a computer lab?",
     "Computer labs can be booked through the department secretary. Priority is given to scheduled classes. Free slots are available for student projects.",
     "lab;booking;computer;reservation;schedule;availability"],
    ["Are labs open after class hours?",
     "Computer labs are open until 17:00 for student use when not scheduled for classes. Check with lab supervisors for availability.",
     "lab;hours;open;time;access;after class;evening"],
    ["Where can I get lab coats for nursing?",
     "Nursing lab coats can be purchased from the Student Center shop or the Nursing Department office. Proper lab attire is mandatory for all practical sessions.",
     "lab coat;nursing;uniform;buy;purchase;requirements;dress"],
], columns=["question", "answer", "tags"])

DEFAULT_SCHEDULE = pd.DataFrame([
    # ========== ARTIFICIAL INTELLIGENCE & DATA SCIENCE ==========
    # Sunday
    ["AI-DS", "Problem Solving and Algorithms", "Sunday", "09:00",
        "11:00", "Hall A11", "M Shima", "Artificial Intelligence"],
    ["AI-DS", "Data Communications", "Sunday", "12:00", "14:00",
        "Hall A11", "M Dana", "Artificial Intelligence"],

    # Monday - Group A1

    ["AI-DS", "Data Communications", "Monday", "09:00", "11:00",
        "Lab 3", "M Dana - Group A2", "Artificial Intelligence"],
    ["AI-DS", "Problem Solving and Algorithms", "Monday", "12:00",
        "14:00", "Lab 4", "M Shima - Group A2", "Artificial Intelligence"],


    # Tuesday
    ["AI-DS", "Introduction to Data Science", "Tuesday", "09:00", "11:00",
        "AI and DataScience Hall A14", "M Hiwa", "Artificial Intelligence"],
    ["AI-DS", "Advanced Mathematics", "Tuesday", "12:00",
        "14:00", "Hall A11", "M Sana", "Artificial Intelligence"],

    # Wednesday

    ["AI-DS", "Introduction to Data Science", "Wednesday", "12:30", "14:30",
        "Lab 4", "M Hiwa - AI and DataScience", "Artificial Intelligence"],

    # Thursday
    ["AI-DS", "Object Oriented Programming Concepts", "Thursday", "09:00",
        "12:00", "Lab 5", "Dr Mazn - Group A2", "Artificial Intelligence"],

    # ========== INFORMATION TECHNOLOGY ==========
    # Sunday
    ["IT", "Problem Solving and Algorithms", "Sunday", "09:00",
        "11:00", "Hall A11", "M Shima", "Information Technology"],
    ["IT", "Data Communications", "Sunday", "11:30", "14:00",
        "Hall A11", "M Dana", "Information Technology"],

    # Monday
    ["IT", "Problem Solving and Algorithms", "Monday", "09:00", "11:00",
        "Lab 2", "M Shima - Group A1", "Information Technology"],
    ["IT", "Data Communications", "Monday", "09:00", "11:00",
        "Lab 3", "M Dana - Group A2", "Information Technology"],
    ["IT", "Problem Solving and Algorithms", "Monday", "12:00", "14:00",
        "Lab 4", "M Shima - Group A2", "Information Technology"],
    ["IT", "Data Communications", "Monday", "12:00", "14:00",
        "Lab 3", "M Dana - Group A1", "Information Technology"],

    # Tuesday
    ["IT", "Web Design (IT Students Only)", "Tuesday", "09:00",
     "11:00", "Lab 1", "M Karwan", "Information Technology"],
    ["IT", "Web Design", "Tuesday", "12:00", "14:00", "Lab 4",
        "M Karwan - Group A1", "Information Technology"],

    # Wednesday
    ["IT", "Object Oriented Programming Concepts", "Wednesday", "09:00",
        "12:00", "Lab 5", "Dr Mazn - Group A1", "Information Technology"],
    ["IT", "Information and Communication Technology (IT Students Only)", "Wednesday",
     "12:30", "14:30", "Hall A11", "M Arez", "Information Technology"],

    # Thursday
    ["IT", "Object Oriented Programming Concepts", "Thursday", "09:00",
        "12:00", "Lab 5", "Dr Mazn - Group A2", "Information Technology"],

    # ========== NURSING ==========
    # Saturday - Group A1
    ["NUR", "Adult Nursing and Lab (hospital)", "Saturday", "09:00",
     "10:00", "Hospital", "Dr. Bayan - Group A1", "Nursing"],
    # Saturday - Group A2
    ["NUR", "Adult Nursing and Lab (hospital)", "Saturday", "12:00",
     "13:00", "Hospital", "Dr. Bayan - Group A2", "Nursing"],

    # Sunday - Group A1 & A2
    ["NUR", "Clinical Biochemistry", "Sunday", "11:00",
        "12:00", "Hall G203", "Dr. Shkar", "Nursing"],
    ["NUR", "Human Growth and Development", "Sunday",
        "13:00", "14:00", "Hall G203", "M. Ari", "Nursing"],

    # Monday - Group A1
    ["NUR", "Ethics for Medical students", "Monday", "09:00",
        "10:00", "Hall G203", "M. Baxan - Group A1", "Nursing"],
    ["NUR", "Clinical Biochemistry & lab", "Monday", "10:00",
        "11:00", "Lab F101", "Dr. Shkar - Group A1", "Nursing"],
    # Monday - Group A2
    ["NUR", "Clinical Biochemistry & lab", "Monday", "14:00",
        "15:00", "Hall F114", "Dr. Shkar - Group A2", "Nursing"],

    # Tuesday - Group A1
    ["NUR", "Medical Microbiology & lab", "Tuesday", "09:00",
        "10:00", "Lab F 101", "M. Gasha - Group A1", "Nursing"],
    ["NUR", "Medical Microbiology & lab", "Tuesday", "14:00",
        "15:00", "Hall G203", "M. Hawbash - Group A1", "Nursing"],
    # Tuesday - Group A2
    ["NUR", "Medical Microbiology & lab", "Tuesday", "11:00",
        "12:00", "Lab F 101", "M. Gasha - Group A2", "Nursing"],

    # Wednesday - Group A1 & A2
    ["NUR", "Adult Nursing and Lab", "Wednesday", "09:00",
        "14:00", "Hall G203", "Dr. Bayan", "Nursing"],

    # ========== MEDICAL LABORATORY SCIENCE (MLS) ==========
    # Sunday
    ["MLS", "Lab Techniques & Instrumentation-[ML-03AMorning]", "Sunday", "09:00",
     "11:00", "F 201", "Hawbash Muhammed-Amin", "Medical Laboratory Science"],
    ["MLS", "Human Biology-[ML-03AMorning]", "Sunday", "13:00",
     "15:00", "F 201", "Huner Hasan Kareem", "Medical Laboratory Science"],

    # Monday
    ["MLS", "Human Biology-[ML-03AMorning]", "Monday", "11:00",
     "12:00", "F 201", "Huner Hasan Kareem", "Medical Laboratory Science"],
    ["MLS", "Ethics for Medical Students-[ML-03AMorning]", "Monday", "12:00",
     "14:00", "F 201", "Bakhan Farih Hamasharif", "Medical Laboratory Science"],

    # Tuesday
    ["MLS", "Analytical Chemistry & Lab-[ML-03AMorning]", "Tuesday", "11:00",
     "13:00", "F 201", "Bayan Salih Azizi", "Medical Laboratory Science"],
    ["MLS", "Analytical Chemistry & Lab-[ML-03AMorning]", "Tuesday", "13:00",
     "15:00", "F 114", "Bayan Salih Azizi", "Medical Laboratory Science"],

    # Wednesday
    ["MLS", "G. Microbiology I & Lab-[ML-03AMorning]", "Wednesday", "09:00",
     "11:00", "F 101", "Gasha Salih Ahmed", "Medical Laboratory Science"],
    ["MLS", "G. Microbiology I & Lab-[ML-03AMorning]", "Wednesday", "13:00",
     "15:00", "F 201", "Gasha Salih Ahmed", "Medical Laboratory Science"],

    # ========== LAW DEPARTMENT ==========
    # Saturday
    ["LAW", "Commercial Law - Principles and Contracts-[LW-2BMorning]", "Saturday", "09:00",
     "11:00", "C 176", "Hardi Tawfiq Mustafa", "Law"],
    ["LAW", "Principles of Islamic Jurisprudence and its Rules-[LW-2BMorning]", "Saturday", "11:00",
     "12:00", "C 176", "Numan Muhammed Almas", "Law"],
    ["LAW", "Administrative Law in English-[LW-2BMorning]", "Saturday", "12:30",
     "14:00", "C 176", "Kardo Kareem Rasheed", "Law"],

    # Sunday
    ["LAW", "Penal Law - Public Part-[LW-2BMorning]", "Sunday", "09:00",
     "10:10", "C 133", "Latif Mustafa Ameen", "Law"],
    ["LAW", "General English Language (Part II)-[LW-2BMorning]", "Sunday", "10:30",
     "12:00", "C 133", "Salam Abdulqadir Abdulrahman", "Law"],
    ["LAW", "Civil Law I - Sources of Obligation-[LW-2BMorning]", "Sunday", "12:00",
     "13:10", "C 133", "Saman Fawzi Omer", "Law"],

    # Monday
    ["LAW", "Constitutional Law - Theory of the State and Political Systems-[LW-2BMorning]", "Monday", "09:00",
     "10:10", "C 133", "Kurdistan Salim Saeed", "Law"],
    ["LAW", "Public International Law-[LW-2BMorning]", "Monday", "10:30",
     "11:40", "C 133", "Karwan Awrahman Ismail", "Law"],
    ["LAW", "Administrative Law - General Principles-[LW-2BMorning]", "Monday", "12:30",
     "14:00", "C 133", "Rawsht Muhammed Ameen", "Law"],

    # Tuesday
    ["LAW", "Civil Law I - Sources of Obligation-[LW-2BMorning]", "Tuesday", "09:00",
     "10:10", "C 133", "Saman Fawzi Omer", "Law"],
    ["LAW", "Principles of Islamic Jurisprudence and its Rules-[LW-2BMorning]", "Tuesday", "10:30",
     "11:40", "C 133", "Numan Muhammed Almas", "Law"],
    ["LAW", "Constitutional Law - Theory of the State and Political Systems-[LW-2BMorning]", "Tuesday", "12:30",
     "13:40", "C 133", "Kurdistan Salim Saeed", "Law"],

    # Wednesday
    ["LAW", "Penal Law - Public Part-[LW-2BMorning]", "Wednesday", "09:00",
     "10:10", "C 133", "Latif Mustafa Ameen", "Law"],
    ["LAW", "Public International Law-[LW-2BMorning]", "Wednesday", "11:00",
     "12:10", "C 133", "Karwan Awrahman Ismail", "Law"],

    # ========== ENGLISH LANGUAGE DEPARTMENT - 3rd A ==========
    # Sunday
    ["ENG-3A", "Academic Writing", "Sunday", "09:00",
     "10:00", "170", "Mr. Araz", "English Language"],
    ["ENG-3A", "Pronunciation", "Sunday", "11:00",
     "12:00", "170", "Dr. Inaad", "English Language"],
    ["ENG-3A", "Novel", "Sunday", "13:00",
     "14:00", "170", "Dr. Chalak", "English Language"],

    # Monday
    ["ENG-3A", "Drama", "Monday", "09:00",
     "10:00", "170", "Dr. Maysaa", "English Language"],
    ["ENG-3A", "Pronunciation", "Monday", "11:00",
     "12:00", "170", "Dr. Inaad", "English Language"],
    ["ENG-3A", "Grammar", "Monday", "12:00",
     "13:00", "170", "Ms. Sahima", "English Language"],

    # Tuesday
    ["ENG-3A", "Novel", "Tuesday", "09:00",
     "10:00", "170", "Dr. Chalak", "English Language"],
    ["ENG-3A", "Drama", "Tuesday", "10:00",
     "11:00", "170", "Dr. Maysaa", "English Language"],
    ["ENG-3A", "Grammar", "Tuesday", "12:00",
     "13:00", "170", "Ms. Sahima", "English Language"],
    ["ENG-3A", "Academic Writing", "Tuesday", "14:00",
     "15:00", "170", "Mr. Araz", "English Language"],

    # Wednesday
    ["ENG-3A", "Grammar", "Wednesday", "10:00",
     "11:00", "170", "Ms. Sahima", "English Language"],
    ["ENG-3A", "Drama", "Wednesday", "12:00",
     "13:00", "170", "Dr. Maysaa", "English Language"],
    ["ENG-3A", "Pronunciation", "Wednesday", "13:00",
     "14:00", "170", "Dr. Inaad", "English Language"],

    # Thursday
    ["ENG-3A", "Novel", "Thursday", "09:00",
     "10:00", "170", "Dr. Chalak", "English Language"],
    ["ENG-3A", "Academic Writing", "Thursday", "11:00",
     "12:00", "170", "Mr. Araz", "English Language"],
    ["ENG-3A", "Academic Writing", "Thursday", "12:00",
     "13:00", "170", "Mr. Araz", "English Language"],
], columns=["course_code", "course_name", "day", "start_time", "end_time", "hall", "lecturer", "department"])

# ================== LOADERS ==================


def _read_csv_safely(path: Path):
    """Read CSV with multiple encoding attempts"""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue
    raise ValueError("Could not read CSV with supported encodings.")


def load_faq():
    """Load FAQ data from CSV or use default"""
    if FAQ_PATH.exists():
        try:
            df = _read_csv_safely(FAQ_PATH)
            if not {"question", "answer"}.issubset(df.columns):
                st.warning(
                    "⚠️ 'faq.csv' missing required columns — using default sample.")
                return DEFAULT_FAQ, "built-in"
            if "tags" not in df.columns:
                df["tags"] = ""
            return df, "local file"
        except Exception as e:
            st.warning(
                f"⚠️ Error reading faq.csv: {e} — using default sample.")
            return DEFAULT_FAQ, "built-in"
    return DEFAULT_FAQ, "built-in"


def load_schedule():
    """Load schedule data from CSV or use default"""
    if SCHED_PATH.exists():
        try:
            df = _read_csv_safely(SCHED_PATH)
            required = {"course_code", "course_name", "day",
                        "start_time", "end_time", "hall", "lecturer"}
            if not required.issubset(df.columns):
                st.warning(
                    "⚠️ 'schedule.csv' missing required columns — using default sample.")
                return DEFAULT_SCHEDULE, "built-in"
            if "department" not in df.columns:
                df["department"] = "General"
            df["day"] = df["day"].astype(str).str.title()
            return df, "local file"
        except Exception as e:
            st.warning(
                f"⚠️ Error reading schedule.csv: {e} — using default sample.")
            return DEFAULT_SCHEDULE, "built-in"
    return DEFAULT_SCHEDULE, "built-in"


# Load data
faq_df, faq_src = load_faq()
sched_df, sched_src = load_schedule()

# Status indicator with time and day
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown(
        f'<span class="status-badge">📚 FAQ: {faq_src}</span>', unsafe_allow_html=True)
with col2:
    st.markdown(
        f'<span class="status-badge">📅 Schedule: {sched_src}</span>', unsafe_allow_html=True)
with col3:
    # Use client-side time display for accurate timezone on Streamlit Cloud
    components.html(
        """
        <div style="
            background: rgba(245, 222, 179, 0.6);
            backdrop-filter: blur(15px);
            padding: 0.8rem 1.8rem;
            border-radius: 35px;
            display: inline-block;
            box-shadow: 0 8px 25px rgba(245, 222, 179, 0.4);
            font-size: 1rem;
            font-weight: 800;
            color: #1a1a1a;
            border: 2px solid rgba(245, 222, 179, 0.7);
            text-shadow: 0 2px 6px rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease;
        " id="time-badge">
            🕐 <span id="time-display">Loading...</span>
        </div>
        
        <script>
        function updateTime() {
            const now = new Date();
            const timeStr = now.toLocaleTimeString('en-US', {
                hour: '2-digit', 
                minute: '2-digit', 
                hour12: false
            });
            const dayStr = now.toLocaleDateString('en-US', {
                weekday: 'long'
            });
            const timeDisplay = document.getElementById('time-display');
            if (timeDisplay) {
                timeDisplay.textContent = `${timeStr} ${dayStr}`;
            }
        }
        
        // Update immediately
        updateTime();
        
        // Update every minute
        setInterval(updateTime, 60000);
        
        // Update when tab becomes visible
        document.addEventListener('visibilitychange', function() {
            if (!document.hidden) {
                updateTime();
            }
        });
        </script>
        """,
        height=50
    )

st.markdown("<br>", unsafe_allow_html=True)

# ================== FAQ SEARCH ENGINE ==================


@st.cache_resource(show_spinner=False)
def build_faq_index(df: pd.DataFrame):
    """Build TF-IDF index for FAQ search"""
    text = (df["question"].astype(str) + " " +
            df["tags"].fillna("").astype(str)).str.lower()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=1000)
    X = vec.fit_transform(text)
    return vec, X


faq_vec, faq_X = build_faq_index(faq_df)


def _tokenize(s: str):
    """Extract tokens from string"""
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def faq_search(query: str, df: pd.DataFrame, vec, X):
    """Search FAQ using hybrid TF-IDF and tag matching with improved accuracy"""
    q = query.lower().strip()
    if not q:
        return None, 0.0

    # TF-IDF similarity
    tfidf = cosine_similarity(vec.transform([q]), X).flatten()

    # Tag overlap score
    qtok = _tokenize(q)
    tag_sets = df["tags"].fillna("").str.lower().apply(
        lambda t: set(re.split(r"[;,\s]+", t.strip())))
    overlap_counts = tag_sets.apply(lambda s: len(qtok & s)).astype(float)
    if overlap_counts.max() > 0:
        overlap = overlap_counts / overlap_counts.max()
    else:
        overlap = overlap_counts

    # Direct keyword matching for better accuracy
    keyword_scores = []
    for idx, row in df.iterrows():
        question_lower = row["question"].lower()
        answer_lower = row["answer"].lower()
        tags_lower = str(row["tags"]).lower()

        # Check for exact keyword matches
        score = 0
        for token in qtok:
            if token in question_lower:
                score += 3  # Higher weight for question matches
            if token in answer_lower:
                score += 2  # Medium weight for answer matches
            if token in tags_lower:
                score += 1  # Lower weight for tag matches

        keyword_scores.append(score)

    keyword_scores = np.array(keyword_scores)
    if keyword_scores.max() > 0:
        keyword_scores = keyword_scores / keyword_scores.max()
    else:
        keyword_scores = np.zeros_like(keyword_scores)

    # Combine scores (40% TF-IDF, 30% tag overlap, 30% keyword matching)
    combined = 0.4 * tfidf + 0.3 * overlap.values + 0.3 * keyword_scores
    idx = combined.argmax()
    return int(idx), float(combined[idx])


# ================== SCHEDULE SEARCH ==================
DAY_MAP = {
    "sun": "sunday", "sunday": "sunday",
    "mon": "monday", "monday": "monday",
    "tue": "tuesday", "tuesday": "tuesday",
    "wed": "wednesday", "wednesday": "wednesday",
    "thu": "thursday", "thursday": "thursday",
    "fri": "friday", "friday": "friday",
    "sat": "saturday", "saturday": "saturday",
}


def parse_day(text: str):
    """Extract day from text"""
    t = text.lower()
    for k, v in DAY_MAP.items():
        if k in t:
            return v.title()
    return None


def extract_code(text: str):
    """Extract course code from text"""
    m = re.search(r"\b([A-Za-z]{2,5})\s?-?\s?(\d{2,3})\b", text)
    return (m.group(1).upper() + m.group(2)) if m else None


STOP_WORDS = {"for", "the", "and", "of", "in", "on",
              "class", "course", "where", "when", "is", "what"}


def _words(s: str):
    """Extract meaningful words"""
    return [w for w in re.findall(r"[a-z]+", s.lower()) if len(w) >= 3 and w not in STOP_WORDS]


# ================== UI TABS - ALL VISIBLE ==================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["❓ FAQ", "📅 Class Schedule", "📋 Full Timetable", "🏛️ Department", "ℹ️ About"])

# ========== FAQ TAB ==========
with tab1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 📚 Ask a Question")
    st.write("*Get answers about library, registrar, wifi, printing, and more.*")

    # Personalization
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "Good morning"
    elif 12 <= current_hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    st.markdown(f"**{greeting}, Student 👋**")
    st.markdown("<br>", unsafe_allow_html=True)

    q = st.text_input("Type your question here...", key="faq_q",
                      placeholder="e.g., Where is the library?")

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_btn = st.button("🔍 Ask", key="faq_btn", use_container_width=True)

    # Smart suggestions - removed for now to avoid session state conflicts
    # st.markdown("**💡 Quick Questions:**")
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     if st.button("What is my next class?", key="suggest1", use_container_width=True):
    #         pass
    # with col2:
    #     if st.button("Who teaches Advanced Mathematics?", key="suggest2", use_container_width=True):
    #         pass
    # with col3:
    #     if st.button("Where can I get wifi password?", key="suggest3", use_container_width=True):
    #         pass

    if ask_btn:
        if not q.strip():
            st.info("ℹ️ Please enter a question.")
        else:
            with st.spinner("Searching..."):
                idx, score = faq_search(q, faq_df, faq_vec, faq_X)
                if score > 0.25:
                    st.success(f"✅ Match found (confidence: {score:.0%})")
                    st.markdown("**Answer:**")
                    # Glass card styling for answer
                    st.markdown(f"""
                    <div class="answer-box">
                        {faq_df.loc[idx, "answer"]}
                        <div style="margin-top: 1rem; font-size: 0.8rem; color: #ccc; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 0.5rem;">
                            📚 Source: FAQ Database (updated Oct 2025)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(
                        "⚠️ Sorry, I couldn't find a good match. Try rephrasing or being more specific!")

    # Show sample questions
    with st.expander("📋 Sample Questions"):
        st.markdown("- Where is the main library?")
        st.markdown("- What is the Wi-Fi network?")
        st.markdown("- Where can I print documents?")
        st.markdown("- How do I register for courses?")
        st.markdown("- Where are the AI labs located?")
        st.markdown("- Who teaches Data Science?")
    st.markdown('</div>', unsafe_allow_html=True)

# ========== SCHEDULE TAB ==========
with tab2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Find a Class")
    st.write("*Search for a specific course by name, code, or day*")
    st.markdown("<br>", unsafe_allow_html=True)

    # Department filter
    depts = sorted([str(d).strip() for d in sched_df["department"].dropna().unique()
                    if str(d).strip() and str(d).lower() != "nan"])
    dept_choice = st.selectbox("Filter by Department:", [
                               "All Departments"] + depts, key="dept_select")

    active_df = sched_df if dept_choice == "All Departments" else sched_df[
        sched_df["department"] == dept_choice]

    qs = st.text_input("Ask about a class...", key="sched_q",
                       placeholder="e.g., Problem Solving, AI-DS, Tuesday classes")

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_sched_btn = st.button(
            "🔍 Search", key="sched_btn", use_container_width=True)

    if ask_sched_btn:
        if not qs.strip():
            st.info("ℹ️ Please enter a course name or code.")
        else:
            with st.spinner("Searching schedule..."):
                # Search for matching courses
                q_lower = qs.lower()

                # Find all courses that match the search
                matches = active_df[
                    active_df["course_name"].str.lower().str.contains(q_lower, na=False) |
                    active_df["course_code"].str.lower().str.contains(q_lower, na=False) |
                    active_df["day"].str.lower().str.contains(q_lower, na=False) |
                    active_df["lecturer"].str.lower(
                    ).str.contains(q_lower, na=False)
                ]

                if not matches.empty:
                    # Group by unique courses
                    unique_courses = matches.groupby(
                        'course_code').first().reset_index()

                    st.success(
                        f"✅ Found {len(unique_courses)} course(s) matching your search!")
                    st.markdown("---")

                    # Display each unique course
                    for _, course in unique_courses.iterrows():
                        st.markdown(
                            f"### {course['course_code']} – {course['course_name']}")
                        st.markdown(
                            f"**📚 Department:** {course['department']}")

                        # Get all sessions for this course
                        course_sessions = matches[matches['course_code']
                                                  == course['course_code']].sort_values('day')

                        st.markdown("**📅 Schedule:**")
                        for _, session in course_sessions.iterrows():
                            st.markdown(
                                f"- **{session['day']}**: ⏰ {session['start_time']} - {session['end_time']} | 🏢 {session['hall']} | 👨‍🏫 {session['lecturer']}")

                        st.markdown("---")
                else:
                    st.warning(
                        "⚠️ No courses found matching your search. Try different keywords!")
    st.markdown('</div>', unsafe_allow_html=True)

# ========== TIMETABLE TAB ==========
with tab3:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Complete University Timetable")
    st.write("*View all classes across all departments*")
    st.markdown("<br>", unsafe_allow_html=True)

    # Filter options
    col1, col2, col3 = st.columns([1.2, 1, 0.8])

    with col1:
        # Department filter
        all_depts = ["All Departments"] + sorted([
            str(d).strip() for d in sched_df["department"].dropna().unique()
            if str(d).strip() and str(d).lower() != "nan"
        ])
        dept_filter = st.selectbox("Department:", all_depts, key="tt_dept")

    with col2:
        # Day filter
        unique_days = list(set(sched_df["day"].dropna().tolist()))
        day_order = ["Saturday", "Sunday", "Monday",
                     "Tuesday", "Wednesday", "Thursday", "Friday"]
        sorted_days = [day for day in day_order if day in unique_days]
        day_options = ["All Days"] + sorted_days
        day_filter = st.selectbox("Day:", options=day_options, key="tt_day")

    with col3:
        # Sort option
        sort_option = st.selectbox(
            "Sort by:", ["Time", "Course Code", "Department"], key="tt_sort")

    # Apply filters
    filtered_df = sched_df.copy()

    if dept_filter != "All Departments":
        filtered_df = filtered_df[filtered_df["department"] == dept_filter]

    if day_filter != "All Days":
        filtered_df = filtered_df[filtered_df["day"] == day_filter]

    # Sort
    if sort_option == "Time":
        filtered_df = filtered_df.sort_values(["day", "start_time"])
    elif sort_option == "Course Code":
        filtered_df = filtered_df.sort_values("course_code")
    else:
        filtered_df = filtered_df.sort_values(["department", "start_time"])

    # Display count
    st.markdown(f"**Showing {len(filtered_df)} classes**")
    st.markdown("---")

    # Display as cards grouped by day
    if not filtered_df.empty:
        if day_filter == "All Days":
            day_order = ["Saturday", "Sunday", "Monday",
                         "Tuesday", "Wednesday", "Thursday", "Friday"]
            for day in day_order:
                day_classes = filtered_df[filtered_df["day"] == day]
                if not day_classes.empty:
                    st.markdown(f"### 📅 {day}")

                    for _, row in day_classes.iterrows():
                        st.markdown(f"""
                        <div class="class-card">
                            <div style="display: grid; grid-template-columns: 2fr 2fr 1.5fr 1.5fr; gap: 1rem;">
                                <div>
                                    <strong style="font-size: 1.1rem; color: #f5deb3;">{row['course_code']}</strong><br>
                                    <span style="color: #6c757d;">{row['course_name']}</span>
                                </div>
                                <div>
                                    <strong>👨‍🏫 {row['lecturer']}</strong><br>
                                    <span style="color: #6c757d;">📚 {row['department']}</span>
                                </div>
                                <div>
                                    <strong>⏰ {row['start_time']}</strong><br>
                                    <span style="color: #6c757d;">→ {row['end_time']}</span>
                                </div>
                                <div>
                                    <strong>🏢 {row['hall']}</strong>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            for _, row in filtered_df.iterrows():
                st.markdown(f"""
                <div class="class-card">
                    <div style="display: grid; grid-template-columns: 2fr 2fr 1.5fr 1.5fr; gap: 1rem;">
                        <div>
                            <strong style="font-size: 1.1rem; color: #f5deb3;">{row['course_code']}</strong><br>
                            <span style="color: #6c757d;">{row['course_name']}</span>
                        </div>
                        <div>
                            <strong>👨‍🏫 {row['lecturer']}</strong><br>
                            <span style="color: #6c757d;">📚 {row['department']}</span>
                        </div>
                        <div>
                            <strong>⏰ {row['start_time']}</strong><br>
                            <span style="color: #6c757d;">→ {row['end_time']}</span>
                        </div>
                        <div>
                            <strong>🏢 {row['hall']}</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Download option
        st.markdown("---")
        st.markdown("### 📥 Export Timetable")
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"uhd_timetable_{day_filter.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )

        # Display as table option
        with st.expander("📊 View as Table"):
            st.dataframe(
                filtered_df[["course_code", "course_name", "day",
                             "start_time", "end_time", "hall", "lecturer", "department"]],
                hide_index=True
            )
    else:
        st.info("No classes found with the selected filters.")
    st.markdown('</div>', unsafe_allow_html=True)

# ========== DEPARTMENT TAB ==========
with tab4:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 🏛️ University Departments (2025-2026)")
    st.write("*Browse all available academic departments at UHD*")
    st.markdown("<br>", unsafe_allow_html=True)

    # Department list in a simpler format
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Medical & Health Sciences:**
        - 🦷 Dentistry
        - 👩‍⚕️ Nursing
        - 🧪 Medical Laboratory Science
        - 💊 Pharmacy
        """)

    with col2:
        st.markdown("""
        **Technology & Business:**
        - 💻 Information Technology (IT)
        - 🤖 Artificial Intelligence
        - 💰 Accounting and Banking Science
        - 📈 Business Administration
        """)

    st.markdown("""
    **Liberal Arts & Law:**
    - 📚 English Language
    - ⚖️ Law
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== ABOUT TAB ==========
with tab5:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ About UHD AI Chatbot")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5deb3 0%, #a0522d 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
        <h2 style="color: white; margin-top: 0;">🎓 UHD AI Chatbot</h2>
        <p style="font-size: 1.1rem; line-height: 1.6;">
            An intelligent assistant for the <strong>University of Human Development</strong>, 
            created to help students and staff quickly find answers about the library, 
            class schedules, registrar office, printing, and other university services — 
            all in one simple and friendly interface.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👥 Development Team")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="developer-card" style="background: rgba(255, 255, 255, 0.08); padding: 1.5rem; border-radius: 12px; 
                    text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.3); border: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="font-size: 3rem;">👨‍💻</div>
            <h4 style="color: #ffffff; margin: 0.5rem 0; line-height: 1.2;">Dyar Abdulla</h4>
            <p style="color: #e0e0e0; margin: 0;">Developer</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="developer-card" style="background: rgba(255, 255, 255, 0.08); padding: 1.5rem; border-radius: 12px; 
                    text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.3); border: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="font-size: 3rem;">👨‍💻</div>
            <h4 style="color: #ffffff; margin: 0.5rem 0; line-height: 1.2;">Anas Sarkawt</h4>
            <p style="color: #e0e0e0; margin: 0;">Developer</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="developer-card" style="background: rgba(255, 255, 255, 0.08); padding: 1.5rem; border-radius: 12px; 
                    text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.3); border: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="font-size: 3rem;">👨‍💻</div>
            <h4 style="color: #ffffff; margin: 0.5rem 0; line-height: 1.2;">Drood Muhammed</h4>
            <p style="color: #e0e0e0; margin: 0;">Developer</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ✨ Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - 🔍 **Smart FAQ Search** - Ask questions about university services
        - 📅 **Class Schedule Finder** - Search courses by name, code, or day
        - 📋 **Complete Timetable** - View all classes with filters
        """)

    with col2:
        st.markdown("""
        - 🎯 **AI-Powered Matching** - TF-IDF algorithm for accurate results
        - 📥 **Export Feature** - Download schedules as CSV
        - 🎨 **Beautiful Interface** - Modern and user-friendly design
        """)

    st.markdown("---")

    st.markdown("### 📊 Data Sources")
    st.markdown(f"""
    - **FAQ Database:** {faq_src} ({len(faq_df)} questions)
    - **Schedule Database:** {sched_src} ({len(sched_df)} classes)
    """)

    st.markdown("---")

    st.markdown("### 💡 How to Use")
    st.markdown("""
    1. **FAQ Tab**: Type your question about university services
    2. **Class Schedule Tab**: Search for specific courses
    3. **Full Timetable Tab**: Browse all classes with filters
    4. **About Tab**: Learn more about this chatbot
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p><strong>© 2024 University of Human Development</strong></p>
    <p>Festival Demo Version | Developed with ❤️ by Dyar Abdulla, Anas Sarkawt & Drood Muhammed</p>
</div>
""", unsafe_allow_html=True)
