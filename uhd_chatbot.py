import streamlit as st
import pandas as pd
import re
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
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .university-name {
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        color: #f0f0f0;
        font-size: 1.3rem;
        margin-top: 0.5rem;
    }
    
    /* Tab styling - FIXED TO ALWAYS SHOW */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: white;
        padding: 1rem 0;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e8eaf0;
        transform: translateY(-2px);
        border-color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.5);
    }
    
    /* Input styling */
    .stTextInput input, .stSelectbox select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    .stTextInput input:hover, .stSelectbox select:hover {
        border-color: #764ba2;
    }
    
    /* Success/Warning/Info boxes */
    .stSuccess, .stWarning, .stInfo, .stError {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .stSuccess {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    
    .stWarning {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    
    .stInfo {
        background-color: #d1ecf1;
        border-left-color: #17a2b8;
    }
    
    .stError {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    
    /* Logo container */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .logo-img {
        border-radius: 10px;
        border: 3px solid white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling for containers */
    .stContainer {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .stContainer:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Download button styling */
    .stDownloadButton button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(40, 167, 69, 0.4);
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
    """, unsafe_allow_html=True)
    st.info("💡 **Tip:** Place your university logo as 'uhd_logo.png' or 'uhd_logo.jpg' in the same folder as this script to display it.")

# ================== FILE PATHS ==================
FAQ_PATH = SCRIPT_DIR / "faq.csv"
SCHED_PATH = SCRIPT_DIR / "schedule.csv"

# ================== DEFAULT DATA ==================
DEFAULT_FAQ = pd.DataFrame([
    ["Where is the main library?",
     "The main library is next to the Administrative Building (north gate). Open 8:30–18:00 Sat–Thu.",
     "library;location;books;study"],
    ["How do I get my student ID?",
     "Go to the Registrar Office with your admission letter + one photo. IDs issued 9:00–14:00.",
     "registrar;id;student card"],
    ["What is the Wi-Fi network?",
     "Use 'Uni-Students'. Login with your student email + password from IT Helpdesk.",
     "wifi;internet;it;network;connection"],
    ["Where is the cafeteria?",
     "Student Center (ground floor). Breakfast 8:00–10:30, lunch 12:00–15:00.",
     "food;cafeteria;dining;meals;restaurant;eat"],
    ["What are the festival timings?",
     "The University Festival runs for three days, 10:00–16:00 daily at the main courtyard.",
     "festival;events;celebration;activities"],
    ["Where can I print or photocopy?",
     "Printing and photocopying are available at the Library (ground floor) and the Student Center shop.",
     "printing;photocopy;services;printer;copy;documents"],
    ["How do I register for courses?",
     "Course registration is done online through the student portal. Registration opens one week before each semester.",
     "registration;courses;enrollment;classes;register;enroll"],
    ["Where is the parking lot?",
     "Student parking is available behind Building C. Parking permits required from Security Office.",
     "parking;car;vehicle;transportation;drive"],
    ["Where are the AI labs located?",
     "AI and Data Science labs (Lab 2, Lab 3, Lab 4, Lab 5) are in the Technology Building. Lab schedules are posted outside each lab.",
     "ai;artificial intelligence;data science;labs;location;computer labs"],
    ["What software do I need for AI courses?",
     "For AI courses, you'll need Python, Jupyter Notebook, and various libraries (TensorFlow, PyTorch, scikit-learn). Install instructions are provided in the first class.",
     "ai;software;python;programming;tools;tensorflow;machine learning"],
], columns=["question", "answer", "tags"])

DEFAULT_SCHEDULE = pd.DataFrame([
    ["AI-DS", "Problem Solving and Algorithms", "Sunday", "09:00", "11:00", "Hall A11", "M Shima", "Artificial Intelligence"],
    ["AI-DS", "Data Communications", "Sunday", "12:00", "14:00", "Hall A11", "M Dana", "Artificial Intelligence"],
    ["IT", "Problem Solving and Algorithms", "Sunday", "09:00", "11:00", "Hall A11", "M Shima", "Information Technology"],
    ["IT", "Data Communications", "Sunday", "11:30", "14:00", "Hall A11", "M Dana", "Information Technology"],
    ["NUR", "Clinical Biochemistry", "Sunday", "11:00", "12:00", "Hall G203", "Dr. Shkar", "Nursing"],
    ["LAW", "Penal Law - Public Part", "Sunday", "09:00", "10:10", "C 133", "Latif Mustafa Ameen", "Law"],
    ["ENG-3A", "Academic Writing", "Sunday", "09:00", "10:00", "170", "Mr. Araz", "English Language"],
], columns=["course_code", "course_name", "day", "start_time", "end_time", "hall", "lecturer", "department"])

# ================== LOADERS ==================
def _read_csv_safely(path):
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
                st.warning("⚠️ 'faq.csv' missing required columns — using default sample.")
                return DEFAULT_FAQ, "built-in"
            if "tags" not in df.columns:
                df["tags"] = ""
            return df, "local file"
        except Exception as e:
            st.warning(f"⚠️ Error reading faq.csv: {e} — using default sample.")
            return DEFAULT_FAQ, "built-in"
    return DEFAULT_FAQ, "built-in"

def load_schedule():
    """Load schedule data from CSV or use default"""
    if SCHED_PATH.exists():
        try:
            df = _read_csv_safely(SCHED_PATH)
            required = {"course_code", "course_name", "day", "start_time", "end_time", "hall", "lecturer"}
            if not required.issubset(df.columns):
                st.warning("⚠️ 'schedule.csv' missing required columns — using default sample.")
                return DEFAULT_SCHEDULE, "built-in"
            if "department" not in df.columns:
                df["department"] = "General"
            df["day"] = df["day"].astype(str).str.title()
            return df, "local file"
        except Exception as e:
            st.warning(f"⚠️ Error reading schedule.csv: {e} — using default sample.")
            return DEFAULT_SCHEDULE, "built-in"
    return DEFAULT_SCHEDULE, "built-in"

# Load data
faq_df, faq_src = load_faq()
sched_df, sched_src = load_schedule()

# Status indicator
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.caption(f"📚 FAQ Source: **{faq_src}**")
with col2:
    st.caption(f"📅 Schedule Source: **{sched_src}**")
with col3:
    st.caption(f"🕐 {datetime.now().strftime('%H:%M')}")

# ================== FAQ SEARCH ENGINE ==================
@st.cache_resource(show_spinner=False)
def build_faq_index(df):
    """Build TF-IDF index for FAQ search"""
    text = (df["question"].astype(str) + " " + df["tags"].fillna("").astype(str)).str.lower()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=1000)
    X = vec.fit_transform(text)
    return vec, X

faq_vec, faq_X = build_faq_index(faq_df)

def faq_search(query, df, vec, X):
    """Search FAQ using hybrid TF-IDF and tag matching"""
    q = query.lower().strip()
    if not q:
        return None, 0.0
    tfidf = cosine_similarity(vec.transform([q]), X).flatten()
    idx = tfidf.argmax()
    return int(idx), float(tfidf[idx])

# ================== UI TABS ==================
tab1, tab2, tab3, tab4 = st.tabs(["❓ FAQ", "📅 Class Schedule", "📋 Full Timetable", "ℹ️ About"])

# FAQ TAB
with tab1:
    st.markdown("### Ask a Question")
    st.markdown("*Get answers about library, registrar, wifi, printing, and more.*")
    q = st.text_input("Type your question here...", key="faq_q", placeholder="e.g., Where is the library?")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_btn = st.button("🔍 Ask", key="faq_btn", use_container_width=True)
    
    if ask_btn:
        if not q.strip():
            st.info("ℹ️ Please enter a question.")
        else:
            with st.spinner("Searching..."):
                idx, score = faq_search(q, faq_df, faq_vec, faq_X)
                if score > 0.25:
                    st.success(f"✅ Match found (confidence: {score:.0%})")
                    st.markdown("**Answer:**")
                    st.info(faq_df.loc[idx, "answer"])
                else:
                    st.warning("⚠️ Sorry, I couldn't find a good match. Try rephrasing or being more specific!")

# SCHEDULE TAB
with tab2:
    st.markdown("### Find a Class")
    st.markdown("*Search for a specific course by name, code, or day*")
    depts = sorted([str(d).strip() for d in sched_df["department"].dropna().unique() if str(d).strip() and str(d).lower() != "nan"])
    dept_choice = st.selectbox("Filter by Department:", ["All Departments"] + depts, key="dept_select")
    active_df = sched_df if dept_choice == "All Departments" else sched_df[sched_df["department"] == dept_choice]
    
    qs = st.text_input("Ask about a class...", key="sched_q", placeholder="e.g., Problem Solving, AI-DS, Tuesday classes")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_sched_btn = st.button("🔍 Search", key="sched_btn", use_container_width=True)
    
    if ask_sched_btn and qs.strip():
        q_lower = qs.lower()
        matches = active_df[
            active_df["course_name"].str.lower().str.contains(q_lower, na=False) |
            active_df["course_code"].str.lower().str.contains(q_lower, na=False) |
            active_df["day"].str.lower().str.contains(q_lower, na=False)
        ]
        if not matches.empty:
            st.success(f"✅ Found {len(matches)} result(s)!")
            for _, row in matches.iterrows():
                st.markdown(f"**{row['course_code']}** – {row['course_name']}")
                st.markdown(f"📅 {row['day']} | ⏰ {row['start_time']}-{row['end_time']} | 🏢 {row['hall']} | 👨‍🏫 {row['lecturer']}")
                st.markdown("---")
        else:
            st.warning("⚠️ No courses found.")

# TIMETABLE TAB
with tab3:
    st.markdown("### 📋 Complete University Timetable")
    st.markdown("*View all classes across all departments*")
    
    col1, col2, col3 = st.columns([1.2, 1, 0.8])
    with col1:
        all_depts = ["All Departments"] + sorted([str(d).strip() for d in sched_df["department"].dropna().unique() if str(d).strip() and str(d).lower() != "nan"])
        dept_filter = st.selectbox("Department:", all_depts, key="tt_dept")
    with col2:
        day_options = ["All Days", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        day_filter = st.selectbox("Day:", options=day_options, key="tt_day")
    with col3:
        sort_option = st.selectbox("Sort by:", ["Time", "Course Code", "Department"], key="tt_sort")
    
    filtered_df = sched_df.copy()
    if dept_filter != "All Departments":
        filtered_df = filtered_df[filtered_df["department"] == dept_filter]
    if day_filter != "All Days":
        filtered_df = filtered_df[filtered_df["day"] == day_filter]
    
    st.markdown(f"**Showing {len(filtered_df)} classes**")
    st.markdown("---")
    
    if not filtered_df.empty:
        for _, row in filtered_df.iterrows():
            col_a, col_b, col_c, col_d = st.columns([2, 2, 1.5, 1.5])
            with col_a:
                st.markdown(f"**{row['course_code']}**")
                st.caption(row['course_name'])
            with col_b:
                st.markdown(f"👨‍🏫 {row['lecturer']}")
                st.caption(f"📚 {row['department']}")
            with col_c:
                st.markdown(f"⏰ {row['start_time']}")
                st.caption(f"→ {row['end_time']}")
            with col_d:
                st.markdown(f"🏢 **{row['hall']}**")
            st.markdown("---")

# ABOUT TAB
with tab4:
    st.markdown("### ℹ️ About UHD AI Chatbot")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0; box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);">
        <h2 style="margin-top: 0; color: white;">🎓 UHD AI Chatbot</h2>
        <p style="font-size: 1.1rem; line-height: 1.6;">
            An intelligent assistant for the <strong>University of Human Development</strong>, created by <strong>Dyar Abdulla</strong>, <strong>Anas Sarkawt</strong>, and <strong>Drood Muhammed</strong>.
        </p>
        <p style="font-size: 1rem; line-height: 1.6;">
            This chatbot helps students and staff quickly find answers about the library, class schedules, registrar office, printing, and other university services — all in one simple and friendly interface.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 👥 Development Team")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div style="text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;"><div style="font-size: 3rem;">👨‍💻</div><h4>Dyar Abdulla</h4><p>Developer</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div style="text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;"><div style="font-size: 3rem;">👨‍💻</div><h4>Anas Sarkawt</h4><p>Developer</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div style="text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;"><div style="font-size: 3rem;">👨‍💻</div><h4>Drood Muhammed</h4><p>Developer</p></div>""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2024 University of Human Development | Created by Dyar Abdulla, Anas Sarkawt, and Drood Muhammed")
