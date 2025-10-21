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
        background-color: #ffffff;
    }
    
    /* Override Streamlit's default dark background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
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
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: white;
        padding: 10px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        color: #495057;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Content cards */
    .content-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styling */
    .stTextInput input, .stSelectbox select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    /* Success/Warning/Info boxes */
    .stSuccess {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stWarning {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stInfo {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        border-radius: 10px;
        padding: 1rem;
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
    
    /* Class card styling */
    .class-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-left: 5px solid #667eea;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .class-card:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Status badge */
    .status-badge {
        background: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        font-size: 0.9rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 2px solid #e9ecef;
        margin-top: 3rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Section headers */
    h1, h2, h3 {
        color: #495057;
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

# ================== FILE PATHS ==================
FAQ_PATH = SCRIPT_DIR / "faq.csv"
SCHED_PATH = SCRIPT_DIR / "schedule.csv"

# ================== DEFAULT DATA ==================
DEFAULT_FAQ = pd.DataFrame([
    # General University Questions
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

    # AI & Data Science Questions
    ["Where are the AI labs located?",
     "AI and Data Science labs (Lab 2, Lab 3, Lab 4, Lab 5) are in the Technology Building. Lab schedules are posted outside each lab.",
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

    # IT Department Questions
    ["Where are the IT labs?",
     "IT labs are located in the Technology Building. Lab 1 is for Web Design, and other labs are shared with AI department for programming courses.",
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
    ["AI-DS", "Problem Solving and Algorithms", "Monday", "09:00",
        "11:00", "Lab 2", "M Shima - Group A1", "Artificial Intelligence"],
    ["AI-DS", "Data Communications", "Monday", "09:00", "11:00",
        "Lab 3", "M Dana - Group A2", "Artificial Intelligence"],
    ["AI-DS", "Problem Solving and Algorithms", "Monday", "12:00",
        "14:00", "Lab 4", "M Shima - Group A2", "Artificial Intelligence"],
    ["AI-DS", "Data Communications", "Monday", "12:00", "14:00",
        "Lab 3", "M Dana - Group A1", "Artificial Intelligence"],

    # Tuesday
    ["AI-DS", "Introduction to Data Science", "Tuesday", "09:00", "11:00",
        "AI and DataScience Hall A14", "M Hiwa", "Artificial Intelligence"],
    ["AI-DS", "Advanced Mathematics", "Tuesday", "12:00",
        "14:00", "Hall A11", "M Sana", "Artificial Intelligence"],

    # Wednesday
    ["AI-DS", "Object Oriented Programming Concepts", "Wednesday", "09:00",
        "12:00", "Lab 5", "Dr Mazn - Group A1", "Artificial Intelligence"],
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

    # ========== SAMPLE DATA FOR OTHER DEPARTMENTS (To be updated) ==========
    # Medical Laboratory Science
    ["MLS", "Clinical Chemistry", "Sunday", "09:00", "11:00",
        "MLS Lab 1", "Dr. Shwan", "Medical Laboratory Science"],
    ["MLS", "Hematology", "Monday", "10:00", "12:00",
        "MLS Lab 2", "Dr. Avan", "Medical Laboratory Science"],

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

# Status indicator
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.markdown(f'<span class="status-badge">📚 FAQ: {faq_src}</span>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<span class="status-badge">📅 Schedule: {sched_src}</span>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<span class="status-badge">🕐 {datetime.now().strftime("%H:%M")}</span>', unsafe_allow_html=True)

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
    """Search FAQ using hybrid TF-IDF and tag matching"""
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

    # Combine scores (70% TF-IDF, 30% tag overlap)
    combined = 0.7 * tfidf + 0.3 * overlap.values
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
tab1, tab2, tab3, tab4 = st.tabs(["❓ FAQ", "📅 Class Schedule", "📋 Full Timetable", "ℹ️ About"])

# ========== FAQ TAB ==========
with tab1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### Ask a Question")
    st.markdown("*Get answers about library, registrar, wifi, printing, and more.*")

    q = st.text_input("Type your question here...", key="faq_q",
                      placeholder="e.g., Where is the library?")

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
                    st.warning(
                        "⚠️ Sorry, I couldn't find a good match. Try rephrasing or being more specific!")

    # Show sample questions
    with st.expander("📋 Sample Questions"):
        st.markdown("- Where is the main library?")
        st.markdown("- How do I get my student ID?")
        st.markdown("- What is the Wi-Fi network?")
        st.markdown("- Where can I print documents?")
        st.markdown("- How do I register for courses?")
        st.markdown("- Where are the AI labs located?")
        st.markdown("- Who teaches Data Science?")
    st.markdown('</div>', unsafe_allow_html=True)

# ========== SCHEDULE TAB ==========
with tab2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### Find a Class")
    st.markdown("*Search for a specific course by name, code, or day*")

    # Department filter
    depts = sorted([str(d).strip() for d in sched_df["department"].dropna().unique()
                    if str(d).strip() and str(d).lower() != "nan"])
    dept_choice = st.selectbox("Filter by Department:", [
                               "All Departments"] + depts, key="dept_select")

    active_df = sched_df if dept_choice == "All Departments" else sched_df[
        sched_df["department"] == dept_choice]

    st.markdown("---")

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
                    active_df["lecturer"].str.lower().str.contains(q_lower, na=False)
                ]

                if not matches.empty:
                    # Group by unique courses
                    unique_courses = matches.groupby('course_code').first().reset_index()

                    st.success(f"✅ Found {len(unique_courses)} course(s) matching your search!")
                    st.markdown("---")

                    # Display each unique course
                    for _, course in unique_courses.iterrows():
                        st.markdown(f"### {course['course_code']} – {course['course_name']}")
                        st.markdown(f"**📚 Department:** {course['department']}")

                        # Get all sessions for this course
                        course_sessions = matches[matches['course_code'] == course['course_code']].sort_values('day')

                        st.markdown("**📅 Schedule:**")
                        for _, session in course_sessions.iterrows():
                            st.markdown(f"- **{session['day']}**: ⏰ {session['start_time']} - {session['end_time']} | 🏢 {session['hall']} | 👨‍🏫 {session['lecturer']}")

                        st.markdown("---")
                else:
                    st.warning("⚠️ No courses found matching your search. Try different keywords!")
    st.markdown('</div>', unsafe_allow_html=True)

# ========== TIMETABLE TAB ==========
with tab3:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Complete University Timetable")
    st.markdown("*View all classes across all departments*")

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
        day_order = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        sorted_days = [day for day in day_order if day in unique_days]
        day_options = ["All Days"] + sorted_days
        day_filter = st.selectbox("Day:", options=day_options, key="tt_day")

    with col3:
        # Sort option
        sort_option = st.selectbox("Sort by:", ["Time", "Course Code", "Department"], key="tt_sort")

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
            day_order = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            for day in day_order:
                day_classes = filtered_df[filtered_df["day"] == day]
                if not day_classes.empty:
                    st.markdown(f"### 📅 {day}")

                    for _, row in day_classes.iterrows():
                        st.markdown(f"""
                        <div class="class-card">
                            <div style="display: grid; grid-template-columns: 2fr 2fr 1.5fr 1.5fr; gap: 1rem;">
                                <div>
                                    <strong style="font-size: 1.1rem; color: #667eea;">{row['course_code']}</strong><br>
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
                            <strong style="font-size: 1.1rem; color: #667eea;">{row['course_code']}</strong><br>
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

# ========== ABOUT TAB ==========
with tab4:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ About UHD AI Chatbot")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
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
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; 
                    text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
            <div style="font-size: 3rem;">👨‍💻</div>
            <h4 style="color: #667eea; margin: 0.5rem 0;">Dyar Abdulla</h4>
            <p style="color: #6c757d; margin: 0;">Developer</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; 
                    text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
            <div style="font-size: 3rem;">👨‍💻</div>
            <h4 style="color: #667eea; margin: 0.5rem 0;">Anas Sarkawt</h4>
            <p style="color: #6c757d; margin: 0;">Developer</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; 
                    text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
            <div style="font-size: 3rem;">👨‍💻</div>
            <h4 style="color: #667eea; margin: 0.5rem 0;">Drood Muhammed</h4>
            <p style="color: #6c757d; margin: 0;">Developer</p>
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
