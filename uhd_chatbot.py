import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
import numpy as np
from pathlib import Path
from collections.abc import Mapping
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import base64
import smtplib
from email.mime.text import MIMEText

# ================== TRANSLATIONS ==================
EN_TRANSLATIONS = {
    "title": "UNIVERSITY OF HUMAN DEVELOPMENT",
    "subtitle": "AI FAQ & Class Schedule Chatbot",
    "ask_question": "Ask a Question",
    "type_here": "Type your question here...",
    "ask_button": "Ask",
    "sample_questions": "Sample Questions",
    "sample_questions_helper": "Click a question to fill the box automatically.",
    "faq": "FAQ",
    "class_schedule": "Class Schedule",
    "full_timetable": "Full Timetable",
    "department": "Department",
    "departments": "Departments",
    "about": "About",
    "feedback": "Feedback",
    "departments_section_heading": "University Departments (2025-2026)",
    "departments_description": "Browse all available academic departments at UHD.",
    "departments_list_left": """
        **Health & Medical Sciences**
        - 💊 Pharmaceuticals
        - 👩‍⚕️ Nursing
        - 🧬 Medical Laboratory Science
        - ⚖️ Law
        - 📚 English Language
        """,
    "departments_list_right": """
        **Technology & Administration**
        - 🤖 Artificial Intelligence and Data Science
        - 💻 Information Technology (IT)
        - 💰 Accounting
        - 🏛️ Labor Administration
        """,
    "about_heading": "About UHD AI Chatbot",
    "about_description": "An intelligent assistant for the University of Human Development, created to help students and staff quickly find answers about the library, class schedules, registrar office, printing, and other university services — all in one simple and friendly interface.",
    "about_intro_html": """
    <div style="background: linear-gradient(135deg, #f5deb3 0%, #a0522d 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
        <h2 style="color: white; margin-top: 0;">🎓 UHD AI Chatbot</h2>
        <p style="font-size: 1.1rem; line-height: 1.6;">
            {about_description}
        </p>
    </div>
    """,
    "development_team_heading": "Development Team",
    "developer_role": "Developer",
    "features_heading": "Features",
    "features_list_left": """
        - 🔍 **Smart FAQ Search** - Ask questions about university services
        - 📅 **Class Schedule Finder** - Search courses by name, code, or day
        - 📋 **Complete Timetable** - View all classes with filters
        """,
    "features_list_right": """
        - 🎯 **AI-Powered Matching** - TF-IDF algorithm for accurate results
        - 📥 **Export Feature** - Download schedules as CSV
        - 🎨 **Beautiful Interface** - Modern and user-friendly design
        """,
    "data_sources_heading": "Data Sources",
    "data_sources_list": """
    - **FAQ Database:** {faq_src} ({faq_count} questions)
    - **Schedule Database:** {sched_src} ({sched_count} classes)
    """,
    "how_to_heading": "How to Use",
    "how_to_list": """
    1. **FAQ Tab**: Type your question about university services
    2. **Class Schedule Tab**: Search for specific courses
    3. **Full Timetable Tab**: Browse all classes with filters
    4. **About Tab**: Learn more about this chatbot
    """,
    "search": "Search",
    "filter_by": "Filter by",
    "filter_by_department": "Filter by Department:",
    "all_departments": "All Departments",
    "all_days": "All Days",
    "day_label": "Day:",
    "department_label": "Department:",
    "sort_by": "Sort by:",
    "sort_time": "Time",
    "sort_course_code": "Course Code",
    "sort_department": "Department",
    "showing_classes": "Showing {count} classes",
    "no_classes_filters": "No classes found with the selected filters.",
    "course_department_label": "Department:",
    "course_schedule_label": "Schedule:",
    "send_feedback": "Send Feedback",
    "name_optional": "Name (optional)",
    "contact_optional": "Email or phone (optional)",
    "message": "Your message",
    "submit": "Submit",
    "feedback_topic_label": "What is this about?",
    "feedback_topic_options": [
        "General suggestion",
        "Bug report",
        "Missing information",
        "Design / usability",
        "Other"
    ],
    "feedback_message_placeholder": "Tell us what happened or what you would like to see improved.",
    "feedback_submit_button": "Send feedback",
    "good_morning": "Good morning, Student",
    "good_afternoon": "Good afternoon, Student",
    "good_evening": "Good evening, Student",
    "good_morning_short": "Good morning",
    "good_afternoon_short": "Good afternoon",
    "good_evening_short": "Good evening",
    "enter_question_prompt": "Please enter a question.",
    "thinking_spinner": "Thinking…",
    "answer_label": "Answer:",
    "match_found": "Match found (confidence: {score:.0%})",
    "no_match_warning": "Sorry, I couldn't find a good match. Try rephrasing or being more specific!",
    "error_response": "Sorry, couldn't get a response. Try again.",
    "faq_answer_source": "Source: FAQ Database (updated Oct 2025)",
    "ask_about_class": "Ask about a class...",
    "class_search_placeholder": "e.g., Problem Solving, AI-DS, Tuesday classes",
    "enter_course_prompt": "Please enter a course name or code.",
    "search_schedule_spinner": "Searching schedule...",
    "course_match_success": "Found {count} course(s) matching your search!",
    "no_classes_found": "No classes found matching your search.",
    "schedule_heading": "Find a Class",
    "schedule_description": "Search for a specific course by name, code, or day",
    "faq_intro": "Get answers about library, registrar, wifi, printing, and more.",
    "timetable_heading": "Complete University Timetable",
    "timetable_description": "View all classes across all departments",
    "download_csv": "Download as CSV",
    "export_timetable": "Export Timetable",
    "view_as_table": "View as Table",
    "no_feedback": "No feedback has been submitted yet.",
    "feedback_heading": "Share Feedback or Report an Issue",
    "feedback_subheading": "Let us know about any problems, ideas, or requests so we can improve the UHD chatbot experience.",
    "feedback_checkbox": "I agree that my feedback may be reviewed by the UHD chatbot team.",
    "feedback_missing_message": "Please add some details to your message before sending.",
    "feedback_missing_consent": "Please confirm that we may review your feedback.",
    "feedback_saved": "Thanks! ✅",
    "feedback_saved_email": "Your feedback has been recorded and emailed to the team.",
    "feedback_saved_no_email": "Your feedback has been recorded.",
    "feedback_error": "Sorry, we couldn't save your message. Please try again later.",
    "download_feedback_log": "Download feedback log",
    "feedback_log_caption": "Feedback entries are saved locally to `{filename}` and emailed to the UHD team.",
    "feedback_email_skipped": "Email notification skipped: {error}",
    "language_label": "Language",
    "language_name_en": "English",
    "language_name_ku": "Kurdish(Sorani)",
    "language_name_ar": "Arabic",
    "current_time_label": "",
    "updates_banner": "Fresh updates are on the way",
    "updates_subtitle": "The best update will be available soon.",
    "update_banner_title": "Fresh updates are on the way",
    "update_banner_subtitle": "The best update will be available soon.",
    "home_button": "← Home",
    "search_placeholder": "e.g., Where is the library?",
    "search_button": "Search",
    "faq_csv_missing": "⚠️ 'faq.csv' missing required columns — using default sample.",
    "faq_csv_read_error": "⚠️ Error reading faq.csv: {error} — using default sample.",
    "schedule_csv_missing": "⚠️ 'schedule.csv' missing required columns — using default sample.",
    "schedule_csv_read_error": "⚠️ Error reading schedule.csv: {error} — using default sample.",
    "error_details": "Error details: {error}",
    "faq_samples": [
        "Where is the main library?",
        "How do I get my student ID?",
        "What is the Wi-Fi network?",
        "Where can I print documents?",
        "How do I register for courses?",
        "Where are the AI labs located?",
        "Who teaches Data Science?"
    ],
}

KU_TRANSLATION_OVERRIDES = {
    "title": "زانکۆی گەشەپێدانی مرۆیی",
    "subtitle": "چاتبۆتی پرسیار و خشتەی وانەکان بە زیرەکی دەستکرد",
    "ask_question": "پرسیارێک بکە",
    "type_here": "پرسیارەکەت لێرە بنووسە...",
    "ask_button": "پرسیار بکە",
    "sample_questions": "پرسیارە نموونەییەکان",
    "sample_questions_helper": "کلیک بکە بۆ پڕکردنەوەی خانەکە بە خۆکار.",
    "faq": "پرسیارە باوەکان",
    "class_schedule": "خشتەی وانەکان",
    "full_timetable": "خشتەی تەواو",
    "department": "بەش",
    "departments": "بەشەکان",
    "about": "دەربارە",
    "about_description": "یارمەتیاری زێرەکی دەستکرد بۆ زانکۆی گەشەپێدانی مرۆیی کە بۆ خوێندکاران و کارکنان درووست کراوە بۆ ئەوەی خێرا وەلامی پرسیارەکانیان بدۆزنەوە دەربارە کتێبخانە، خشتەی وانەکان، تۆماری قوتابخانە، چاپکردن و دەستەواژەی خزمەتگوزاریەکانی زانکۆ — هەمووی لە ڕووکاریەکی سادە و هاوسەنگدا.",
    "feedback": "ڕەخنە و پێشنیار",
    "departments_section_heading": "بەشەکانی زانکۆ (٢٠٢٥-٢٠٢٦)",
    "departments_description": "بەشەکان ببینە و زانیاری زیاتر بدۆزەوە.",
    "search": "گەڕان",
    "filter_by": "پاڵاوتن بەپێی",
    "filter_by_department": "پاڵاوتن بەپێی بەش:",
    "all_departments": "هەموو بەشەکان",
    "all_days": "هەموو ڕۆژەکان",
    "day_label": "ڕۆژ:",
    "department_label": "بەش:",
    "sort_by": "ڕیزکردن بەپێی:",
    "sort_time": "کات",
    "sort_course_code": "کۆدی وانە",
    "sort_department": "بەش",
    "showing_classes": "{count} وانە نیشان دەدرێت",
    "no_classes_filters": "هیچ وانەیەک بەپێی ئەم پاڵاوتنە نەدۆزرایە.",
    "course_department_label": "بەش:",
    "course_schedule_label": "خشتەی وانە:",
    "send_feedback": "ناردنی ڕەخنە",
    "name_optional": "ناو (ئارەزوومەندانە)",
    "contact_optional": "ئیمەیڵ یان ژمارەی تەلەفۆن (ئارەزوومەندانە)",
    "message": "پەیامەکەت",
    "feedback_message_placeholder": "وردەکاری کێشەکەت یان پێشنیارەکەت بنووسە.",
    "feedback_submit_button": "ناردن",
    "good_morning": "بەیانی باش، قوتابی",
    "good_afternoon": "دوای نیوەڕۆ باش، قوتابی",
    "good_evening": "ئێوارە باش، قوتابی",
    "good_morning_short": "بەیانی باش",
    "good_afternoon_short": "نیڤڕۆ باش",
    "good_evening_short": "ئێوارە باش",
    "enter_question_prompt": "تکایە پرسیارەکەت بنووسە.",
    "thinking_spinner": "چاوەڕێ بکە...",
    "answer_label": "وەلام:",
    "match_found": "برگەی یەکخستن {score:.0%} بدۆزرایەوە",
    "no_match_warning": "ببورە، وەلامێکی گونجاو نەدۆزرایەوە. جارێکی تر هەوڵ بدە!",
    "error_response": "هەڵەیەک ڕوویدا. تکایە دووبارە هەوڵ بدەرەوە",
    "faq_answer_source": "سەرچاوە: بنکەی پرسیارەکان (نوێکراوی تشرینی یەکەمی ٢٠٢٥)",
    "ask_about_class": "دەر باری وانەیەک بپرسە...",
    "class_search_placeholder": "نموونە: وانەی هەڵبژاردن، AI-DS، وانەکانی سەیشەمە",
    "enter_course_prompt": "تکایە ناو یان کۆدی وانەیەک بنووسە.",
    "search_schedule_spinner": "چاوەڕێ بکە...",
    "course_match_success": "{count} وانە دۆزرایەوە!",
    "no_classes_found": "هیچ ئەنجامێک نەدۆزرایەوە",
    "schedule_heading": "وانەیەک بدۆزەوە",
    "schedule_description": "گەڕان بۆ وانەکان بە ناو، کۆد یان ڕۆژ",
    "faq_intro": "وەلام بۆ پرسیارەکانی کتێبخانە، تۆمارکردن، وایفای، چاپکردن و زیاتر بدۆزەوە.",
    "timetable_heading": "خشتەی تەواوی زانکۆ",
    "timetable_description": "هەموو وانەکان بە پاڵاوتن ببینە",
    "download_csv": "داگرتنی CSV",
    "export_timetable": "هێنانی خشتەکە",
    "view_as_table": "نیشاندانی وەک خشتە",
    "no_feedback": "هێشتا هیچ ڕەخنەیەک نەگەیشتووە.",
    "feedback_heading": "بەشداریکردن بە ڕەخنە یان کێشە",
    "feedback_subheading": "کێشەت، بیرۆکەت یان پێشنیارەکەت پێمان بڵێ بۆ باشترکردنی ئەزموونەکە.",
    "feedback_checkbox": "پەسەندی ئەوە دەکەم کە ڕەخنەکەم بەلای تیمەکەوە بپشکنرێت.",
    "feedback_missing_message": "تکایە پەیامەکەت پێش ناردن درێژتر بکە.",
    "feedback_missing_consent": "تکایە پشتڕاست بکەوە کە دەتوانین ڕەخنەکەت بپشکنین.",
    "feedback_saved": "سوپاس! ✅",
    "feedback_saved_email": "پەیامەکەت تۆمار کرا و بۆ تیمەکە نێردرا.",
    "feedback_saved_no_email": "پەیامەکەت تۆمار کرا.",
    "feedback_error": "هەڵەیەک ڕوویدا. تکایە دووبارە هەوڵ بدەرەوە",
    "download_feedback_log": "داگرتنی تۆماری ڕەخنەکان",
    "feedback_log_caption": "ڕەخنەکان لە ناو `{filename}` دەخزنرێن و بۆ تیمەکە نێردرێن.",
    "feedback_email_skipped": "ئاگاداری ئیمەیڵ نەنووسرا: {error}",
    "language_label": "زمان",
    "language_name_en": "ئینگلیزی",
    "language_name_ku": "کوردی (سۆرانی)",
    "language_name_ar": "عەرەبی",
    "current_time_label": "",
    "updates_banner": "نوێکارییەکان لە ڕێگادان",
    "updates_subtitle": "باشترین نوێکاری بەم زووانە بەردەست دەبێت",
    "update_banner_title": "نوێکردنەوەی تازە لە ڕێگادانە",
    "update_banner_subtitle": "باشترین نوێکردنەوە بە زووترین کات دەگەیەنرێت.",
    "home_button": "← ماڵەوە",
    "search_placeholder": "نموونە: کتێبخانە لە کوێیە؟",
    "search_button": "گەڕان",
    "feedback": "ڕەخنە و پێشنیار",
    "success": "سوپاس! ✅",
    "faq_samples": [
        "کتێبخانەی سەرەکی لە کوێیە؟",
        "چۆن دەتوانم کارتی قوتابی دەستم بێت؟",
        "ناوی تۆڕی وایفای چەندە؟",
        "لای کوێ دەتوانم بەلگە چاپ بکەم؟",
        "چۆن خۆم تۆمار بکەم بۆ وانەکان؟",
        "لابۆراتۆری AI لە کوێیە؟"
    ],
}

AR_TRANSLATION_OVERRIDES = {
    "title": "جامعة التنمية البشرية",
    "subtitle": "روبوت الدردشة للأسئلة والجدول الدراسي بالذكاء الاصطناعي",
    "ask_question": "اطرح سؤالاً",
    "type_here": "اكتب سؤالك هنا...",
    "ask_button": "اسأل",
    "sample_questions": "أسئلة نموذجية",
    "sample_questions_helper": "انقر لملء خانة السؤال تلقائياً.",
    "faq": "الأسئلة الشائعة",
    "class_schedule": "جدول الحصص",
    "full_timetable": "الجدول الكامل",
    "department": "القسم",
    "departments": "الأقسام",
    "about": "حول",
    "about_description": "مساعد ذكي لجامعة التنمية البشرية، صُمم لمساعدة الطلبة والموظفين على العثور بسرعة على الإجابات المتعلقة بالمكتبة وجداول الحصص ومكتب التسجيل والطباعة والخدمات الجامعية الأخرى — كل ذلك ضمن واجهة بسيطة وودودة.",
    "feedback": "الملاحظات",
    "search": "بحث",
    "filter_by": "تصفية حسب",
    "filter_by_department": "تصفية حسب القسم:",
    "all_departments": "جميع الأقسام",
    "all_days": "جميع الأيام",
    "day_label": "اليوم:",
    "department_label": "القسم:",
    "sort_by": "ترتيب حسب:",
    "sort_time": "الوقت",
    "sort_course_code": "رمز المقرر",
    "sort_department": "القسم",
    "showing_classes": "عرض {count} حصة",
    "no_classes_filters": "لا توجد حصص وفق عوامل التصفية المختارة.",
    "course_department_label": "القسم:",
    "course_schedule_label": "مواعيد الحصة:",
    "send_feedback": "إرسال الملاحظات",
    "name_optional": "الاسم (اختياري)",
    "contact_optional": "البريد الإلكتروني أو الهاتف (اختياري)",
    "message": "رسالتك",
    "feedback_message_placeholder": "اكتب تفاصيل المشكلة أو اقتراحك هنا.",
    "feedback_submit_button": "إرسال",
    "good_morning": "صباح الخير، طالب",
    "good_afternoon": "مساء الخير، طالب",
    "good_evening": "مساء الخير، طالب",
    "good_morning_short": "صباح الخير",
    "good_afternoon_short": "مساء الخير",
    "good_evening_short": "مساء الخير",
    "enter_question_prompt": "الرجاء إدخال سؤال.",
    "thinking_spinner": "جاري التحميل...",
    "answer_label": "الإجابة:",
    "match_found": "تم العثور على تطابق بنسبة {score:.0%}",
    "no_match_warning": "عذراً، لم نعثر على إجابة مناسبة. حاول بصياغة مختلفة!",
    "error_response": "حدث خطأ. يرجى المحاولة مرة أخرى",
    "faq_answer_source": "المصدر: قاعدة بيانات الأسئلة (محدثة أكتوبر 2025)",
    "ask_about_class": "اسأل عن حصة...",
    "class_search_placeholder": "مثال: حل المسائل، AI-DS، حصص الثلاثاء",
    "enter_course_prompt": "الرجاء إدخال اسم أو رمز الحصة.",
    "search_schedule_spinner": "جاري التحميل...",
    "course_match_success": "تم العثور على {count} حصة!",
    "no_classes_found": "لم يتم العثور على نتائج",
    "schedule_heading": "اعثر على حصة",
    "schedule_description": "ابحث عن الحصص بالاسم أو الرمز أو اليوم",
    "faq_intro": "احصل على إجابات حول المكتبة، التسجيل، شبكة الواي فاي، الطباعة والمزيد.",
    "timetable_heading": "الجدول الجامعي الكامل",
    "timetable_description": "استعرض جميع الحصص مع خيارات التصفية",
    "download_csv": "تنزيل CSV",
    "export_timetable": "تصدير الجدول",
    "view_as_table": "عرض كجدول",
    "no_feedback": "لا توجد ملاحظات بعد.",
    "feedback_heading": "شارك ملاحظة أو مشكلة",
    "feedback_subheading": "أخبرنا بالمشاكل أو الأفكار لتحسين تجربة روبوت الجامعة.",
    "feedback_checkbox": "أوافق على أن يتم الاطلاع على ملاحظتي من قبل فريق الروبوت.",
    "feedback_missing_message": "الرجاء كتابة بعض التفاصيل قبل الإرسال.",
    "feedback_missing_consent": "الرجاء تأكيد السماح لنا بمراجعة ملاحظتك.",
    "feedback_saved": "شكراً! ✅",
    "feedback_saved_email": "تم تسجيل رسالتك وإرسالها إلى الفريق.",
    "feedback_saved_no_email": "تم تسجيل رسالتك.",
    "feedback_error": "حدث خطأ. يرجى المحاولة مرة أخرى",
    "download_feedback_log": "تنزيل سجل الملاحظات",
    "feedback_log_caption": "تُحفظ الملاحظات في الملف `{filename}` ويتم إرسالها إلى الفريق.",
    "feedback_email_skipped": "تعذّر إرسال البريد الإلكتروني: {error}",
    "language_label": "اللغة",
    "language_name_en": "الإنجليزية",
    "language_name_ku": "الكردية (السورانية)",
    "language_name_ar": "العربية",
    "current_time_label": "",
    "updates_banner": "التحديثات قادمة",
    "updates_subtitle": "أفضل تحديث سيكون متاحًا قريبًا",
    "update_banner_title": "تحديثات جديدة في الطريق",
    "update_banner_subtitle": "سيصل أفضل تحديث قريباً.",
    "home_button": "← الرئيسية",
    "search_placeholder": "مثال: أين تقع المكتبة؟",
    "search_button": "بحث",
    "faq_csv_missing": "⚠️ الملف 'faq.csv' يفتقد الأعمدة المطلوبة — سيتم استخدام العينة المدمجة.",
    "faq_csv_read_error": "⚠️ حدث خطأ أثناء قراءة faq.csv: {error} — سيتم استخدام العينة المدمجة.",
    "schedule_csv_missing": "⚠️ الملف 'schedule.csv' يفتقد الأعمدة المطلوبة — سيتم استخدام العينة المدمجة.",
    "schedule_csv_read_error": "⚠️ حدث خطأ أثناء قراءة schedule.csv: {error} — سيتم استخدام العينة المدمجة.",
    "error_details": "تفاصيل الخطأ: {error}",
    "faq_samples": [
        "أين تقع المكتبة الرئيسية؟",
        "كيف أحصل على هوية الطالب؟",
        "ما هي شبكة الواي فاي؟",
        "أين يمكنني طباعة المستندات؟",
        "كيف أسجل للمقررات؟",
        "أين تقع مختبرات الذكاء الاصطناعي؟"
    ],
}

translations = {
    "en": EN_TRANSLATIONS,
    "ku": EN_TRANSLATIONS | KU_TRANSLATION_OVERRIDES,
    "ar": EN_TRANSLATIONS | AR_TRANSLATION_OVERRIDES,
}


def translate(lang: str, key: str, **kwargs):
    lang_dict = translations.get(lang, translations["en"])
    if key not in lang_dict:
        fallback = translations["en"].get(key, key)
    else:
        fallback = lang_dict[key]
        if fallback is None:
            fallback = translations["en"].get(key, key)
    if isinstance(fallback, str) and kwargs:
        try:
            return fallback.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            pass
    return fallback



# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="UHD AI Chatbot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ================== LANGUAGE STATE ==================
if "language" not in st.session_state:
    st.session_state["language"] = "en"


def is_rtl_language() -> bool:
    return st.session_state.get("language") in {"ku", "ar"}


@contextmanager
def rtl_container():
    if is_rtl_language():
        st.markdown("<div dir='rtl' style='text-align: right;'>",
                    unsafe_allow_html=True)
    try:
        yield
    finally:
        if is_rtl_language():
            st.markdown("</div>", unsafe_allow_html=True)


current_language = st.session_state["language"]

language_codes = ["en", "ku", "ar"]
language_label = translate(current_language, "language_label")
with rtl_container():
    selected_language = st.sidebar.selectbox(
        language_label,
        language_codes,
        index=language_codes.index(current_language),
        format_func=lambda code: translate(
            current_language, f"language_name_{code}")
    )

with rtl_container():
    col_lang1, col_lang2, col_lang3 = st.columns(3)
    with col_lang1:
        if st.button(f"🇬🇧 {translate(current_language, 'language_name_en')}", key="lang_en_btn"):
            st.session_state["language"] = "en"
            st.rerun()
    with col_lang2:
        if st.button(f"🟢 {translate(current_language, 'language_name_ku')}", key="lang_ku_btn"):
            st.session_state["language"] = "ku"
            st.rerun()
    with col_lang3:
        if st.button(f"🇸🇦 {translate(current_language, 'language_name_ar')}", key="lang_ar_btn"):
            st.session_state["language"] = "ar"
            st.rerun()

st.markdown(
    """
<style>
.stButton > button {
    border-radius: 20px;
    padding: 8px 20px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
</style>
""",
    unsafe_allow_html=True,
)

if selected_language != current_language:
    if selected_language == "ku":
        st.session_state["language"] = "ku"
    elif selected_language == "ar":
        st.session_state["language"] = "ar"
    else:
        st.session_state["language"] = "en"
    st.rerun()

current_language = st.session_state["language"]


def tl(key: str, **kwargs):
    return translate(current_language, key, **kwargs)

def get_greeting() -> str:
    hour = datetime.now().hour
    if 6 <= hour < 12:
        return tl("good_morning")
    elif 12 <= hour < 18:
        return tl("good_afternoon")
    else:
        return tl("good_evening")

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

    .update-banner {
        background: linear-gradient(135deg, rgba(135, 206, 235, 0.85) 0%, rgba(245, 222, 179, 0.92) 100%);
        backdrop-filter: blur(15px);
        padding: 1.25rem 2rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
        display: flex;
        align-items: center;
        gap: 1rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.35);
        color: #11364d !important;
    }

    .update-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.35), transparent);
        animation: banner-shimmer 4s ease-in-out infinite;
    }

    .update-banner-icon {
        font-size: 1.8rem;
        animation: pulse 2.8s ease-in-out infinite;
        color: #0b2b3c !important;
        position: relative;
        z-index: 1;
    }

    .update-text {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        position: relative;
        z-index: 1;
        color: #11364d !important;
    }

    .update-text strong {
        font-size: 1.1rem;
        letter-spacing: 0.6px;
        color: #0b2b3c !important;
    }

    .update-text span {
        font-size: 0.95rem;
        color: #134a66 !important;
    }

    @keyframes banner-shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
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
    
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #a0a0a0 !important;
        opacity: 1;
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
         padding: 12px 24px !important;
         margin: 0 !important;
         transition: all 0.3s ease !important;
         position: relative !important;
         min-width: 180px !important;
         display: inline-flex !important;
         justify-content: center !important;
         align-items: center !important;
         white-space: nowrap !important;
     }
     
     .stTabs [data-baseweb="tab"]:hover {
         background: rgba(255, 255, 255, 0.15) !important;
         backdrop-filter: blur(20px) !important;
         border: 1px solid rgba(255, 255, 255, 0.3) !important;
         transform: translateY(-1px) !important;
         box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
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

    /* ========== RESPONSIVE ADJUSTMENTS ========== */
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem !important;
        }

        .header-container {
            padding: 1.5rem !important;
            text-align: center !important;
        }

        .logo-container {
            flex-direction: column !important;
            align-items: center !important;
            text-align: center !important;
        }

        .logo-container img {
            width: 80px !important;
        }

        .university-name {
            font-size: 1.6rem !important;
        }

        .subtitle {
            font-size: 1rem !important;
        }

        .update-banner {
            flex-direction: column !important;
            text-align: center !important;
            gap: 0.5rem !important;
            padding: 1rem 1.25rem !important;
        }

        .update-banner-icon {
            font-size: 1.4rem !important;
        }

        .content-card,
        .glass-card {
            padding: 1.25rem !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: wrap !important;
            justify-content: center !important;
            gap: 0.75rem !important;
        }

        .stTabs [data-baseweb="tab"] {
            width: 100% !important;
            min-width: 0 !important;
            text-align: center !important;
            padding: 12px 20px !important;
        }

        .status-badge {
            font-size: 0.9rem !important;
            padding: 0.6rem 1.2rem !important;
        }

        .class-card {
            padding: 1.25rem !important;
        }

        .class-card div[style*="grid-template-columns"] {
            display: grid !important;
            grid-template-columns: 1fr !important;
            gap: 0.75rem !important;
        }

        .class-card strong {
            font-size: 1rem !important;
        }

        .stColumns {
            display: block !important;
        }

        .stColumns > div {
            width: 100% !important;
            margin-bottom: 1rem !important;
        }
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


# ================== EMAIL HELPERS ==================
def format_feedback_email(entry: dict) -> MIMEText:
    """Generate a MIMEText email for a feedback entry"""
    body = f"""
New UHD Chatbot feedback received:

Timestamp: {entry.get('timestamp')}
Name: {entry.get('name') or 'Anonymous'}
Contact: {entry.get('contact') or 'Not provided'}
Topic: {entry.get('topic')}

Message:
{entry.get('message')}
""".strip()

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = f"[UHD Chatbot] New feedback: {entry.get('topic')}"
    return msg


def try_send_feedback_email(entry: dict):
    """Send email notification if SMTP secrets are configured"""
    if not hasattr(st, "secrets") or "smtp" not in st.secrets:
        return False, "SMTP settings not configured."

    smtp_conf = st.secrets["smtp"]
    required_keys = {"user", "password"}
    if not required_keys.issubset(smtp_conf.keys()):
        return False, "SMTP credentials incomplete."

    server = smtp_conf.get("server", "smtp.gmail.com")
    port = int(smtp_conf.get("port", 587))
    use_tls = smtp_conf.get("use_tls", True)
    sender = smtp_conf.get("sender", smtp_conf["user"])
    recipients = smtp_conf.get("to", FEEDBACK_DEFAULT_RECIPIENTS)
    if isinstance(recipients, str):
        recipients = [r.strip() for r in recipients.split(",") if r.strip()]
    if not recipients:
        recipients = [FEEDBACK_DEFAULT_RECIPIENTS]

    message = format_feedback_email(entry)
    message["From"] = sender
    message["To"] = ", ".join(recipients)

    try:
        with smtplib.SMTP(server, port, timeout=15) as smtp:
            if use_tls:
                smtp.starttls()
            smtp.login(smtp_conf["user"], smtp_conf["password"])
            smtp.send_message(message, from_addr=sender, to_addrs=recipients)
        return True, None
    except Exception as exc:
        return False, str(exc)


# ================== HEADER ==================
LOGO_FILE = get_logo()

# Create header with logo
if LOGO_FILE:
    logo_b64 = get_base64_image(LOGO_FILE)
    if logo_b64:
        title_text = tl("title")
        subtitle_text = tl("subtitle")
        with rtl_container():
            st.markdown(f"""
            <div class="header-container">
                <div class="logo-container">
                    <img src="data:image/png;base64,{logo_b64}" class="logo-img" width="100">
                    <div>
                        <h1 class="university-name">{title_text}</h1>
                        <p class="subtitle">🎓 {subtitle_text}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        title_text = tl("title")
        subtitle_text = tl("subtitle")
        with rtl_container():
            st.markdown(f"""
            <div class="header-container">
                <h1 class="university-name">🎓 {title_text}</h1>
                <p class="subtitle">{subtitle_text}</p>
            </div>
            """, unsafe_allow_html=True)
else:
    title_text = tl("title")
    subtitle_text = tl("subtitle")
    with rtl_container():
        st.markdown(f"""
        <div class="header-container">
            <h1 class="university-name">🎓 {title_text}</h1>
            <p class="subtitle">{subtitle_text}</p>
        </div>
        <div class="uhd-watermark">UHD</div>
        """, unsafe_allow_html=True)

# ================== FILE PATHS ==================
FAQ_PATH = SCRIPT_DIR / "faq.csv"
SCHED_PATH = SCRIPT_DIR / "schedule.csv"

# Feedback storage at project root
FEEDBACK_PATH = SCRIPT_DIR / "report_and_feedback.csv"
# Ensure feedback file exists with headers
if not FEEDBACK_PATH.exists():
    PD_COLUMNS = ["timestamp", "name", "contact", "topic", "message"]
    pd.DataFrame(columns=PD_COLUMNS).to_csv(
        FEEDBACK_PATH, index=False, encoding="utf-8")

ADMIN_SETTINGS = {}
if hasattr(st, "secrets"):
    try:
        admin_section = st.secrets.get("admin", {})
        if isinstance(admin_section, Mapping):
            ADMIN_SETTINGS = dict(admin_section)
    except Exception:
        ADMIN_SETTINGS = {}

FEEDBACK_DEFAULT_RECIPIENTS = ADMIN_SETTINGS.get(
    "feedback_email", "dyarabdula15@gmail.com")

# ================== DEFAULT DATA ==================
DEFAULT_FAQ = pd.DataFrame([
    # General University Questions
    ["Where is the main library?",
     "The Main library under building  (C). Open 8:30–18:00 Sat–Thu.",
     "library;location;books;study"],
    ["How do I get my student ID?",
     "Go to the Registrar Office with your admission letter + one photo. IDs issued 9:00–14:00.",
     "registrar;id;student card"],
    ["What is the Wi-Fi network?",
     "Free WiFi without password is available in the cafeteria under the G building",
     "wifi;internet;it;network;connection"],
    ["Where is the cafeteria?",
     "We have two cafeterias, the first is a coffee shop to the left of the entrance and the second is on the ground floor of the building (C) . Breakfast 8:00–10:30, lunch 12:00–15:00.",
     "food;cafeteria;dining;meals;restaurant;eat"],
    ["What are the festival timings?",
     "The University Festival runs for three days, 10:00–16:00 daily at the main courtyard.",
     "festival;events;celebration;activities"],
    ["Where can I print or photocopy?",
     "It is next to Building (A).",
     "printing;photocopy;services;printer;copy;documents"],
    ["How do I register for courses?",
     "Go to uhd website login and enter Email and password You can Register For courses. Registration opens one week before each semester.",
     "registration;courses;enrollment;classes;register;enroll"],
    ["Where is the parking lot?",
     "Student parking is available behind Building C. Parking permits required from Security Office.",
     "parking;car;vehicle;transportation;drive"],

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
                st.warning(tl("faq_csv_missing"))
                return DEFAULT_FAQ, "built-in"
            if "tags" not in df.columns:
                df["tags"] = ""
            return df, "local file"
        except Exception as e:
            st.warning(tl("faq_csv_read_error", error=e))
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
                st.warning(tl("schedule_csv_missing"))
                return DEFAULT_SCHEDULE, "built-in"
            if "department" not in df.columns:
                df["department"] = "General"
            df["day"] = df["day"].astype(str).str.title()
            return df, "local file"
        except Exception as e:
            st.warning(tl("schedule_csv_read_error", error=e))
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
    # Time chip with RTL support for Kurdish/Arabic - shows only time
    is_rtl = is_rtl_language()
    time_label = tl("current_time_label")  # Empty string, but using translation system
    components.html(
        f"""
        <style>
        .time-badge {{
            background: rgba(245, 222, 179, 0.6) !important;
            backdrop-filter: blur(15px) !important;
            padding: 0.8rem 1.8rem !important;
            border-radius: 35px !important;
            display: inline-block !important;
            box-shadow: 0 8px 25px rgba(245, 222, 179, 0.4) !important;
            font-size: 1rem !important;
            font-weight: 800 !important;
            color: #1a1a1a !important;
            border: 2px solid rgba(245, 222, 179, 0.7) !important;
            text-shadow: 0 2px 6px rgba(0, 0, 0, 0.5) !important;
            transition: all 0.3s ease !important;
            direction: {'rtl' if is_rtl else 'ltr'} !important;
            text-align: {'right' if is_rtl else 'left'} !important;
        }}
        
        .time-badge:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 35px rgba(245, 222, 179, 0.6) !important;
            background: rgba(245, 222, 179, 0.7) !important;
        }}
        
        .time-badge::before {{
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: -100% !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent) !important;
            transition: left 0.5s !important;
        }}
        
        .time-badge:hover::before {{
            left: 100% !important;
        }}
        </style>
        <span class="time-badge" id="time-badge"><span id="time-display">Loading...</span></span>
        
        <script>
        function updateTime() {{
            const now = new Date();
            const timeStr = now.toLocaleTimeString('en-US', {{
                hour: '2-digit', 
                minute: '2-digit', 
                hour12: true,
                timeZone: 'Asia/Baghdad'
            }});
            const dayStr = now.toLocaleDateString('en-US', {{
                weekday: 'long',
                timeZone: 'Asia/Baghdad'
            }});
            const timeDisplay = document.getElementById('time-display');
            if (timeDisplay) {{
                timeDisplay.textContent = '🕒 ' + timeStr + ' ' + dayStr;
            }}
        }}
        
        updateTime();
        setInterval(updateTime, 60000);
        </script>
        """,
        height=60
    )

st.markdown("<br>", unsafe_allow_html=True)

# ================== FAQ SEARCH ENGINE ==================

# Keyword translation mapping for Kurdish/Arabic to English
KEYWORD_TRANSLATIONS = {
    # Kurdish keywords
    "کتابخانه": "library",
    "کتێبخانە": "library",
    "کتێبخانه": "library",
    "سه‌ره": "main",
    "سەرەکی": "main",
    "سه‌رەکی": "main",
    "کولیه": "college",
    "کۆلێژ": "college",
    "کۆلێج": "college",
    "له": "in",
    "لە": "in",
    "کوێ": "where",
    "کوێیە": "where",
    "کوێیه": "where",
    "توومارکردن": "registration",
    "تۆمارکردن": "registration",
    "وایفای": "wifi",
    "وای-فای": "wifi",
    "چاپکردن": "printing",
    "چاپ": "printing",
    "هۆڵ": "hall",
    "هال": "hall",
    "لاب": "lab",
    "لابۆراتۆری": "lab",
    "قوتابخانە": "university",
    "زانکۆ": "university",
    "قوتابی": "student",
    "پارکینگ": "parking",
    "کافێ": "cafeteria",
    "کافێتێریا": "cafeteria",
    "بەش": "department",
    "وانە": "course",
    "دەرس": "course",
    "مامۆستا": "teacher",
    "ماموستا": "teacher",
    "پروگرام": "program",
    "بەرنامە": "program",
    
    # Arabic keywords
    "مكتبة": "library",
    "المكتبة": "library",
    "تسجيل": "registration",
    "التسجيل": "registration",
    "واي فاي": "wifi",
    "الواي فاي": "wifi",
    "طباعة": "printing",
    "الطباعة": "printing",
    "قاعة": "hall",
    "القاعة": "hall",
    "مختبر": "lab",
    "المختبر": "lab",
    "جامعة": "university",
    "الجامعة": "university",
    "طالب": "student",
    "الطالب": "student",
    "موقف": "parking",
    "مطعم": "cafeteria",
    "المطعم": "cafeteria",
    "قسم": "department",
    "القسم": "department",
    "مقرر": "course",
    "المقرر": "course",
    "أستاذ": "teacher",
    "الأستاذ": "teacher",
    "برنامج": "program",
    "البرنامج": "program",
    "أين": "where",
    "متى": "when",
    "ما": "what",
    "كيف": "how",
    "من": "who",
}

def translate_query_keywords(query: str) -> str:
    """Translate Kurdish/Arabic keywords in query to English for better matching"""
    query_lower = query.lower()
    translated_parts = []
    
    # Check for full word matches (for compound words and phrases)
    for kur_ar_word, eng_word in KEYWORD_TRANSLATIONS.items():
        if kur_ar_word.lower() in query_lower:
            translated_parts.append(eng_word)
    
    # Also tokenize and check individual tokens
    import re
    tokens = re.findall(r'\b\w+\b', query_lower, re.UNICODE)
    for token in tokens:
        if token in KEYWORD_TRANSLATIONS:
            if KEYWORD_TRANSLATIONS[token] not in translated_parts:
                translated_parts.append(KEYWORD_TRANSLATIONS[token])
    
    # Combine original query with translations
    if translated_parts:
        enhanced_query = query + " " + " ".join(translated_parts)
        return enhanced_query
    return query


def unicode_tokenize(text: str):
    """Tokenize text supporting Unicode characters (Kurdish, Arabic, English)"""
    # Use Unicode word boundaries to handle all languages
    # re.UNICODE is default in Python 3, but keeping for clarity
    tokens = re.findall(r'\b\w+\b', text.lower(), re.UNICODE)
    return tokens

@st.cache_resource(show_spinner=False)
def build_faq_index(df: pd.DataFrame):
    """Build TF-IDF index for FAQ search"""
    text = (df["question"].astype(str) + " " +
            df["tags"].fillna("").astype(str)).str.lower()
    # Configure TfidfVectorizer to handle Unicode with custom tokenizer
    vec = TfidfVectorizer(
        tokenizer=unicode_tokenize,
        ngram_range=(1, 2), 
        min_df=1, 
        max_features=1000,
        token_pattern=None  # Use custom tokenizer instead
    )
    X = vec.fit_transform(text)
    return vec, X


faq_vec, faq_X = build_faq_index(faq_df)


def _tokenize(s: str):
    """Extract tokens from string - supports Unicode (Kurdish, Arabic, English)"""
    # Use Unicode word boundaries to handle all languages
    tokens = re.findall(r'\b\w+\b', s.lower(), re.UNICODE)
    return set(tokens)


def faq_search(query: str, df: pd.DataFrame, vec, X):
    """Search FAQ using hybrid TF-IDF and tag matching with improved accuracy"""
    # Translate Kurdish/Arabic keywords to English for better matching
    enhanced_query = translate_query_keywords(query)
    q = enhanced_query.lower().strip()
    if not q:
        return None, 0.0

    # TF-IDF similarity
    tfidf = cosine_similarity(vec.transform([q]), X).flatten()

    # Tag overlap score
    qtok = _tokenize(q)
    # Tokenize tags with Unicode support
    tag_sets = df["tags"].fillna("").str.lower().apply(
        lambda t: set(_tokenize(str(t))))
    overlap_counts = tag_sets.apply(lambda s: len(qtok & s)).astype(float)
    if overlap_counts.max() > 0:
        overlap = overlap_counts / overlap_counts.max()
    else:
        overlap = overlap_counts

    # Direct keyword matching for better accuracy (Unicode-aware)
    keyword_scores = []
    query_tokens = _tokenize(q)
    
    for idx, row in df.iterrows():
        question_lower = row["question"].lower()
        answer_lower = row["answer"].lower()
        tags_lower = str(row["tags"]).lower()
        
        # Tokenize question, answer, and tags for better matching
        question_tokens = _tokenize(question_lower)
        answer_tokens = _tokenize(answer_lower)
        tags_tokens = _tokenize(tags_lower)

        # Check for token matches (works with Unicode)
        score = 0
        for token in query_tokens:
            if token in question_tokens:
                score += 3  # Higher weight for question matches
            if token in answer_tokens:
                score += 2  # Medium weight for answer matches
            if token in tags_tokens:
                score += 1  # Lower weight for tag matches
            # Also check substring matches for better cross-language matching
            if len(token) >= 3:  # Only for meaningful tokens
                if token in question_lower:
                    score += 1.5
                if token in answer_lower:
                    score += 1

        keyword_scores.append(score)

    keyword_scores = np.array(keyword_scores)
    if keyword_scores.max() > 0:
        keyword_scores = keyword_scores / keyword_scores.max()
    else:
        keyword_scores = np.zeros_like(keyword_scores)

    # Adjust weights based on TF-IDF performance
    # If TF-IDF scores are very low (likely cross-language query), favor keyword matching
    max_tfidf = tfidf.max()
    if max_tfidf < 0.1:  # Very low TF-IDF suggests language mismatch
        # Give more weight to keyword matching for cross-language queries
        combined = 0.2 * tfidf + 0.2 * overlap.values + 0.6 * keyword_scores
    else:
        # Normal weights for same-language queries
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
tab_labels = [
    f"❓ {tl('faq')}",
    f"📅 {tl('class_schedule')}",
    f"📋 {tl('full_timetable')}",
    f"🏛️ {tl('department')}",
    f"ℹ️ {tl('about')}",
    f"💡 {tl('feedback')}",
]
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_labels)

# ========== FAQ TAB ==========
with tab1:
    with rtl_container():
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown(f"### 📚 {tl('ask_question')}")
        st.write(f"*{tl('faq_intro')}*")

        # Personalization
        st.markdown(f"**{get_greeting()} 👋**")
        st.markdown("<br>", unsafe_allow_html=True)

        # Initialize session state keys
        if "pending_question" not in st.session_state:
            st.session_state["pending_question"] = ""
        
        # Check if there's a pending question from sample button click
        # Set user_input before widget creation to avoid Streamlit error
        if st.session_state["pending_question"]:
            st.session_state["user_input"] = st.session_state["pending_question"]
            st.session_state["pending_question"] = ""
        elif "user_input" not in st.session_state:
            st.session_state["user_input"] = ""

        q = st.text_input(
            tl("type_here"),
            key="user_input",
            placeholder=tl("search_placeholder")
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            ask_btn = st.button(f"🔍 {tl('ask_button')}",
                                key="faq_btn", use_container_width=True)

        if ask_btn:
            if not q.strip():
                st.info(f"ℹ️ {tl('enter_question_prompt')}")
            else:
                try:
                    with st.spinner(tl("thinking_spinner")):
                        idx, score = faq_search(q, faq_df, faq_vec, faq_X)
                    
                    # Lower threshold for non-English queries (Kurdish/Arabic)
                    # Check if query contains non-ASCII characters
                    has_non_ascii = any(ord(char) > 127 for char in q)
                    threshold = 0.15 if has_non_ascii else 0.25
                    
                    if score > threshold:
                        st.success(f"✅ {tl('match_found', score=score)}")
                        st.markdown(f"**{tl('answer_label')}**")
                        st.markdown(f"""
                        <div class="answer-box">
                            {faq_df.loc[idx, "answer"]}
                            <div style="margin-top: 1rem; font-size: 0.8rem; color: #ccc; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 0.5rem;">
                                📚 {tl("faq_answer_source")}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ {tl('no_match_warning')}")
                except Exception as exc:
                    st.error(tl("error_response"))
                    st.caption(tl("error_details", error=exc))

        with st.expander(tl("sample_questions")):
            sample_questions = tl("faq_samples") or []
            st.caption(tl("sample_questions_helper"))
            if not isinstance(sample_questions, list):
                sample_questions = []
            for idx, question in enumerate(sample_questions):
                if st.button(question, key=f"sample_{current_language}_{idx}"):
                    st.session_state["pending_question"] = question
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ========== SCHEDULE TAB ==========
with tab2:
    with rtl_container():
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown(f"### 🔍 {tl('schedule_heading')}")
        st.write(f"*{tl('schedule_description')}*")
        st.markdown("<br>", unsafe_allow_html=True)

        # Department filter
        depts = sorted([str(d).strip() for d in sched_df["department"].dropna().unique()
                        if str(d).strip() and str(d).lower() != "nan"])
        dept_choice = st.selectbox(
            tl("filter_by_department"),
            [tl("all_departments")] + depts,
            key="dept_select"
        )

        active_df = sched_df if dept_choice == tl("all_departments") else sched_df[
            sched_df["department"] == dept_choice]

        qs = st.text_input(
            tl("ask_about_class"),
            key="sched_q",
            placeholder=tl("class_search_placeholder")
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            ask_sched_btn = st.button(
                f"🔍 {tl('search_button')}", key="sched_btn", use_container_width=True)

        if ask_sched_btn:
            if not qs.strip():
                st.info(f"ℹ️ {tl('enter_course_prompt')}")
            else:
                with st.spinner(tl("search_schedule_spinner")):
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
                            f"✅ {tl('course_match_success', count=len(unique_courses))}")
                        st.markdown("---")

                        # Display each unique course
                        for _, course in unique_courses.iterrows():
                            st.markdown(
                                f"### {course['course_code']} – {course['course_name']}")
                            st.markdown(
                                f"**📚 {tl('course_department_label')}** {course['department']}")

                            # Get all sessions for this course
                            course_sessions = matches[matches['course_code']
                                                      == course['course_code']].sort_values('day')

                            st.markdown(f"**📅 {tl('course_schedule_label')}**")
                            for _, session in course_sessions.iterrows():
                                st.markdown(
                                    f"- **{session['day']}**: ⏰ {session['start_time']} - {session['end_time']} | 🏢 {session['hall']} | 👨‍🏫 {session['lecturer']}")

                            st.markdown("---")
                    else:
                        st.info(tl("no_classes_found"))
        st.markdown('</div>', unsafe_allow_html=True)

# ========== TIMETABLE TAB ==========
with tab3:
    if is_rtl_language():
        st.markdown("<div dir='rtl' style='text-align: right;'>", unsafe_allow_html=True)
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown(f"### 📋 {tl('timetable_heading')}")
    st.write(f"*{tl('timetable_description')}*")
    st.markdown("<br>", unsafe_allow_html=True)

    # Filter options
    col1, col2, col3 = st.columns([1.2, 1, 0.8])

    with col1:
        # Department filter
        all_depts = [tl("all_departments")] + sorted([
            str(d).strip() for d in sched_df["department"].dropna().unique()
            if str(d).strip() and str(d).lower() != "nan"
        ])
        dept_filter = st.selectbox(tl("department_label"), all_depts, key="tt_dept")

    with col2:
        # Day filter
        unique_days = list(set(sched_df["day"].dropna().tolist()))
        day_order = ["Saturday", "Sunday", "Monday",
                     "Tuesday", "Wednesday", "Thursday", "Friday"]
        sorted_days = [day for day in day_order if day in unique_days]
        day_options = [tl("all_days")] + sorted_days
        day_filter = st.selectbox(tl("day_label"), options=day_options, key="tt_day")

    with col3:
        # Sort option
        sort_option = st.selectbox(
            tl("sort_by"),
            [tl("sort_time"), tl("sort_course_code"), tl("sort_department")],
            key="tt_sort"
        )

    # Apply filters
    filtered_df = sched_df.copy()

    if dept_filter != tl("all_departments"):
        filtered_df = filtered_df[filtered_df["department"] == dept_filter]

    if day_filter != tl("all_days"):
        filtered_df = filtered_df[filtered_df["day"] == day_filter]

    # Sort
    if sort_option == tl("sort_time"):
        filtered_df = filtered_df.sort_values(["day", "start_time"])
    elif sort_option == tl("sort_course_code"):
        filtered_df = filtered_df.sort_values("course_code")
    else:
        filtered_df = filtered_df.sort_values(["department", "start_time"])

    # Display count
    st.markdown(f"**{tl('showing_classes', count=len(filtered_df))}**")
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
        st.markdown(f"### 📥 {tl('export_timetable')}")
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label=f"📥 {tl('download_csv')}",
            data=csv,
            file_name=f"uhd_timetable_{day_filter.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )

        # Display as table option
        with st.expander(f"📊 {tl('view_as_table')}"):
            st.dataframe(
                filtered_df[["course_code", "course_name", "day",
                             "start_time", "end_time", "hall", "lecturer", "department"]],
                hide_index=True
            )
    else:
        st.info(tl("no_classes_filters"))
    st.markdown('</div>', unsafe_allow_html=True)
    if is_rtl_language():
        st.markdown("</div>", unsafe_allow_html=True)

# ========== DEPARTMENT TAB ==========
with tab4:
    if is_rtl_language():
        st.markdown("<div dir='rtl' style='text-align: right;'>", unsafe_allow_html=True)
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown(f"### 🏛️ {tl('departments_section_heading')}")
    st.write(f"*{tl('departments_description')}*")
    st.markdown("<br>", unsafe_allow_html=True)

    # Department list in a simpler format
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(tl("departments_list_left"))

    with col2:
        st.markdown(tl("departments_list_right"))
    st.markdown('</div>', unsafe_allow_html=True)
    if is_rtl_language():
        st.markdown("</div>", unsafe_allow_html=True)

# ========== ABOUT TAB ==========
with tab5:
    if is_rtl_language():
        st.markdown("<div dir='rtl' style='text-align: right;'>", unsafe_allow_html=True)
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown(f"### ℹ️ {tl('about_heading')}")

    st.markdown(
        tl("about_intro_html").format(about_description=tl("about_description")),
        unsafe_allow_html=True
    )

    st.markdown(f"### 👥 {tl('development_team_heading')}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="developer-card" style="background: rgba(255, 255, 255, 0.08); padding: 1.5rem; border-radius: 12px; 
                    text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.3); border: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="font-size: 3rem;">👨‍💻</div>
            <h4 style="color: #ffffff; margin: 0.5rem 0; line-height: 1.2;">Dyar Abdulla</h4>
            <p style="color: #e0e0e0; margin: 0;">{role}</p>
        </div>
        """.format(role=tl("developer_role")), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="developer-card" style="background: rgba(255, 255, 255, 0.08); padding: 1.5rem; border-radius: 12px; 
                    text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.3); border: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="font-size: 3rem;">👨‍💻</div>
            <h4 style="color: #ffffff; margin: 0.5rem 0; line-height: 1.2;">Anas Sarkawt</h4>
            <p style="color: #e0e0e0; margin: 0;">{role}</p>
        </div>
        """.format(role=tl("developer_role")), unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="developer-card" style="background: rgba(255, 255, 255, 0.08); padding: 1.5rem; border-radius: 12px; 
                    text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.3); border: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="font-size: 3rem;">👨‍💻</div>
            <h4 style="color: #ffffff; margin: 0.5rem 0; line-height: 1.2;">Drood Muhammed</h4>
            <p style="color: #e0e0e0; margin: 0;">{role}</p>
        </div>
        """.format(role=tl("developer_role")), unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(f"### ✨ {tl('features_heading')}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(tl("features_list_left"))

    with col2:
        st.markdown(tl("features_list_right"))

    st.markdown("---")

    st.markdown(f"### 📊 {tl('data_sources_heading')}")
    st.markdown(
        tl(
            "data_sources_list",
            faq_src=faq_src,
            faq_count=len(faq_df),
            sched_src=sched_src,
            sched_count=len(sched_df),
        )
    )

    st.markdown("---")

    st.markdown(f"### 💡 {tl('how_to_heading')}")
    st.markdown(tl("how_to_list"))

    st.markdown('</div>', unsafe_allow_html=True)

# ========== FEEDBACK TAB ==========
with tab6:
    if is_rtl_language():
        st.markdown("<div dir='rtl' style='text-align: right;'>", unsafe_allow_html=True)
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown(f"### 💡 {tl('feedback_heading')}")
    st.write(f"*{tl('feedback_subheading')}*")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("feedback_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(tl("name_optional"), max_chars=60)
        with col2:
            contact = st.text_input(tl("contact_optional"), max_chars=60)

        topic = st.selectbox(
            tl("feedback_topic_label"),
            tl("feedback_topic_options")
        )
        message = st.text_area(
            tl("message"),
            height=180,
            placeholder=tl("feedback_message_placeholder")
        )
        agree = st.checkbox(
            tl("feedback_checkbox"))

        submitted = st.form_submit_button(
            f"📨 {tl('feedback_submit_button')}", use_container_width=True)

        if submitted:
            if not message.strip():
                st.warning(
                    f"⚠️ {tl('feedback_missing_message')}")
            elif not agree:
                st.info(f"ℹ️ {tl('feedback_missing_consent')}")
            else:
                entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "name": name.strip(),
                    "contact": contact.strip(),
                    "topic": topic,
                    "message": message.strip()
                }
                try:
                    entry_df = pd.DataFrame([entry])
                    header_needed = True
                    if FEEDBACK_PATH.exists():
                        try:
                            header_needed = FEEDBACK_PATH.stat().st_size == 0
                        except OSError:
                            header_needed = True

                    entry_df.to_csv(
                        FEEDBACK_PATH,
                        mode="a",
                        index=False,
                        header=header_needed,
                        encoding="utf-8"
                    )
                    email_sent = False
                    email_error = None
                    email_status = try_send_feedback_email(entry)
                    if isinstance(email_status, tuple):
                        email_sent, email_error = email_status
                    elif isinstance(email_status, bool):
                        email_sent = email_status

                    st.success(tl("feedback_saved"))
                    if email_sent:
                        st.caption(
                            tl("feedback_saved_email"))
                    else:
                        st.caption(tl("feedback_saved_no_email"))
                        if email_error:
                            st.caption(
                                tl("feedback_email_skipped", error=email_error))
                except Exception as exc:
                    st.error(
                        f"🚫 {tl('feedback_error')}")
                    st.caption(tl("error_details", error=exc))

    st.markdown('</div>', unsafe_allow_html=True)
    if is_rtl_language():
        st.markdown("</div>", unsafe_allow_html=True)

