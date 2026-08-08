import streamlit as st
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="College FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# NLTK SETUP
# ============================================================

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ============================================================
# FAQ DATASET
# ============================================================

faq_data = [
    {
        "question": "What are the admission requirements?",
        "answer": "Admission requirements generally include meeting the required academic eligibility criteria and submitting the necessary documents."
    },
    {
        "question": "How can I apply for admission?",
        "answer": "You can apply by completing the college admission application form and submitting the required documents before the deadline."
    },
    {
        "question": "What documents are required for admission?",
        "answer": "Common documents include academic certificates, identity proof, photographs, address proof, and other documents specified by the college."
    },
    {
        "question": "What courses are available?",
        "answer": "The college may offer undergraduate and postgraduate programs across engineering, technology, management, science, and other academic fields."
    },
    {
        "question": "What is the admission process?",
        "answer": "The admission process generally includes application, document verification, eligibility checking, merit or entrance-based selection, and fee payment."
    },
    {
        "question": "Is there an entrance exam?",
        "answer": "Some programs may require an entrance examination depending on the course and applicable admission rules."
    },
    {
        "question": "What is the eligibility criteria?",
        "answer": "Eligibility criteria depend on the selected program and usually include specific educational qualifications and minimum marks."
    },
    {
        "question": "Is there an application fee?",
        "answer": "An application fee may apply depending on the program and admission process."
    },
    {
        "question": "When does admission start?",
        "answer": "Admission dates vary each academic year. Students should check the official admission schedule for current dates."
    },
    {
        "question": "When is the last date to apply?",
        "answer": "The last date varies by program and admission cycle. Please check the official admission notice for the exact deadline."
    },
    {
        "question": "Can I apply online?",
        "answer": "Yes, if online applications are available, students can complete the application process through the designated admission portal."
    },
    {
        "question": "How can I check my admission status?",
        "answer": "Admission status can usually be checked through the official admission portal using your application or registration details."
    },
    {
        "question": "What is the fee structure?",
        "answer": "The fee structure depends on the selected course and academic year. Students should refer to the official fee structure for accurate information."
    },
    {
        "question": "Are scholarships available?",
        "answer": "Scholarships may be available based on academic performance, government schemes, category, financial need, or other eligibility criteria."
    },
    {
        "question": "How can I apply for a scholarship?",
        "answer": "Students can apply for eligible scholarships through the appropriate scholarship portal or college administration."
    },
    {
        "question": "Is hostel accommodation available?",
        "answer": "Hostel accommodation may be available for students depending on campus facilities and room availability."
    },
    {
        "question": "How can I apply for hostel?",
        "answer": "Students can apply for hostel accommodation through the college hostel administration after completing the admission process."
    },
    {
        "question": "Is there a transportation facility?",
        "answer": "Transportation facilities may be available on selected routes depending on the institution's transport services."
    },
    {
        "question": "Are there placement opportunities?",
        "answer": "Many colleges provide placement support through training, career guidance, recruitment drives, internships, and industry interactions."
    },
    {
        "question": "Does the college provide internships?",
        "answer": "Internship opportunities may be provided through industry collaborations, training programs, and placement or career development cells."
    },
    {
        "question": "What departments are available?",
        "answer": "Departments vary by institution and may include Computer Science, Artificial Intelligence and Machine Learning, Mechanical, Civil, Electrical, Electronics, and other disciplines."
    },
    {
        "question": "Is attendance compulsory?",
        "answer": "Students are generally expected to maintain the minimum attendance required by the institution and university regulations."
    },
    {
        "question": "What are the college timings?",
        "answer": "College timings vary by program, semester, and timetable. Students should refer to their official class timetable."
    },
    {
        "question": "How can I contact the college?",
        "answer": "You can contact the college through its official website, administration office, admission office, phone number, or official email address."
    },
    {
        "question": "Where is the college located?",
        "answer": "The college location can be found on its official website or official campus information page."
    }
]


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Clean and normalize text for NLP processing.
    """

    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove very common words
    stop_words = {
        "the",
        "is",
        "are",
        "a",
        "an",
        "to",
        "of",
        "for",
        "and",
        "in",
        "on",
        "can",
        "i",
        "what",
        "how",
        "when",
        "where",
        "do",
        "does"
    }

    tokens = [
        token for token in tokens
        if token not in stop_words
    ]

    return " ".join(tokens)


# ============================================================
# PREPARE FAQ QUESTIONS
# ============================================================

faq_questions = [
    preprocess_text(item["question"])
    for item in faq_data
]


# ============================================================
# TF-IDF MODEL
# ============================================================

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(faq_questions)


# ============================================================
# FIND BEST FAQ MATCH
# ============================================================

def find_best_answer(user_question):

    processed_question = preprocess_text(user_question)

    if not processed_question:
        return None, 0

    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )[0]

    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[best_match_index]

    answer = faq_data[best_match_index]["answer"]

    return answer, best_score


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #cccccc;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🤖 College FAQ Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about college admission, courses, fees, scholarships and facilities.</div>',
    unsafe_allow_html=True
)


# ============================================================
# INFORMATION
# ============================================================

st.info(
    "💡 Ask a question such as: "
    "\"What are the admission requirements?\""
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# USER INPUT
# ============================================================

user_question = st.chat_input(
    "Type your question here..."
)


# ============================================================
# CHATBOT RESPONSE
# ============================================================

if user_question:

    # Display user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)


    # Find answer
    answer, score = find_best_answer(user_question)


    # Minimum confidence threshold
    threshold = 0.20


    with st.chat_message("assistant"):

        if answer is not None and score >= threshold:

            st.markdown(answer)

            st.caption(
                f"Match confidence: {score:.2f}"
            )

        else:

            fallback_message = (
                "I'm sorry, I couldn't find a suitable answer "
                "to your question. Please try asking about "
                "admissions, courses, fees, scholarships, "
                "hostel, placements, or college facilities."
            )

            st.markdown(fallback_message)


    # Save chatbot response

    if answer is not None and score >= threshold:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    else:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": fallback_message
            }
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 FAQ Categories")

    st.write("You can ask about:")

    st.write("🎓 Admissions")
    st.write("📖 Courses")
    st.write("💰 Fees")
    st.write("🏆 Scholarships")
    st.write("🏠 Hostel")
    st.write("🚌 Transportation")
    st.write("💼 Placements")
    st.write("🧑‍💻 Internships")
    st.write("🏫 Departments")
    st.write("📞 Contact Information")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# ABOUT PROJECT
# ============================================================

with st.expander("ℹ️ About this Project"):

    st.write(
        """
        This FAQ Chatbot was developed as part of the
        CodeAlpha Artificial Intelligence Internship.

        The chatbot uses Natural Language Processing (NLP)
        to preprocess user questions and TF-IDF vectorization
        with cosine similarity to identify the most relevant
        FAQ question.

        Technologies used:

        • Python
        • Streamlit
        • NLTK
        • Scikit-learn
        • TF-IDF
        • Cosine Similarity
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="text-align:center; margin-top:40px;">
        CodeAlpha AI Internship | FAQ Chatbot
    </div>
    """,
    unsafe_allow_html=True
)