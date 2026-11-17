import streamlit as st
from google import genai
import pandas as pd
import requests
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
@st.cache_resource
def get_gemini_client():
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

client = get_gemini_client()
st.set_page_config(page_title="SkillBridge", page_icon="🚀", layout="wide")


st.markdown("""
<style>
.stApp {
    background-color: #F7F7FC;
}

[data-testid="stSidebar"] {
    background-color: #191629;
}

[data-testid="stSidebar"] * {
    color: #F4F0FF;
}

.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #211B35;
    font-weight: 700;
}
/* Hero heading must remain white */
.sb-hero h1 {
    color: #FFFFFF !important;
    font-size: 36px !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}

.sb-hero p {
    color: #E9DFFF !important;
    font-size: 16px !important;
    opacity: 1 !important;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #EAE5F5;
    padding: 20px;
    border-radius: 16px;
}

div[data-testid="stMetricValue"] {
    color: #7C3AED;
}

.stButton > button {
    background: #7C3AED;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

.stButton > button:hover {
    background: #6D28D9;
    color: white;
}

.sb-hero {
    background: linear-gradient(
        120deg, #21183D, #542B91, #7C3AED
    );
    padding: 35px;
    border-radius: 22px;
    margin-bottom: 25px;
}
div[data-testid="stMarkdownContainer"] .sb-hero h1 {
    color: white !important;
    -webkit-text-fill-color: white !important;
    font-size: 36px !important;
    font-weight: 800 !important;
}
.sb-hero h1 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}

.sb-hero p {
    color: #E9DFFF !important;
    opacity: 1 !important;
}
.sb-hero p {
    color: #E9DFFF;
    font-size: 16px;
}

/* Hero subtitle */
.sb-hero p {
    color: #E9DFFF !important;
}

/* Form labels */
[data-testid="stWidgetLabel"] p {
    color: #39334D !important;
    font-weight: 600 !important;
}

/* Text area */
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #29243A !important;
}

/* Dropdown */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #29243A !important;
}

/* Upload area */
[data-testid="stFileUploader"] section {
    background-color: #FFFFFF !important;
    border-radius: 12px;
}

/* Buttons */
.stButton > button {
    color: #FFFFFF !important;
}

/* Dashboard headings */
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5 {
    color: #21183D !important;
    font-weight: 750 !important;
}

/* Section titles and emphasized text */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] strong {
    color: #30204F !important;
}

/* Metric titles */
[data-testid="stMetricLabel"] {
    color: #514568 !important;
    font-weight: 600 !important;
}

/* Captions */
[data-testid="stMain"] .stCaption {
    color: #625873 !important;
}

/* Career match and skill analysis text */
[data-testid="stMain"] {
    color: #29243A;
}

[data-testid="stMain"] p,
[data-testid="stMain"] li,
[data-testid="stMain"] label {
    color: #39334D;
}

/* AI Career Advisor response */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li {
    color: #39334D !important;
}

/* Subheadings and result titles */
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5 {
    color: #21183D !important;
}

/* Expander content */
[data-testid="stExpander"] p,
[data-testid="stExpander"] li {
    color: #39334D !important;
}

/* Success and information messages */
[data-testid="stAlert"] p {
    color: #29243A !important;
}

/* Hero subtitle */
[data-testid="stMain"] .sb-hero p {
    color: #F3ECFF !important;
    -webkit-text-fill-color: #F3ECFF !important;
    opacity: 1 !important;
}

/* Career success message */
[data-testid="stAlert"] p {
    color: #29243A !important;
    -webkit-text-fill-color: #29243A !important;
    opacity: 1 !important;
    font-weight: 500 !important;
}

/* Improve career success message contrast */
[data-testid="stAlert"] {
    background-color: #EDE9FE !important;
    border: 1px solid #C4B5FD !important;
    border-radius: 12px !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] strong {
    color: #21183D !important;
    -webkit-text-fill-color: #21183D !important;
    opacity: 1 !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] strong {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# 🌉 SkillBridge")
    st.caption("CAREER INTELLIGENCE")
    st.divider()

    st.markdown("### Your Workspace")
    st.markdown("📊 Career Dashboard")
    st.markdown("💼 Explore Opportunities")
    st.markdown("🧠 Skill Intelligence")
    st.markdown("🗺️ Learning Roadmap")
    st.markdown("✨ AI Career Advisor")

    st.divider()
    st.caption("Powered by AI")

st.markdown("""
<div class="sb-hero">
    <h1>Build Your Future With Clarity.</h1>
    <p>
        Understand your skills, discover career opportunities,
        and create a personalized path towards your goals.
    </p>
</div>
""", unsafe_allow_html=True)


skills_df = pd.read_csv("skills.csv")
careers_df = pd.read_csv("careers.csv")
resources_df = pd.read_csv("learning_resources.csv")
skill_aliases = {
    "nlp": "natural language processing",
    "ml": "machine learning",
    "dl": "deep learning",
    "sklearn": "scikit-learn",
    "powerbi": "power bi",
    "js": "javascript",
    "cpp": "c++",
    "llm": "large language models",
    "llms": "large language models",
    "git hub": "github",
    "microsoft excel": "excel",
    "excel proficiency": "excel",
    "proficiency in microsoft excel": "excel",
    "data analysis and interpretation": "data analysis",
    "statistical knowledge and skills": "statistics",
    "statistical analysis": "statistics",
    "scikit learn": "scikit-learn"
}

def normalize_skill(skill):
    skill = skill.strip().lower()
    return skill_aliases.get(skill, skill)

def clean_job_skills(skills):
    skill_list = []

    known_skills = [
        "python", "java", "c++", "sql", "excel",
        "power bi", "tableau", "machine learning",
        "deep learning", "statistics", "data analysis",
        "pandas", "numpy", "scikit-learn", "tensorflow",
        "pytorch", "nlp", "natural language processing",
        "javascript", "html", "css", "git",
        "data structures", "algorithms",
        "database management", "communication",
        "problem solving", "analytical skills",
        "debugging", "attention to detail"
    ]

    for item in skills:
        item = item.lower().strip()

        found = False

        for skill in known_skills:
            if skill in item:
                skill_list.append(normalize_skill(skill))
                found = True

        if not found and len(item.split()) <= 3:
            skill_list.append(normalize_skill(item))

    return list(set(skill_list))

@st.cache_data(ttl=1800)
def fetch_live_jobs(search_query):
    base_url = "https://jobsforceindia.com/api/public/v1/jobs"

    response = requests.get(
     base_url,
     params={
        "q": search_query,
          "per_page": 10
     },
     timeout=15
    )

    response.raise_for_status()
    jobs = response.json().get("results", [])

    live_jobs = []

    for job in jobs:
        detail_url = f"{base_url}/{job['id']}"

        detail_response = requests.get(
            detail_url,
            timeout=15
        )

        if detail_response.status_code != 200:
            continue

        detail = detail_response.json()

        location = detail.get("location", {})
        skills = clean_job_skills(detail.get("skills", []))
        preferred = clean_job_skills(detail.get("skills_preferred", []))

        live_jobs.append({
            "Role": detail.get("title", ""),
            "Type": detail.get("job_type", ""),
            "Location": location.get("city") or "India",
            "Required Skills": ", ".join(skills),
            "Preferred Skills": ", ".join(preferred),
            "Salary": (detail.get("salary") or {}).get("label", "Not specified"),
            "Work Mode": detail.get("work_mode", "Not specified"),
            "Experience": (detail.get("experience") or {}).get("label", "Not specified"),
            "Summary": detail.get("summary", ""),
            "URL": detail.get("url", ""),
            "Apply URL": detail.get("apply_url", "")
        })

    return pd.DataFrame(live_jobs)

def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
uploaded_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

if uploaded_file is not None:
    from pypdf import PdfReader

    reader = PdfReader(uploaded_file)

    resume_text = ""

    for page in reader.pages:
        resume_text += page.extract_text() or ""

    skill_list = skills_df["skill_name"].tolist()

    found_skills = [
        skill for skill in skill_list
        if skill.lower() in resume_text.lower()
    ]

    st.write("### Skills Found in Your Resume")
    st.write(", ".join(found_skills) if found_skills else "No matching skills found.")

    if found_skills:
        default_skills = ", ".join(found_skills)
    else:
        default_skills = ""
else:
    default_skills = "Python, SQL, Pandas, Machine Learning, Statistics"
    
career_options = careers_df["career_role"].tolist()

target_career = st.selectbox(
    "Choose your target career",
    career_options
)
student_skills = st.text_area(
    "Enter your skills, separated by commas",
    value=default_skills
)



analyze_clicked = st.button("Analyze My Profile")

if analyze_clicked or st.session_state.get("show_results", False):
    st.session_state["show_results"] = True
    if student_skills.strip():
        student = {
            normalize_skill(s)
            for s in student_skills.split(",")
            if s.strip()
        }

        career_results = []

        for _, career in careers_df.iterrows():
            required = {
                normalize_skill(s)
                for s in career["required_skills"].split(",")
                if s.strip()
            }

            matched = student & required
            missing = required - student
            score = len(matched) / len(required) * 100 if required else 0

            career_results.append({
                "Career": career["career_role"],
                "Match Score": round(score, 2),
                "Matched Skills": ", ".join(sorted(matched)),
                "Missing Skills": ", ".join(sorted(missing))
            })

        career_results = pd.DataFrame(career_results).sort_values(
            "Match Score", ascending=False
        )
        
        st.header("📊 Career Readiness Dashboard")

        top_career = career_results.iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Best Career Match", top_career["Career"])
        col2.metric("Match Score", f'{top_career["Match Score"]}%')

        missing_count = len([
            s for s in top_career["Missing Skills"].split(",")
            if s.strip()
        ])

        col3.metric("Skills to Learn", missing_count)

        st.subheader("Your Top Career Matches")

        st.bar_chart(
            career_results.head(5).set_index("Career")[["Match Score"]]
        )


        
        live_jobs = fetch_live_jobs(target_career)

        if live_jobs.empty:
            st.warning("No live jobs found. Please try again later.")
            st.stop()

        texts = (
            live_jobs["Role"] + " " +
            live_jobs["Summary"] + " " +
            live_jobs["Required Skills"]
        ).tolist()

        embeddings = model.encode([student_skills] + texts)

        semantic_scores = cosine_similarity(
            embeddings[:1], embeddings[1:]
        )[0]

        opportunity_results = []

        for i, (_, opportunity) in enumerate(live_jobs.iterrows()):
            required = {
                normalize_skill(s)
                for s in opportunity["Required Skills"].split(",")
                if s.strip()
            }

            matched = student & required
            missing = required - student

            exact_score = (
                len(matched) / len(required) * 100 if required else 0
            )

            semantic_score = semantic_scores[i] * 100
            hybrid_score = 0.5 * exact_score + 0.5 * semantic_score

            opportunity_results.append({
                "Role": opportunity["Role"],
                "Type": opportunity["Type"],
                "Location": opportunity["Location"],
                "Exact Score": round(exact_score, 2),
                "Semantic Score": round(semantic_score, 2),
                "Hybrid Score": round(hybrid_score, 2),
                "Matched Skills": ", ".join(sorted(matched)),
                "Missing Skills": ", ".join(sorted(missing)),
                "Salary": opportunity["Salary"],
                "Work Mode": opportunity["Work Mode"],
                "Experience": opportunity["Experience"],
                "Apply URL": opportunity["Apply URL"]
            })

        opportunity_results = pd.DataFrame(opportunity_results).sort_values(
            "Hybrid Score", ascending=False
        )

        st.header("🎯 Top Career Recommendations")
        st.dataframe(career_results.head(5), use_container_width=True)

        st.header("💼 Top Opportunities")
        st.dataframe(opportunity_results.head(10), use_container_width=True)
        beginner_skills = {
            "python", "sql", "statistics", "excel",
            "data analysis", "git", "github",
            "numpy", "pandas"
         }

        intermediate_skills = {
             "machine learning", "scikit-learn",
             "feature engineering", "model evaluation",
             "predictive modeling", "matplotlib",
             "seaborn", "power bi", "tableau",
             "fastapi", "rest api"
         }
        
        st.header("🧭 Your Learning Roadmap")

        target_row = careers_df[
            careers_df["career_role"] == target_career
        ].iloc[0]

        target_skills = {
            normalize_skill(s)
            for s in target_row["required_skills"].split(",")
            if s.strip()
        }

        matched_target_skills = target_skills & student
        missing_target_skills = sorted(target_skills - student)

        readiness = (
            len(matched_target_skills) / len(target_skills) * 100
            if target_skills else 0
        )

        st.subheader(f"🎯 Readiness for {target_career}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Career Readiness", f"{readiness:.1f}%")
        col2.metric("Skills Matched", len(matched_target_skills))
        col3.metric("Skills Remaining", len(missing_target_skills))

        st.progress(int(readiness))

        st.subheader("📊 Skill Gap Overview")

        gap_data = pd.DataFrame({
            "Status": ["Already Have", "Need to Learn"],
            "Skills": [
                len(matched_target_skills),
                len(missing_target_skills)
            ]
        })

        st.bar_chart(gap_data.set_index("Status"))

        if missing_target_skills:
            st.subheader("🚀 Your Next 3 Skills to Learn")

            for skill in missing_target_skills[:3]:
                st.write(f"• {skill.title()}")

            stages = {
                "🌱 Beginner": [],
                "🛠️ Intermediate": [],
                "🚀 Advanced": []
            }

            for skill in missing_target_skills:
                if skill in beginner_skills:
                    stages["🌱 Beginner"].append(skill)
                elif skill in intermediate_skills:
                    stages["🛠️ Intermediate"].append(skill)
                else:
                    stages["🚀 Advanced"].append(skill)

            for stage, skills in stages.items():
                st.subheader(stage)

                if skills:
                    for skill in skills:
                        st.write(f"**{skill.title()}**")

                        resource = resources_df[
                            resources_df["skill"].str.strip().str.lower()
                            == skill.lower()
                        ]

                        if not resource.empty:
                            item = resource.iloc[0]
                            st.write(f"Resource: {item['resource_name']}")
                            st.write(f"Priority: {item['priority']}")
                            st.markdown(
                                f"[Start Learning]({item['resource_url']})"
                            )
                        else:
                            st.write("Learning resource coming soon.")
                else:
                     st.write("No skills in this stage.")

        else:
            st.success(
                "🎉 You already have all the skills listed for this career!"
            )


        st.header("📚 Skill Gap Analysis")

        top = opportunity_results.iloc[0]

        st.write("**Recommended role:**", top["Role"])
        st.write(
            "**Missing skills:**",
            top["Missing Skills"] or "No missing skills detected."
        )
        
        st.header("🤖 AI Career Advisor")

       
        
       
        if st.button("Generate AI Career Advice"):
            try:
                with st.spinner("Your AI Career Advisor is preparing recommendations..."):

                    prompt = f"""
                    You are an expert career advisor for computer science students.

                    Student information:
                    Target career: {target_career}
                    Current skills: {", ".join(sorted(student))}
                    Missing skills: {top["Missing Skills"]}
                    Career readiness: {readiness:.1f}%

                    Generate a complete career report.

                    IMPORTANT:
                    You MUST include all FOUR sections below.
                    Do not skip, combine, or shorten any section.
                    Follow the exact headings and order.

                    SECTION 1: SUITABILITY ASSESSMENT
                    Explain the student's current suitability for the target career.
                    Mention strengths and areas that need improvement.

                    SECTION 2: TOP 3 SKILLS TO LEARN
                    Recommend exactly three important missing skills.
                    Explain why each skill matters.

                    SECTION 3: TWO PRACTICAL PROJECT IDEAS
                    Suggest exactly two projects.
                    Explain what each project does and which skills it develops.

                    SECTION 4: PERSONALIZED 30-DAY LEARNING PLAN

                    Divide the plan into these four weeks:

                    Week 1: Learning the fundamentals.
                    Week 2: Practising the important skills.
                    Week 3: Building a practical project.
                    Week 4: Completing the project and preparing a portfolio.

                    Give at least two specific tasks for each week.
                    Make the tasks relevant to the student's missing skills
                    and target career.

                    End with a short encouraging recommendation.

                    Return all four sections in complete detail.
                    Do not omit the 30-day learning plan.
                    Do not guarantee job or internship eligibility.
                    """

                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompt,
                        config={
                            "max_output_tokens": 3000,
                            "temperature": 0.3
                        }
                    )

                
                if response.text:
                    st.markdown(response.text)

                    if response.candidates:
                        finish_reason = response.candidates[0].finish_reason

                        if str(finish_reason) == "MAX_TOKENS":
                            st.warning("The AI response was cut short. Please generate again.")
                else:
                    st.warning("Gemini returned an empty response.")


            except Exception as e:
                st.error(f"Error: {e}")



    
                    

    
                    

    

           
                
            

    else:
        st.warning("Please enter at least one skill.")
