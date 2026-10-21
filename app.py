import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillBridge", page_icon="🚀", layout="wide")

st.title("🚀 SkillBridge")
st.subheader("AI-Powered Career Intelligence")

skills_df = pd.read_csv("skills.csv")
careers_df = pd.read_csv("careers.csv")
opportunities_df = pd.read_csv("opportunities.csv")
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
    "git hub": "github"
}

def normalize_skill(skill):
    skill = skill.strip().lower()
    return skill_aliases.get(skill, skill)

@st.cache_resource
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



if st.button("Analyze My Profile"):
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

        texts = opportunities_df["required_skills"].tolist()

        embeddings = model.encode([student_skills] + texts)

        semantic_scores = cosine_similarity(
            embeddings[:1], embeddings[1:]
        )[0]

        opportunity_results = []
        for i, (_, opportunity) in enumerate(opportunities_df.iterrows()):
            required = {
                normalize_skill(s)
                for s in opportunity["required_skills"].split(",")
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
                "Role": opportunity["role"],
                "Type": opportunity["type"],
                "Location": opportunity["location"],
                "Exact Score": round(exact_score, 2),
                "Semantic Score": round(semantic_score, 2),
                "Hybrid Score": round(hybrid_score, 2),
                "Matched Skills": ", ".join(sorted(matched)),
                "Missing Skills": ", ".join(sorted(missing))
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

        missing_target_skills = sorted(target_skills - student)

        if missing_target_skills:
            st.write("### Skills to Learn")

            for i, skill in enumerate(missing_target_skills, start=1):
                st.write(f"{i}. {skill.title()}")

        else:
            st.success("You already have all the skills listed for this career!")

        st.header("📚 Skill Gap Analysis")

        top = opportunity_results.iloc[0]

        st.write("**Recommended role:**", top["Role"])
        st.write(
            "**Missing skills:**",
            top["Missing Skills"] or "No missing skills detected."
        )
    else:
        st.warning("Please enter at least one skill.")

