import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillBridge", page_icon="🚀", layout="wide")

st.title("🚀 SkillBridge")
st.subheader("AI-Powered Career Intelligence")

careers_df = pd.read_csv("careers.csv")
opportunities_df = pd.read_csv("opportunities.csv")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

student_skills = st.text_area(
    "Enter your skills, separated by commas",
    "Python, SQL, Pandas, Machine Learning, Statistics"
)

if st.button("Analyze My Profile"):
    if student_skills.strip():
        student = {s.strip().lower() for s in student_skills.split(",") if s.strip()}

        career_results = []
        for _, career in careers_df.iterrows():
            required = {s.strip().lower() for s in career["required_skills"].split(",")}
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
                s.strip().lower()
                for s in opportunity["required_skills"].split(",")
            }
            matched = student & required
            missing = required - student
            exact_score = len(matched) / len(required) * 100 if required else 0
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

        st.header("📚 Skill Gap Analysis")
        top = opportunity_results.iloc[0]
        st.write("**Recommended role:**", top["Role"])
        st.write("**Missing skills:**", top["Missing Skills"] or "No missing skills detected.")

    else:
        st.warning("Please enter at least one skill.")
