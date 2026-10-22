import streamlit as st
from google import genai
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillBridge", page_icon="🚀", layout="wide")

st.title("🚀 SkillBridge")
st.subheader("AI-Powered Career Intelligence")

skills_df = pd.read_csv("skills.csv")
careers_df = pd.read_csv("careers.csv")
opportunities_df = pd.read_csv("opportunities.csv")
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
                client = genai.Client(
                    api_key=st.secrets["GEMINI_API_KEY"]
                )

                prompt = f"""
                You are an expert career advisor for computer science students.

                Target career: {target_career}
                Student skills: {", ".join(sorted(student))}
                Missing skills: {top["Missing Skills"]}
                Career readiness: {readiness:.1f}%

                Provide:
                1. A short assessment of the student's suitability.
                2. Three important skills to learn next.
                3. Two practical project ideas.
                4. A simple 30-day learning plan.

                Be realistic, specific, and easy to understand.
                Do not guarantee job or internship eligibility.
                """

                response = client.models.generate_content(
                      model="gemini-3.8-flash",
                      contents=prompt
                )    
                    
            
                    
           
    


             st.markdown(response.text)

            except Exception as e:
                 st.error(f"Error: {e}")
                
            

    else:
        st.warning("Please enter at least one skill.")
