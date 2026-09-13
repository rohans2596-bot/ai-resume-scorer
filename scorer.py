"""
AI-Powered Resume Scorer
Author: ROHAN S (rohans2596@gmail.com)
Extracts key technical skills, computes TF-IDF semantic cosine similarity,
and generates actionable relevance scores and skill gap reports against Job Descriptions.
"""

import re
import math
from collections import Counter

class ResumeScorer:
    # Industry Skill Taxonomy Dictionary
    KNOWN_SKILLS = {
        "python", "java", "c", "c++", "javascript", "typescript", "html", "css",
        "react", "next.js", "node.js", "fastapi", "express", "sql", "postgresql",
        "mongodb", "redis", "docker", "kubernetes", "git", "github", "aws", "gcp",
        "neural networks", "numpy", "pandas", "scikit-learn", "nlp", "pytorch",
        "machine learning", "deep learning", "prompt engineering", "data structures",
        "algorithms", "webrtc", "rest api", "graphql", "ci/cd"
    }

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
        "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
        "will", "with", "we", "you", "i", "our", "your", "my", "this", "or", "an"
    }

    def tokenize(self, text):
        clean = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text.lower())
        tokens = [t.strip() for t in clean.split() if len(t.strip()) > 1 and t.strip() not in self.STOP_WORDS]
        return tokens

    def extract_skills(self, text):
        lower = text.lower()
        found = set()
        for skill in self.KNOWN_SKILLS:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, lower):
                found.add(skill)
        return sorted(list(found))

    def compute_cosine_similarity(self, tokens1, tokens2):
        counter1 = Counter(tokens1)
        counter2 = Counter(tokens2)
        all_words = set(counter1.keys()).union(set(counter2.keys()))

        dot_product = sum(counter1[w] * counter2[w] for w in all_words)
        mag1 = math.sqrt(sum(val ** 2 for val in counter1.values()))
        mag2 = math.sqrt(sum(val ** 2 for val in counter2.values()))

        if not mag1 or not mag2:
            return 0.0
        return dot_product / (mag1 * mag2)

    def analyze(self, resume_text, job_description):
        resume_tokens = self.tokenize(resume_text)
        jd_tokens = self.tokenize(job_description)

        resume_skills = set(self.extract_skills(resume_text))
        jd_skills = set(self.extract_skills(job_description))

        matching_skills = sorted(list(resume_skills.intersection(jd_skills)))
        missing_skills = sorted(list(jd_skills.difference(resume_skills)))

        cosine_sim = self.compute_cosine_similarity(resume_tokens, jd_tokens)

        # Composite Score: 60% Skill Match + 40% Text Cosine Similarity
        skill_coverage = len(matching_skills) / max(1, len(jd_skills))
        composite_score = round((skill_coverage * 0.6 + cosine_sim * 0.4) * 100, 1)
        composite_score = min(100.0, max(0.0, composite_score))

        return {
            "score": composite_score,
            "skill_match_percentage": round(skill_coverage * 100, 1),
            "semantic_similarity": round(cosine_sim * 100, 1),
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "resume_skill_count": len(resume_skills),
            "jd_required_skill_count": len(jd_skills)
        }


if __name__ == "__main__":
    scorer = ResumeScorer()

    sample_resume = """
    ROHAN S - Computer Engineering undergraduate.
    Skilled in Python, Java, C, HTML, CSS, JavaScript, full-stack development.
    Built neural networks & NumPy ML systems from first principles.
    Experienced in NLP basics, AI tooling, prompt engineering, Git, GitHub, data structures & algorithms.
    """

    sample_jd = """
    Software Engineering / AI-ML Intern.
    Looking for candidates skilled in Python, machine learning, neural networks, NumPy, Git, and data structures.
    Familiarity with full-stack development and NLP is a strong plus.
    """

    result = scorer.analyze(sample_resume, sample_jd)
    print("=== AI RESUME SCORING ANALYSIS ===")
    print(f"Overall Relevance Score: {result['score']}%")
    print(f"Skill Coverage: {result['skill_match_percentage']}%")
    print(f"Matching Skills ({len(result['matching_skills'])}): {', '.join(result['matching_skills'])}")
    print(f"Missing / Skill Gaps ({len(result['missing_skills'])}): {', '.join(result['missing_skills'])}")
