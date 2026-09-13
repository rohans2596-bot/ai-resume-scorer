# AI-Powered Resume Scorer
**Author:** ROHAN S ([rohans2596@gmail.com](mailto:rohans2596@gmail.com)) &bull; Chennai, India

An NLP-powered text processing engine and interactive web application that scores candidate resumes against target job descriptions by extracting required skills, computing TF-IDF cosine semantic similarity, and identifying actionable skill gaps.

## 🚀 Key Engineering Specifications
- **NLP Text Tokenization:** Custom stop-word filtering, regex sanitization, and technical vocabulary n-gram extraction.
- **Taxonomy Skill Matching:** Rule-based and semantic extraction across languages (Python, Java, C, JavaScript), AI/ML tools (Neural Networks, NumPy, NLP), and developer tooling.
- **Cosine Semantic Similarity:** Vector space comparison measuring linguistic and contextual overlap between job requirements and candidate profiles.
- **Actionable Gap Analysis:** Immediate diagnostic categorization of matched qualifications vs missing prerequisites.

## 🛠 File Structure
- `scorer.py` &mdash; Python text-processing, vector similarity, and skill extraction module.
- `index.html` &mdash; Interactive real-time browser tester with pre-loaded candidate benchmarks and scoring dashboard.
- `package.json` &mdash; Project metadata.

## 💻 Run & Verify
```bash
python scorer.py
```
Or test the interactive browser interface live at `/apps/resume-scorer/`.
