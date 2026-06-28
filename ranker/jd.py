"""
JD intent model — a structured, auditable encoding of what the Senior AI Engineer
role at Redrob *means*, not just the words in the job description.

The job description is deliberately adversarial (see the "Final note for hackathon
participants" section). The right answer is *not* "most AI keywords". It is reasoning
about the gap between what the JD says and what it means:

  - A "Marketing Manager" with every AI keyword as a skill is NOT a fit.
  - A candidate who never writes "RAG" or "Pinecone" but built a recommendation
    system at a product company IS a fit.
  - A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5%
    recruiter response rate is, for hiring purposes, not actually available.

This module captures all of that as data so the scorer (scoring.py) can stay small
and readable, and so every weight is defensible in the Stage-5 interview.
"""

# ---------------------------------------------------------------------------
# The "ideal candidate" narrative — used as the query text for semantic matching.
# Phrased the way the JD describes the role *intent*, deliberately mixing
# vocabulary (RAG, retrieval, recommendation, ranking, search relevance) so that
# plain-language candidates who built the right systems still match.
# ---------------------------------------------------------------------------
JD_QUERY_TEXT = (
    "Senior AI / machine learning engineer who owns the intelligence layer of a product: "
    "the ranking, retrieval, search relevance and recommendation systems that decide what "
    "users see. Production experience with embeddings-based retrieval (sentence-transformers, "
    "BGE, E5, OpenAI embeddings) and vector / hybrid search infrastructure (FAISS, Pinecone, "
    "Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, BM25). Has shipped an end-to-end "
    "ranking, search or recommendation system to real users at meaningful scale at a product "
    "company, not pure research and not pure services. Strong Python and code quality. Designs "
    "evaluation frameworks for ranking systems: NDCG, MRR, MAP, offline-to-online correlation, "
    "A/B testing. Knows information retrieval and NLP, hybrid vs dense retrieval, when to "
    "fine-tune vs prompt an LLM. Scrappy product-engineering attitude, ships fast, writes well."
)

# ---------------------------------------------------------------------------
# Role taxonomy. Current title is the single strongest anti-keyword-stuffer signal,
# but career history can rescue a plain-language pivot, so we classify both.
# Matching is done on a lowercased title via substring / regex in features.py.
# ---------------------------------------------------------------------------

# Core target roles — the JD's bullseye.
CORE_AI_TITLES = [
    "machine learning engineer", "ml engineer", "ai engineer", "applied scientist",
    "applied ml", "research engineer", "ai research", "nlp engineer", "data scientist",
    "search engineer", "relevance engineer", "recommendation", "search relevance",
    "ml scientist", "ai/ml", "deep learning engineer", "computer vision engineer",
]

# Adjacent engineering roles — credible path into the role, scored a notch below core.
ADJACENT_ENG_TITLES = [
    "data engineer", "analytics engineer", "software engineer", "backend engineer",
    "backend developer", "full stack", "fullstack", "platform engineer", "swe",
    "senior software", "staff engineer", "principal engineer", "data analyst",
    "cloud engineer", "software developer", "python developer",
]

# Roles that the JD explicitly does NOT want, or that signal keyword-stuffer traps
# when paired with AI skills. A current title here with no engineering substance in
# the career history collapses the role-credibility gate.
OFF_TARGET_TITLES = [
    "hr manager", "human resource", "recruiter", "accountant", "accounting",
    "mechanical engineer", "civil engineer", "electrical engineer", "sales executive",
    "sales manager", "content writer", "copywriter", "graphic designer", "ux designer",
    "customer support", "customer success", "operations manager", "business analyst",
    "project manager", "marketing manager", "marketing", "product manager",
    "qa engineer", "quality assurance", "test engineer", "scrum master", "teacher",
    "consultant", "finance", "supply chain", "logistics",
]

# Pure-management / "haven't written code in 18 months" titles the JD warns against.
NON_CODING_LEADERSHIP_TITLES = [
    "engineering manager", "director", "vp ", "vice president", "head of",
    "cto", "chief", "architect",  # architect: JD explicitly flags "moved into architecture"
]

# ---------------------------------------------------------------------------
# Skill ontology. Weighted by how central each skill is to the JD's "absolutely
# need" list. Skill *names* are cheap to stuff, so scoring.py trust-weights them
# by endorsements, duration and Redrob assessment scores — but the ontology
# defines what's even relevant.
# ---------------------------------------------------------------------------
SKILL_WEIGHTS = {
    # Retrieval / vector search infrastructure (JD: "absolutely need")
    "faiss": 1.0, "pinecone": 1.0, "weaviate": 1.0, "qdrant": 1.0, "milvus": 1.0,
    "elasticsearch": 0.9, "opensearch": 0.9, "vector search": 1.0, "vector database": 1.0,
    "semantic search": 1.0, "hybrid search": 1.0, "bm25": 0.9, "information retrieval": 1.0,
    "retrieval": 0.9, "ranking": 1.0, "learning to rank": 1.0, "recommendation": 0.9,
    "recommender systems": 0.9, "recsys": 0.9,
    # Embeddings / modern NLP (JD: "absolutely need")
    "embeddings": 1.0, "sentence-transformers": 1.0, "sentence transformers": 1.0,
    "bge": 0.9, "e5": 0.8, "transformers": 0.8, "huggingface": 0.7, "hugging face": 0.7,
    "bert": 0.7, "nlp": 0.8, "natural language processing": 0.8, "rag": 0.8, "llm": 0.6,
    "llms": 0.6, "fine-tuning": 0.7, "lora": 0.6, "qlora": 0.6, "peft": 0.6,
    "spacy": 0.5, "word2vec": 0.5,
    # Eval / ML ranking (JD: "absolutely need" — eval frameworks)
    "ndcg": 1.0, "mrr": 0.9, "map": 0.7, "a/b testing": 0.7, "ab testing": 0.7,
    "xgboost": 0.7, "lightgbm": 0.6, "learning-to-rank": 1.0,
    # Core ML
    "machine learning": 0.7, "deep learning": 0.6, "pytorch": 0.7, "tensorflow": 0.6,
    "scikit-learn": 0.6, "sklearn": 0.6, "mlops": 0.6, "mlflow": 0.5, "kubeflow": 0.4,
    # Strong Python / data engineering (JD: "Strong Python. Yes really.")
    "python": 0.6, "spark": 0.4, "airflow": 0.35, "sql": 0.3, "pandas": 0.3, "numpy": 0.3,
    # Lower-relevance but not off-topic
    "kubernetes": 0.25, "docker": 0.25, "aws": 0.25, "gcp": 0.25, "kafka": 0.25,
}

# Career-narrative phrases that signal the candidate actually *built* the right
# kind of system (used to reward plain-language Tier-5 candidates who never list
# the buzzword skills but describe the real work in their job descriptions).
CAREER_SUBSTANCE_PHRASES = [
    "recommendation system", "recommender", "ranking system", "ranking model",
    "search relevance", "search engine", "semantic search", "vector search",
    "retrieval", "information retrieval", "personalization", "matching system",
    "embeddings", "sentence-transformer", "nearest neighbor", "ann ", "faiss",
    "elasticsearch", "opensearch", "bm25", "learning to rank", "learning-to-rank",
    "relevance", "candidate generation", "click-through", "ctr ", "feed ranking",
    "feature store", "ml pipeline", "model serving", "inference", "fine-tun",
    "a/b test", "ndcg", "deployed to production", "real users", "at scale",
    "recsys", "natural language", "nlp", "llm", "rag ",
]

# Phrases that signal rigorous evaluation thinking — a JD "absolutely need".
EVAL_PHRASES = ["ndcg", "mrr", "map ", "mean average precision", "offline metric",
                "a/b test", "ab test", "online experiment", "eval framework",
                "offline-to-online", "recall@", "precision@", "hit rate"]

# ---------------------------------------------------------------------------
# Company / industry classification. The JD penalizes lifelong services/consulting
# and rewards product-company applied-ML experience.
# ---------------------------------------------------------------------------

# Consulting / services firms the JD names explicitly (entire-career-here = penalty).
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "ltimindtree", "mindtree", " l&t infotech",
    "larsen", "mphasis", "hexaware", "birlasoft", "deloitte", "pwc", "kpmg", "ey ",
    "ibm services", "dxc", "ntt data", "persistent", "coforge",
]

# Industries that indicate a product company (reward) vs a services shop (penalty).
SERVICES_INDUSTRIES = {"it services", "consulting", "services", "bpo", "staffing"}
PRODUCT_INDUSTRIES = {
    "software", "fintech", "e-commerce", "ecommerce", "food delivery", "saas",
    "ai/ml", "ai services", "gaming", "edtech", "healthtech", "healthtech ai",
    "adtech", "conversational ai", "voice ai", "internet", "transportation",
    "insurance tech", "consumer electronics", "media",
}

# ---------------------------------------------------------------------------
# Location preference. JD: Pune/Noida preferred, hybrid; open to relocation from
# Tier-1 Indian cities; outside India case-by-case, no visa sponsorship.
# ---------------------------------------------------------------------------
PREFERRED_CITIES = {"pune", "noida"}             # JD offices
TIER1_INDIA_CITIES = {                            # JD: "Hyderabad, Pune, Mumbai, Delhi NCR"
    "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "bangalore", "bengaluru",
    "noida", "pune", "chennai", "kolkata",
}

# Experience band. JD: "5-9 years ... ideal candidate roughly 6-8 years."
EXP_IDEAL_LOW, EXP_IDEAL_HIGH = 6.0, 8.0
EXP_SOFT_LOW, EXP_SOFT_HIGH = 5.0, 9.0
