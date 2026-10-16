"""Canonical skill taxonomy shared by resume parsing and job requirement extraction.

Matching on canonical names (not raw strings) is what makes "DRF" on a resume
match "Django REST Framework" in a job description. Extend freely; aliases are
matched case-insensitively on word boundaries.
"""
from __future__ import annotations

import re
from functools import lru_cache

# canonical name -> (category, aliases)
TAXONOMY: dict[str, tuple[str, list[str]]] = {
    "Python": ("language", ["python", "python3"]),
    "Java": ("language", ["java"]),
    "JavaScript": ("language", ["javascript", "js", "ecmascript"]),
    "TypeScript": ("language", ["typescript", "ts"]),
    "Go": ("language", ["golang", "go lang"]),
    "Rust": ("language", ["rust"]),
    "C++": ("language", ["c++", "cpp"]),
    "C#": ("language", ["c#", "csharp"]),
    "Ruby": ("language", ["ruby"]),
    "Scala": ("language", ["scala"]),
    "Kotlin": ("language", ["kotlin"]),
    "SQL": ("language", ["sql"]),
    "Bash": ("language", ["bash", "shell scripting"]),
    "Django": ("framework", ["django"]),
    "Django REST Framework": ("framework", ["django rest framework", "drf", "django-rest-framework"]),
    "Flask": ("framework", ["flask"]),
    "FastAPI": ("framework", ["fastapi"]),
    "Celery": ("framework", ["celery"]),
    "Angular": ("framework", ["angular", "angular 2+", "angularjs"]),
    "React": ("framework", ["react", "react.js", "reactjs"]),
    "Vue": ("framework", ["vue", "vue.js", "vuejs"]),
    "Node.js": ("framework", ["node.js", "nodejs"]),
    "Spring": ("framework", ["spring boot", "spring framework"]),
    "RxJS": ("framework", ["rxjs"]),
    "Pandas": ("library", ["pandas"]),
    "NumPy": ("library", ["numpy"]),
    "PyTorch": ("library", ["pytorch", "torch"]),
    "TensorFlow": ("library", ["tensorflow", "keras"]),
    "scikit-learn": ("library", ["scikit-learn", "sklearn"]),
    "LangChain": ("library", ["langchain"]),
    "LlamaIndex": ("library", ["llamaindex", "llama index"]),
    "Hugging Face": ("library", ["hugging face", "huggingface", "transformers"]),
    "PostgreSQL": ("database", ["postgresql", "postgres", "psql"]),
    "MySQL": ("database", ["mysql"]),
    "MongoDB": ("database", ["mongodb", "mongo"]),
    "Redis": ("database", ["redis"]),
    "Elasticsearch": ("database", ["elasticsearch", "opensearch"]),
    "pgvector": ("database", ["pgvector"]),
    "Vector Databases": ("database", ["vector database", "vector databases", "pinecone", "weaviate", "qdrant", "milvus", "chroma"]),
    "Kafka": ("infra", ["kafka", "apache kafka"]),
    "RabbitMQ": ("infra", ["rabbitmq"]),
    "Docker": ("infra", ["docker", "containers", "containerization"]),
    "Kubernetes": ("infra", ["kubernetes", "k8s", "eks", "gke", "aks"]),
    "Terraform": ("infra", ["terraform"]),
    "AWS": ("cloud", ["aws", "amazon web services", "ec2", "s3", "lambda", "rds", "sqs", "ecs"]),
    "GCP": ("cloud", ["gcp", "google cloud"]),
    "Azure": ("cloud", ["azure"]),
    "CI/CD": ("practice", ["ci/cd", "continuous integration", "github actions", "jenkins", "gitlab ci"]),
    "Git": ("tool", ["git"]),
    "Linux": ("tool", ["linux", "unix"]),
    "Nginx": ("infra", ["nginx"]),
    "REST APIs": ("practice", ["rest api", "rest apis", "restful", "rest framework"]),
    "GraphQL": ("practice", ["graphql"]),
    "gRPC": ("practice", ["grpc"]),
    "Microservices": ("practice", ["microservices", "microservice architecture"]),
    "Distributed Systems": ("practice", ["distributed systems", "distributed system"]),
    "System Design": ("practice", ["system design", "software architecture"]),
    "Unit Testing": ("practice", ["unit testing", "pytest", "unittest", "tdd", "jest", "jasmine", "karma"]),
    "Agile": ("practice", ["agile", "scrum", "kanban"]),
    "Machine Learning": ("ai", ["machine learning", "ml"]),
    "Deep Learning": ("ai", ["deep learning"]),
    "NLP": ("ai", ["nlp", "natural language processing"]),
    "LLMs": ("ai", ["llm", "llms", "large language models", "large language model", "generative ai", "genai", "gpt", "claude", "openai"]),
    "RAG": ("ai", ["rag", "retrieval augmented generation", "retrieval-augmented generation"]),
    "Prompt Engineering": ("ai", ["prompt engineering"]),
    "MLOps": ("ai", ["mlops"]),
    "Computer Vision": ("ai", ["computer vision"]),
    "Data Engineering": ("data", ["data engineering", "etl", "elt", "data pipelines", "data pipeline"]),
    "Airflow": ("data", ["airflow", "apache airflow"]),
    "Spark": ("data", ["spark", "pyspark", "apache spark"]),
    "dbt": ("data", ["dbt"]),
    "Snowflake": ("data", ["snowflake"]),
    "BigQuery": ("data", ["bigquery"]),
    "HTML/CSS": ("frontend", ["html", "css", "scss", "sass", "html5", "css3"]),
    "Bootstrap": ("frontend", ["bootstrap"]),
    "Angular Material": ("frontend", ["angular material"]),
    "Leadership": ("soft", ["team lead", "tech lead", "technical lead", "mentoring", "mentored", "led a team", "people management"]),
}


@lru_cache(maxsize=1)
def _patterns():
    compiled = []
    for canonical, (_category, aliases) in TAXONOMY.items():
        for alias in sorted(set(aliases + [canonical.lower()]), key=len, reverse=True):
            # Word-ish boundaries that tolerate symbols such as C++ / C# / Node.js
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9+#])", re.I)
            compiled.append((canonical, alias, pattern))
    return compiled


def extract_skills(text: str) -> list[str]:
    """Return canonical skills mentioned in text, in order of first appearance."""
    if not text:
        return []
    found: dict[str, int] = {}
    for canonical, alias, pattern in _patterns():
        match = pattern.search(text)
        if not match:
            continue
        if alias == "go" and not re.search(r"(?<![A-Za-z])Go(?![A-Za-z])", text):
            continue  # require capitalised "Go" to avoid the English verb
        pos = match.start()
        if canonical not in found or pos < found[canonical]:
            found[canonical] = pos
    return [name for name, _ in sorted(found.items(), key=lambda kv: kv[1])]


def canonicalize(name: str) -> str:
    """Map a free-form skill name to its canonical form (or a tidy version of the input)."""
    cleaned = (name or "").strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    for canonical, (_category, aliases) in TAXONOMY.items():
        if lowered == canonical.lower() or lowered in aliases:
            return canonical
    return cleaned[:100]


def category_of(name: str) -> str:
    entry = TAXONOMY.get(name)
    return entry[0] if entry else "other"
