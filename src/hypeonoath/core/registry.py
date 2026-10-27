"""
Project + document registry (document-registry plan v2, DR-4 / DR-3).

Owner-edited data, kept in core so the serving side can read it without
importing the ingestion package (design doc invariant I5: serving imports only
hypeonoath.core). hypeonoath.ingestion.config re-exports these names, so
Phase 0 / Phase 1 code keeps reading them as config.<NAME>. Lookups and checks
live in hypeonoath.core.document_registry. Tests that change the registry for
serving / validation monkeypatch THIS module.
"""
from __future__ import annotations

# --- Project registry (Phase 1 guideline v4, Sections 3A.12 / 16) -----------
# THE project registry for code AND documents (document-registry plan v2,
# DR-4); a project may have no code file. Every project has a compulsory
# display name and description, defined ONCE here; code files point to it via
# CODE_PROJECT_MAP, documents via DOCUMENTS[...]["project"], so a project's
# description can never drift between files. Keys are internal: derived once
# from the name, never asked from the owner, never changed on rename (DR-10).
#
# Values are verbatim from the user ("gliimr" 2026-10-01, "hypeonoath"
# 2026-10-27) -- do not reword: name and description are shown to the LLM as
# grounded, citable context, and Phase 1's stale-metadata check compares these
# strings exactly.
CODE_PROJECTS: dict[str, dict[str, str]] = {
    "gliimr": {
        "name": "Gliimr: A personal health intelligence app",
        "description": (
            "On-device personal health intelligence app. No cloud. No hallucinations.\n"
            "Gliimr is a React Native health app that passively senses how you live, "
            "extracts meaning from what you say, and maintains a clinically grounded "
            "probabilistic model of your health - entirely on your phone, entirely in "
            "private, entirely explainable."
        ),
    },
    "hypeonoath": {
        "name": "HypeOnOath: A Privacy-Preserving Professional RAG Agent",
        "description": (
            "A grounded RAG chatbot that answers recruiters' questions about me in first person, "
            "from my own documents and code, with citations and private details kept out."
        ),
    },
}

# --- Document registry (document-registry plan v2, DR-3 / DR-8) --------------
# Every document, keyed by path relative to corpus/raw/: a compulsory
# description (factual, <= 300 characters, no private value) and an optional
# project (a key of CODE_PROJECTS). Shown to the LLM when the prompt is built
# (serving/prompt.py), never embedded. An indexed document with no entry stops
# the index build (indexing/build_index.py). Replaces PROJECT_NAME_MAP.
# Values are verbatim from the user (2026-10-27) -- do not reword.
DOCUMENTS: dict[str, dict[str, str | None]] = {
    "system-design-rag-chatbot.md": {
        "description": ("System design document for my recruiter-facing grounded RAG chatbot: architecture, "
                        "technology decisions, data design, risks and build roadmap."),
        "project": "hypeonoath",
    },
    "README.md": {
        "description": ("README of my Gliimr app: what it does, its architecture, the three-model pipeline, "
                        "the Dynamic Bayesian Network and the conversational pipeline."),
        "project": "gliimr",
    },
    "CV_Udbhav.pdf": {
        "description": "My CV tailored for AI and machine learning roles.",
        "project": None,
    },
    "CV_Udbhav2.pdf": {
        "description": "My CV tailored for data analytics, business analytics and related roles.",
        "project": None,
    },
    "CV_Udbhav3.pdf": {
        "description": ("Another version of the CV tailored for data analytics and business analytics and "
                        "business intelligence and related roles."),
        "project": None,
    },
    "Cover_Letter_Udbhav.pdf": {
        "description": "A sample cover letter: motivation, background and what I bring to the role.",
        "project": None,
    },
    "AcceptanceLetter1_Zulassung.pdf": {
        "description": "My admission letter (in German and English) for my master's at TU Darmstadt.",
        "project": None,
    },
    "Coursera AnalyzingBigDataWithSQL Certificate.pdf": {
        "description": "My Coursera course certificate for Analyzing Big Data with SQL.",
        "project": None,
    },
    "Coursera ImprovingDeepNeuralNetworks (1).pdf": {
        "description": "My Coursera course certificate for Improving Deep Neural Networks.",
        "project": None,
    },
    "Coursera ModernBigDataAnalysisWithSQL Certificate.pdf": {
        "description": "My Coursera course certificate for Modern Big Data Analysis with SQL.",
        "project": None,
    },
    "Coursera NeuralNetworksAndDeepLearning.pdf": {
        "description": "My Coursera course certificate for Neural Networks and Deep Learning.",
        "project": None,
    },
    "Coursera StructuringMachineLearningProjects.pdf": {
        "description": "My Coursera course certificate for Structuring Machine Learning Projects.",
        "project": None,
    },
}

# File -> project_key, keyed by path relative to corpus/raw/. A key ending in
# "/" is a folder prefix: every code file under it belongs to that project
# (umbrella project). The longest matching key wins.
CODE_PROJECT_MAP: dict[str, str] = {
    "agent.ts": "gliimr",
}
