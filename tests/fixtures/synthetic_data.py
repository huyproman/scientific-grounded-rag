"""
Deterministic synthetic fixtures for unit and integration testing.
"""

from typing import Dict, List, Tuple

from scientific_rag.data.models import Document, Query


def create_synthetic_corpus() -> List[Document]:
    """Creates a deterministic synthetic corpus of 5 documents."""
    return [
        Document(
            doc_id="doc_101",
            title="CRISPR-Cas9 Genome Editing Mechanism",
            text="CRISPR-Cas9 enables precise targeted genetic modifications in eukaryotic cells. The Cas9 endonuclease is guided by an engineered single-guide RNA. Double-strand breaks are repaired by non-homologous end joining or homology-directed repair.",
            metadata={"year": 2020, "field": "Genetics"},
        ),
        Document(
            doc_id="doc_102",
            title="Histone H3 Epigenetic Regulation",
            text="Post-translational modifications of histone H3 regulate chromatin architecture and transcription. Acetylation of lysine residues diminishes electrostatic affinity between histones and DNA. Methylation serves as a recruitment platform for chromatin remodelers.",
            metadata={"year": 2021, "field": "Epigenetics"},
        ),
        Document(
            doc_id="doc_103",
            title="Mitochondrial Dynamics in Neuronal Health",
            text="Mitochondrial fission and fusion govern metabolic homeostasis in mammalian neurons. Mitofusin 1 and 2 mediate outer mitochondrial membrane fusion. Impaired dynamics trigger neurodegenerative pathology.",
            metadata={"year": 2022, "field": "Neuroscience"},
        ),
        Document(
            doc_id="doc_104",
            title="Transformer Attention Mechanisms in Information Retrieval",
            text="Multi-head self-attention captures long-range lexical dependencies across document tokens. Inverted indices remain highly competitive for exact keyword matching. Dense bi-encoders project queries and passages into unified metric spaces.",
            metadata={"year": 2023, "field": "Computer Science"},
        ),
        Document(
            doc_id="doc_105",
            title="Therapeutic Antibodies Targeting Immune Checkpoints",
            text="Monoclonal antibodies targeting PD-1 and CTLA-4 restore cytotoxic T-cell activity against tumor cells. Checkpoint inhibition has transformed oncology paradigms. Combination immunotherapy demonstrates enhanced response rates.",
            metadata={"year": 2024, "field": "Immunology"},
        ),
    ]


def create_synthetic_queries_and_qrels() -> Tuple[Dict[str, Query], Dict[str, Dict[str, int]]]:
    """Creates synthetic queries with gold document-level relevance judgments."""
    queries = {
        "q1": Query(
            query_id="q1",
            text="How does Cas9 induce targeted double-strand breaks in DNA?",
            metadata={"domain": "Genetics"},
        ),
        "q2": Query(
            query_id="q2",
            text="What is the effect of histone H3 lysine acetylation on chromatin?",
            metadata={"domain": "Epigenetics"},
        ),
        "q3": Query(
            query_id="q3",
            text="Which proteins regulate outer mitochondrial membrane fusion?",
            metadata={"domain": "Neuroscience"},
        ),
        "q4": Query(
            query_id="q4",
            text="How do dense bi-encoders perform semantic passage retrieval?",
            metadata={"domain": "Computer Science"},
        ),
    }

    qrels = {
        "q1": {"doc_101": 1},
        "q2": {"doc_102": 1},
        "q3": {"doc_103": 1},
        "q4": {"doc_104": 1},
    }

    return queries, qrels
