"""
Unit tests for DatasetLoader parsing and schema validation.
"""

import tempfile
from pathlib import Path
from scientific_rag.data.loader import DatasetLoader


def test_loader_parses_valid_data():
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = Path(tmp_dir)
        qrels_dir = data_path / "qrels"
        qrels_dir.mkdir(parents=True)

        with open(data_path / "corpus.jsonl", "w", encoding="utf-8") as f:
            f.write('{"_id": "doc1", "title": "Paper One", "text": "Content of paper one.", "metadata": {}}\n')
            f.write('{"_id": "doc2", "title": "Paper Two", "text": "Content of paper two.", "metadata": {}}\n')

        with open(data_path / "queries.jsonl", "w", encoding="utf-8") as f:
            f.write('{"_id": "q1", "text": "Query one?", "metadata": {}}\n')

        with open(qrels_dir / "test.tsv", "w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            f.write("q1\tdoc1\t1\n")

        loader = DatasetLoader(data_path)
        docs = loader.load_corpus()
        assert len(docs) == 2
        assert docs[0].doc_id == "doc1"
        assert docs[1].title == "Paper Two"

        queries = loader.load_queries()
        assert len(queries) == 1
        assert queries["q1"].text == "Query one?"

        qrels = loader.load_qrels(split="test")
        assert qrels["q1"]["doc1"] == 1


def test_loader_handles_malformed_and_duplicate():
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = Path(tmp_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        with open(data_path / "corpus.jsonl", "w", encoding="utf-8") as f:
            f.write("this is not json\n")
            f.write('{"_id": "doc1", "title": "Valid", "text": "Valid text"}\n')
            f.write('{"_id": "doc1", "title": "Duplicate", "text": "Duplicate text"}\n')
            f.write('{"title": "Missing ID", "text": "No ID text"}\n')

        loader = DatasetLoader(data_path)
        docs = loader.load_corpus()
        assert len(docs) == 1
        assert docs[0].doc_id == "doc1"
        assert docs[0].title == "Valid"
