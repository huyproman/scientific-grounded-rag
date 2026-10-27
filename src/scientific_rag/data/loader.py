"""
Fast, robust parser and loader for BEIR SciFact JSONL and TSV data files.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from scientific_rag.common.paths import RAW_DATA_DIR
from scientific_rag.data.models import Document, Query

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads and validates documents, queries, and relevance assessments (qrels)."""

    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        self.data_dir = Path(data_dir).resolve()
        self.corpus_file = self.data_dir / "corpus.jsonl"
        self.queries_file = self.data_dir / "queries.jsonl"
        self.qrels_train_file = self.data_dir / "qrels" / "train.tsv"
        self.qrels_test_file = self.data_dir / "qrels" / "test.tsv"

    def load_corpus(self, max_docs: Optional[int] = None) -> List[Document]:
        """Loads corpus documents from corpus.jsonl."""
        if not self.corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found at {self.corpus_file}. Run download first.")

        documents: List[Document] = []
        seen_ids: Set[str] = set()

        with open(self.corpus_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSON at line %d: %s", line_no, e)
                    continue

                doc_id = str(data.get("_id", data.get("id", "")).strip())
                if not doc_id:
                    logger.warning("Skipping document without ID at line %d", line_no)
                    continue

                if doc_id in seen_ids:
                    logger.warning("Duplicate doc_id '%s' encountered at line %d; skipping duplicate.", doc_id, line_no)
                    continue
                seen_ids.add(doc_id)

                title = str(data.get("title", "")).strip()
                text = str(data.get("text", "")).strip()
                metadata = data.get("metadata", {})

                documents.append(
                    Document(
                        doc_id=doc_id,
                        title=title,
                        text=text,
                        metadata=metadata,
                    )
                )

                if max_docs and len(documents) >= max_docs:
                    break

        logger.info("Loaded %d documents from %s", len(documents), self.corpus_file)
        return documents

    def load_queries(self, split: Optional[str] = None) -> Dict[str, Query]:
        """Loads queries from queries.jsonl, optionally filtering by split."""
        if not self.queries_file.exists():
            raise FileNotFoundError(f"Queries file not found at {self.queries_file}.")

        queries: Dict[str, Query] = {}
        with open(self.queries_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed query at line %d: %s", line_no, e)
                    continue

                query_id = str(data.get("_id", data.get("id", "")).strip())
                if not query_id:
                    continue

                text = str(data.get("text", "")).strip()
                metadata = data.get("metadata", {})

                queries[query_id] = Query(
                    query_id=query_id,
                    text=text,
                    metadata=metadata,
                )

        if split:
            qrels = self.load_qrels(split=split)
            filtered_queries = {qid: q for qid, q in queries.items() if qid in qrels}
            logger.info("Filtered to %d queries for split '%s'", len(filtered_queries), split)
            return filtered_queries

        logger.info("Loaded %d total queries from %s", len(queries), self.queries_file)
        return queries

    def load_qrels(self, split: str = "test") -> Dict[str, Dict[str, int]]:
        """
        Loads relevance assessments for a split.
        Returns: {query_id: {doc_id: relevance_score}}
        """
        if split == "test":
            file_path = self.qrels_test_file
        elif split == "train":
            file_path = self.qrels_train_file
        else:
            raise ValueError(f"Unknown split: {split}. Expected 'test' or 'train'.")

        if not file_path.exists():
            raise FileNotFoundError(f"Qrels file not found at {file_path}.")

        qrels: Dict[str, Dict[str, int]] = {}
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)  # query-id, corpus-id, score

            for row_no, row in enumerate(reader, start=2):
                if not row or len(row) < 3:
                    continue
                qid = str(row[0]).strip()
                doc_id = str(row[1]).strip()
                try:
                    score = int(float(row[2]))
                except ValueError:
                    logger.warning("Invalid score '%s' at row %d", row[2], row_no)
                    continue

                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = score

        logger.info("Loaded qrels for %d queries from %s", len(qrels), file_path)
        return qrels

    def load_gold_rationales(self, claims_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
        """
        Loads official SciFact gold rationale annotations (e.g. claims_dev.jsonl).
        Returns: {claim_id: {doc_id: [{'sentences': [s1, s2], 'label': 'SUPPORT'/'CONTRADICT'}]}}
        """
        if claims_path is not None:
            resolved_claims_path = Path(claims_path)
            if not resolved_claims_path.exists():
                raise FileNotFoundError(f"Explicitly requested SciFact gold annotations file not found at {resolved_claims_path}.")
            path = resolved_claims_path
        else:
            candidates = [
                self.data_dir.parent / "scifact_official" / "data" / "claims_dev.jsonl",
                self.data_dir / "scifact_official" / "data" / "claims_dev.jsonl",
                self.data_dir / "claims_dev.jsonl",
                self.data_dir / "scifact" / "claims_dev.jsonl",
            ]
            path = next((p for p in candidates if p.exists()), None)
            if not path:
                logger.warning("SciFact gold rationale file not found in candidate paths. Gold rationale alignment will be empty.")
                return {}

        rationales: Dict[str, Dict[str, Any]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                claim_id = str(data.get("id", "")).strip()
                evidence_dict = data.get("evidence", {})
                rationales[claim_id] = evidence_dict

        logger.info("Loaded gold rationales for %d claims from %s", len(rationales), path)
        return rationales
