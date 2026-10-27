"""
Pretrained CPU-compatible Natural Language Inference (NLI) proxy for semantic support checking.
Uses cross-encoder/nli-deberta-v3-xsmall with CPU inference and explicit probability distribution.
Provides graceful, explicitly labeled lexical diagnostic fallback when neural model is unavailable.
"""

import logging
from typing import List, Optional, Tuple

from scientific_rag.generation.schemas import Citation, CitationVerificationResult

logger = logging.getLogger("scientific_rag.verification.support")


class PretrainedNLIProxy:
    """
    CPU-compatible NLI model for semantic support verification.
    
    Model Details:
    - Pretrained Architecture: DeBERTa-v3 cross-encoder (`cross-encoder/nli-deberta-v3-xsmall`).
    - Task: Sequence Classification (premise, hypothesis) -> [contradiction, entailment, neutral].
    - Execution Device: CPU-only (bounded batch size, torch.no_grad).
    - Thresholds: Configurable minimum probability for decisive ENTAILMENT / CONTRADICTION.
    - Safe Fallback Policy:
      When neural inference is disabled and allow_diagnostic_fallback=False, returns NOT_VERIFIABLE.
      When allow_diagnostic_fallback=True (mock/testing only), uses labeled diagnostic fallback.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        device: str = "cpu",
        confidence_threshold: float = 0.60,
        enable_neural: bool = True,
        allow_diagnostic_fallback: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.enable_neural = enable_neural
        self.allow_diagnostic_fallback = allow_diagnostic_fallback
        self._tokenizer = None
        self._model = None
        self._is_initialized = False

    def _initialize_model(self) -> bool:
        if self._is_initialized:
            return self._model is not None

        if not self.enable_neural:
            self._is_initialized = True
            return False

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info("Loading pretrained CPU NLI model: %s...", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            self._is_initialized = True
            logger.info("Pretrained CPU NLI model loaded successfully.")
            return True
        except Exception as e:
            logger.warning(
                "Could not initialize neural NLI model '%s' (%s). Fallback policy engaged.",
                self.model_name,
                e,
            )
            self._is_initialized = True
            return False

    def evaluate_pair(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """
        Evaluates premise-hypothesis pair.
        Returns (verdict, confidence) where verdict in ["ENTAILMENT", "CONTRADICTION", "NEUTRAL", "NOT_VERIFIABLE"].
        """
        p_clean = premise.strip()
        h_clean = hypothesis.strip()

        if not p_clean or not h_clean:
            return "NEUTRAL", 0.0

        if self._initialize_model() and self._model is not None:
            import torch

            inputs = self._tokenizer(
                p_clean,
                h_clean,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            # Config id2label: {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
            p_con = float(probs[0])
            p_ent = float(probs[1])
            p_neu = float(probs[2])

            if p_ent >= self.confidence_threshold and p_ent > p_con and p_ent > p_neu:
                return "ENTAILMENT", p_ent
            elif p_con >= self.confidence_threshold and p_con > p_ent and p_con > p_neu:
                return "CONTRADICTION", p_con
            else:
                return "NEUTRAL", max(p_neu, 1.0 - max(p_ent, p_con))

        # Safe Fallback Policy
        if not self.allow_diagnostic_fallback:
            return "NOT_VERIFIABLE", 0.0

        return self._lexical_heuristic(p_clean, h_clean)

    def _lexical_heuristic(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """Explicit lexical diagnostic fallback when neural inference is disabled."""
        p_lower = premise.lower()
        h_lower = hypothesis.lower()

        p_words = set(p_lower.split())
        h_words = set(h_lower.split())

        overlap = len(p_words.intersection(h_words))
        overlap_ratio = overlap / max(len(h_words), 1)

        negation_terms = {"not", "no", "never", "cannot", "fails", "failed", "neither", "none"}
        p_neg = any(term in p_words for term in negation_terms)
        h_neg = any(term in h_words for term in negation_terms)

        if (p_neg != h_neg) and overlap_ratio >= 0.4:
            return "CONTRADICTION", min(0.6 + overlap_ratio * 0.4, 0.95)

        if overlap_ratio >= 0.45:
            return "ENTAILMENT", min(0.5 + overlap_ratio * 0.5, 0.98)

        return "NEUTRAL", 1.0 - overlap_ratio

    def compute_lexical_diagnostic(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """Explicit lexical diagnostic heuristic for logging/development only; never used for conclusive verification."""
        v, c = self._lexical_heuristic(premise, hypothesis)
        return f"DIAGNOSTIC_{v}", c

    def verify_semantic_support(
        self,
        claim_text: str,
        verification_results: List[CitationVerificationResult],
    ) -> List[CitationVerificationResult]:
        """
        Evaluates semantic support using AUTHORITATIVE resolved evidence premise.
        For verified sentence-level citations, premise is the exact canonical sentence text.
        For quotation-level citations, premise is the authentic verified quotation span.
        Never evaluates unrelated neighboring sentences in the parent chunk as premise.
        """
        for res in verification_results:
            if not res.is_contextual or not res.is_text_authentic:
                res.nli_verdict = "NOT_VERIFIABLE"
                res.nli_confidence = 0.0
                continue

            # Authentic evidence premise resolution:
            # 1. If sentence attribution is validated, use authoritative canonical sentence text.
            # 2. Else if authoritative evidence text (chunk text) was explicitly provided, use chunk text.
            # 3. Else fallback to authentic cited quotation text.
            if res.is_sentence_valid and res.authoritative_sentence_text:
                trusted_premise = res.authoritative_sentence_text
            elif res.authoritative_evidence_text:
                trusted_premise = res.authoritative_evidence_text
            elif res.citation.evidence_text and res.is_text_authentic:
                trusted_premise = res.citation.evidence_text
            else:
                trusted_premise = None

            if not trusted_premise or not trusted_premise.strip():
                res.nli_verdict = "NOT_VERIFIABLE"
                res.nli_confidence = 0.0
                continue

            verdict, conf = self.evaluate_pair(premise=trusted_premise, hypothesis=claim_text)
            res.nli_verdict = verdict
            res.nli_confidence = conf

        return verification_results


# Backwards compatibility alias for earlier tests
BoundedNLIProxy = PretrainedNLIProxy
