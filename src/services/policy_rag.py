import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Protocol

from src.schemas.retrieval import PolicyChunk


DEFAULT_POLICY_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"
_PREFIXES = frozenset("הבלומכש")
_INTENT_TOPICS = {
    "accident_towing": {"accident", "towing"},
    "maintenance_garage": {"maintenance", "warranty"},
    "leasing_contract": {"replacement_car", "towing"},
    "insurance_claim": {"accident", "insurance", "deductible"},
}
_DOCUMENT_HEADER = re.compile(r"^# (.+)\n<!-- doc_id: ([\w-]+) -->\s*$")
_SECTION_HEADER = re.compile(r"^([^\n]+)\n<!-- applies_to: ([^|\n]+) \| topics: ([^\n]+) -->\n(.+)$", re.S)


class PolicyRetriever(Protocol):
    def retrieve(
        self, query: str, contract_type: str | None = None, intent: str | None = None,
        top_k: int = 3,
    ) -> list[PolicyChunk]: ...


def _words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"[^\W_]+", normalized)


def _lexical_terms(words: list[str]) -> Counter[str]:
    terms = Counter(words)
    for word in words:
        if len(word) >= 4 and word[0] in _PREFIXES and re.fullmatch("[א-ת]+", word):
            terms[word[1:]] += 1
    return terms


def _character_grams(words: list[str]) -> Counter[str]:
    return Counter(
        word[start:start + size]
        for word in words
        for size in (3, 4)
        for start in range(len(word) - size + 1)
    )


def _idf(documents: list[Counter[str]]) -> dict[str, float]:
    frequencies = Counter(term for document in documents for term in document)
    return {
        term: math.log1p((len(documents) - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in frequencies.items()
    }


def _unit_tfidf(grams: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    weights = {gram: (1 + math.log(count)) * idf[gram] for gram, count in grams.items() if gram in idf}
    norm = math.sqrt(sum(weight * weight for weight in weights.values()))
    return {gram: weight / norm for gram, weight in weights.items()}


def _load_chunks(policy_dir: Path) -> list[PolicyChunk]:
    chunks = []
    doc_ids = set()
    for path in sorted(policy_dir.glob("*.md")):
        header, *sections = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)
        document = _DOCUMENT_HEADER.fullmatch(header)
        if document is None or not sections:
            raise ValueError(f"Missing policy document metadata or sections in {path.name}")
        title, doc_id = document.groups()
        if doc_id in doc_ids:
            raise ValueError(f"Duplicate policy doc_id {doc_id!r} in {path.name}")
        doc_ids.add(doc_id)
        for index, section in enumerate(sections, start=1):
            metadata = _SECTION_HEADER.fullmatch(section.strip())
            if metadata is None:
                raise ValueError(f"Invalid policy metadata in {path.name}, section {index}")
            heading, applies_to, topics, content = metadata.groups()
            chunks.append(PolicyChunk(
                chunk_id=f"{doc_id}-{index}",
                doc_id=doc_id,
                title=title,
                section=heading,
                applies_to=[tag.strip() for tag in applies_to.split(",")],
                topics=[tag.strip() for tag in topics.split(",")],
                content=content,
                score=0.0,
            ))
    if not chunks:
        raise ValueError(f"No Markdown policy sections found in {policy_dir}")
    return chunks


class LocalHybridPolicyRetriever:
    """Rank local policy sections with BM25 and character TF-IDF.

    A production HybridVectorPolicyRetriever can use Gemini Embeddings and
    pgvector (or Snowflake Cortex Search) behind the same PolicyRetriever protocol.
    """

    def __init__(self, policy_dir: str | Path = DEFAULT_POLICY_DIR) -> None:
        self._chunks = _load_chunks(Path(policy_dir))
        words = [_words(f"{chunk.title} {chunk.section} {chunk.content} {' '.join(chunk.topics)}")
                 for chunk in self._chunks]
        self._terms = [_lexical_terms(document) for document in words]
        self._lengths = [sum(terms.values()) for terms in self._terms]
        self._average_length = sum(self._lengths) / len(self._lengths)
        self._lexical_idf = _idf(self._terms)
        grams = [_character_grams(document) for document in words]
        self._gram_idf = _idf(grams)
        self._vectors = [_unit_tfidf(document, self._gram_idf) for document in grams]

    def _bm25(self, index: int, query_terms: Counter[str]) -> float:
        terms = self._terms[index]
        k1, b = 1.2, 0.75
        length_penalty = k1 * (1 - b + b * self._lengths[index] / self._average_length)
        return sum(
            self._lexical_idf[term] * terms[term] * (k1 + 1) / (terms[term] + length_penalty)
            for term in query_terms if term in terms
        )

    def retrieve(
        self, query: str, contract_type: str | None = None, intent: str | None = None,
        top_k: int = 3,
    ) -> list[PolicyChunk]:
        if top_k <= 0:
            return []
        words = _words(query)
        query_terms = _lexical_terms(words)
        query_vector = _unit_tfidf(_character_grams(words), self._gram_idf)
        aligned_topics = _INTENT_TOPICS.get(intent, {intent} if intent else set())
        ranked = []
        for index, chunk in enumerate(self._chunks):
            exact_contract = contract_type is not None and contract_type in chunk.applies_to
            if contract_type is not None and not exact_contract and "all" not in chunk.applies_to:
                continue
            lexical = self._bm25(index, query_terms)
            vector = self._vectors[index]
            cosine = sum(weight * vector.get(gram, 0.0) for gram, weight in query_vector.items())
            score = 0.5 * lexical / (lexical + 5) + 0.5 * cosine
            if score <= 0:
                continue
            if exact_contract:
                score *= 1.08
            if aligned_topics.intersection(chunk.topics):
                score *= 1.05
            ranked.append(chunk.model_copy(update={"score": score}, deep=True))
        return sorted(ranked, key=lambda chunk: (-chunk.score, chunk.chunk_id))[:top_k]
