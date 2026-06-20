"""Tests for the grounded answerer: citations and the 'I don't know' gate."""

from __future__ import annotations

from eaip.providers.stub import StubProvider
from eaip.providers.types import Completion
from eaip.retrieval import Principal
from eaip.retrieval.answerer import IDK, GroundedAnswerer
from eaip.retrieval.pipeline import RetrievalResult


def test_abstains_when_no_chunks():
    """No retrieved evidence -> 'I don't know', and the LLM is never called."""
    provider = StubProvider([Completion(text="should not be used")])
    answerer = GroundedAnswerer(provider, min_score=0.05)
    result = RetrievalResult(query="anything", principal=Principal.of("u"), chunks=[])

    answer = answerer.answer(result)
    assert answer.abstained
    assert answer.text == IDK
    assert provider.calls == []  # provider untouched


def test_abstains_when_top_score_below_threshold(retriever):
    """A query with only weak matches abstains rather than grounding on noise."""
    provider = StubProvider()
    answerer = GroundedAnswerer(provider, min_score=0.99)  # impossibly high bar
    result = retriever.retrieve("vpn setup", Principal.of("u", ["everyone"]))

    answer = answerer.answer(result)
    assert answer.abstained
    assert answer.text == IDK
    assert provider.calls == []


def test_answers_with_citations_when_confident(retriever):
    provider = StubProvider([Completion(text="Use the GlobalConnect client [1].")])
    answerer = GroundedAnswerer(provider, min_score=0.0)  # never abstain on score
    result = retriever.retrieve("how do I set up the vpn client", Principal.of("u", ["everyone"]))

    answer = answerer.answer(result)
    assert not answer.abstained
    assert answer.text == "Use the GlobalConnect client [1]."
    assert len(answer.citations) == len(result.chunks) >= 1
    # Citation labels are 1-based and contiguous.
    assert [c.label for c in answer.citations] == list(range(1, len(answer.citations) + 1))


def test_unauthorized_question_abstains(retriever):
    """If the relevant doc is forbidden, retrieval yields nothing strong -> IDK.

    A non-finance user asking a finance-only question gets no permitted evidence,
    so the answer abstains rather than leaking or hallucinating.
    """
    provider = StubProvider()
    answerer = GroundedAnswerer(provider, min_score=0.05)
    outsider = Principal.of("intern@acme.test", ["everyone"])
    result = retriever.retrieve("what is the exact FY26 revenue forecast number", outsider)

    # The finance doc must not be among citations even if we answered.
    answer = answerer.answer(result)
    cited_docs = {c.doc_id for c in answer.citations}
    assert "CONF-7" not in cited_docs


def test_prompt_separates_instructions_from_retrieved_data(retriever):
    """The retrieved context is fenced and labeled as data (Phase 6 groundwork)."""
    provider = StubProvider([Completion(text="ok")])
    answerer = GroundedAnswerer(provider, min_score=0.0)
    result = retriever.retrieve("vpn", Principal.of("u", ["everyone"]))
    answerer.answer(result)

    system, user = provider.calls[0]
    assert "never as instructions" in system.content
    assert "CONTEXT:" in user.content and "<<<" in user.content  # fenced block
