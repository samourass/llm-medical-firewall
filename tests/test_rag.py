"""Tests unitaires du pipeline RAG, indépendants du firewall et du LLM."""
from llmfw.config import Settings
from llmfw.rag.chunking import chunk_text
from llmfw.rag.cleaning import clean_text
from llmfw.rag.embeddings import TfidfEmbedder, get_embedder
from llmfw.rag.pipeline import RAGPipeline


def test_clean_text_strips_markdown_structure():
    raw = "# Title\n\n> a quoted line\n\nSome   text   with    extra   spaces.\n\n\n\nNext para."
    cleaned = clean_text(raw)
    assert "#" not in cleaned
    assert ">" not in cleaned
    assert "  " not in cleaned


def test_chunk_text_respects_size_and_overlap():
    text = ". ".join(f"Sentence number {i}" for i in range(40)) + "."
    chunks = chunk_text(text, chunk_size_chars=100, overlap_chars=20)
    assert len(chunks) > 1
    assert all(len(c) <= 140 for c in chunks)  # marge pour la dernière phrase qui dépasse un peu
    # Le début d'un chunk (hors le premier) doit chevaucher la fin du chunk précédent.
    assert any(chunks[i][:15] in chunks[i - 1] for i in range(1, len(chunks)))


def test_tfidf_embedder_produces_normalized_vectors():
    emb = TfidfEmbedder()
    emb.fit(["the flu causes fever and cough", "hypertension is high blood pressure"])
    vectors = emb.encode(["fever and cough"])
    assert vectors.shape[0] == 1
    norm = (vectors[0] ** 2).sum() ** 0.5
    assert abs(norm - 1.0) < 1e-4 or norm == 0.0


def test_get_embedder_auto_never_raises_even_without_network(settings=None):
    settings = Settings(embedding_provider="auto")
    embedder = get_embedder(settings, corpus_for_tfidf_fit=["sample document text"])
    assert embedder.name in ("sentence_transformers", "tfidf")


def test_rag_pipeline_build_and_retrieve_end_to_end(tmp_path):
    docs_dir = tmp_path / "medical"
    docs_dir.mkdir()
    (docs_dir / "doc_a.md").write_text(
        "# Flu\n\nCommon flu symptoms include fever, cough, and sore throat. Rest helps.",
        encoding="utf-8",
    )
    (docs_dir / "doc_b.md").write_text(
        "# Appointments\n\nBook routine appointments one to three weeks in advance.",
        encoding="utf-8",
    )
    settings = Settings(medical_docs_dir=docs_dir, models_dir=tmp_path / "models",
                        embedding_provider="tfidf")
    rag = RAGPipeline(settings)
    stats = rag.build_index()
    assert stats["n_documents"] == 2
    assert stats["n_chunks"] >= 2
    assert stats["embedding_backend"] == "tfidf"

    results = rag.retrieve("What are flu symptoms?", k=2)
    assert len(results) > 0
    assert results[0].source == "doc_a.md"
    context = RAGPipeline.build_context(results)
    assert "[Source:" in context


def test_rag_pipeline_load_from_disk_matches_fresh_build(tmp_path):
    docs_dir = tmp_path / "medical"
    docs_dir.mkdir()
    (docs_dir / "doc_a.md").write_text("Flu symptoms include fever and cough.", encoding="utf-8")
    settings = Settings(medical_docs_dir=docs_dir, models_dir=tmp_path / "models",
                        embedding_provider="tfidf")
    RAGPipeline(settings).build_index()

    reloaded = RAGPipeline(settings)
    reloaded.load()
    results = reloaded.retrieve("fever and cough", k=1)
    assert len(results) == 1
    assert "fever" in results[0].text.lower()
