# pyright: reportAttributeAccessIssue=false

import json
import os
import re
import time
import atexit
from typing import Dict, List

from google import genai
from google.genai import types


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

import numpy as np
import cv2
from scipy.linalg import lstsq


def set_proxy_from_local_default() -> None:
    proxy_url = 'socks5h://127.0.0.1:7890'
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    os.environ['ALL_PROXY'] = proxy_url


# Tip-path utilities previously in modules.TipPath_Fuc
def center_to_square(center, edge):
    x, y = center
    x1 = round(x - 0.5 * edge)
    y1 = round(y - 0.5 * edge)
    x2 = round(x + 0.5 * edge)
    y2 = round(y + 0.5 * edge)
    return (x1, y1), (x2, y2)


def increase_radius(scan_qulity, R_last, R_init, R_max, R_step):
    if not scan_qulity:
        if R_last < R_max:
            R = R_last + R_step
        else:
            R = R_max
    else:
        R = R_init
    return R


def pix_to_nanocoordinate(pix_point, plane_size=2000):
    (X, Y) = pix_point
    X = round(X - plane_size / 2) * 1e-9
    Y = (-1) * round(Y - plane_size / 2) * 1e-9
    return (X, Y)


def linear_whole(matrix):
    rows, cols = matrix.shape
    Y, X = np.indices((rows, cols))
    X = X.ravel()
    Y = Y.ravel()
    data = matrix.ravel()

    A = np.column_stack([X, Y, np.ones(rows * cols)])
    c, _, _, _ = lstsq(A, data)

    fitted_plane = c[0] * X + c[1] * Y + c[2]
    fitted_plane = fitted_plane.reshape(rows, cols)

    processed_matrix = matrix - fitted_plane
    return processed_matrix


def linear_normalize_whole(matrix):
    rows, cols = matrix.shape
    Y, X = np.indices((rows, cols))
    X = X.ravel()
    Y = Y.ravel()
    data = matrix.ravel()

    A = np.column_stack([X, Y, np.ones(rows * cols)])
    c, _, _, _ = lstsq(A, data)

    fitted_plane = c[0] * X + c[1] * Y + c[2]
    fitted_plane = fitted_plane.reshape(rows, cols)

    processed_matrix = matrix - fitted_plane

    min_val = processed_matrix.min()
    max_val = processed_matrix.max()
    if min_val == max_val:
        return processed_matrix

    normalized_matrix = 255 * (processed_matrix - min_val) / (max_val - min_val)
    normalized_matrix = np.array(normalized_matrix, dtype=np.uint8)
    return normalized_matrix


def images_equalization(image, alpha=0.5):
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    equalized_image = cv2.equalizeHist(norm_image)
    adjusted_image = cv2.addWeighted(norm_image, 1 - alpha, equalized_image, alpha, 0)
    return adjusted_image


def Hex_to_BGR(hex_color: str):
    """Convert a hex color string to an OpenCV BGR tuple."""
    if not isinstance(hex_color, str):
        raise ValueError('Hex_to_BGR requires a string')
    color = hex_color.strip()
    if color.startswith('#'):
        color = color[1:]
    if len(color) != 6:
        raise ValueError('Hex_to_BGR requires a 6-digit hex color')
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return (b, g, r)


class ConditioningAgentMixin:
    """Two-stage Conditioning Agent: Literature Learning Agent + Action Agent."""

    def _ensure_conditioning_state(self):
        if not hasattr(self, 'conditioning_learned_ranges'):
            self.conditioning_learned_ranges = {}
        if not hasattr(self, 'conditioning_learning_time'):
            self.conditioning_learning_time = None
        if not hasattr(self, 'conditioning_history_file'):
            self.conditioning_history_file = None
        if not hasattr(self, '_conditioning_embedding_model'):
            self._conditioning_embedding_model = None
        if not hasattr(self, '_conditioning_vector_store'):
            self._conditioning_vector_store = None
        if not hasattr(self, '_conditioning_reranker'):
            self._conditioning_reranker = None
        if not hasattr(self, '_conditioning_exit_hook_registered'):
            self._conditioning_exit_hook_registered = False
        self._register_conditioning_exit_hooks()

    def _literature_dir(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(root_dir, 'literature')

    def _conditioning_rag_dir(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        rag_dir = os.path.join(root_dir, '.conditioning_rag')
        os.makedirs(rag_dir, exist_ok=True)
        return rag_dir

    def _conditioning_faiss_index_path(self):
        return os.path.join(self._conditioning_rag_dir(), 'literature_faiss')

    def _conditioning_rag_manifest_path(self):
        return os.path.join(self._conditioning_rag_dir(), 'literature_manifest.json')

    def _conditioning_literature_state_path(self):
        return os.path.join(self._conditioning_rag_dir(), 'literature_state_last_run.json')

    def _conditioning_learning_cache_path(self):
        return os.path.join(self._conditioning_rag_dir(), 'conditioning_learned_ranges_cache.json')

    def _repair_conditioning_experience_path(self):
        return os.path.join(os.path.dirname(__file__), 'repair_conditioning_for_agent.json')

    def _save_json_file(self, path, data):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f'[Conditioning Agent] failed to save json: {path}, err={exc}')

    def _load_json_file(self, path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _collect_literature_basic_info(self):
        literature_dir = self._literature_dir()
        files = []
        if os.path.isdir(literature_dir):
            for name in sorted(os.listdir(literature_dir)):
                if not name.lower().endswith('.pdf'):
                    continue
                path = os.path.join(literature_dir, name)
                try:
                    size_bytes = int(os.path.getsize(path))
                except Exception:
                    size_bytes = -1
                files.append({'name': name, 'size_bytes': size_bytes})

        return {
            'saved_at': time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())),
            'literature_dir': literature_dir,
            'pdf_count': len(files),
            'files': files,
        }

    def _literature_signature_from_basic_info(self, basic_info):
        if not isinstance(basic_info, dict):
            return ''
        files = basic_info.get('files', [])
        if not isinstance(files, list):
            return ''
        parts = []
        for item in files:
            if not isinstance(item, dict):
                continue
            name = str(item.get('name', ''))
            size_bytes = int(item.get('size_bytes', -1))
            parts.append(f'{name}:{size_bytes}')
        return '|'.join(parts)

    def _save_literature_state_for_next_run(self):
        info = self._collect_literature_basic_info()
        info['signature'] = self._literature_signature_from_basic_info(info)
        self._save_json_file(self._conditioning_literature_state_path(), info)

    def _register_conditioning_exit_hooks(self):
        if getattr(self, '_conditioning_exit_hook_registered', False):
            return
        try:
            atexit.register(self._save_literature_state_for_next_run)
            self._conditioning_exit_hook_registered = True
        except Exception:
            self._conditioning_exit_hook_registered = False

    def _rag_config(self):
        return {
            'max_files': int(os.getenv('CONDITIONING_RAG_MAX_FILES', '10')),
            'chunk_size': int(os.getenv('CONDITIONING_RAG_CHUNK_SIZE', '4096')),
            'chunk_overlap': int(os.getenv('CONDITIONING_RAG_CHUNK_OVERLAP', '1024')),
            'retrieval_k': int(os.getenv('CONDITIONING_RAG_RETRIEVAL_K', '12')),
            'rerank_top_n': int(os.getenv('CONDITIONING_RAG_RERANK_TOP_N', '4')),
            'embedding_model_path': os.getenv('CONDITIONING_EMBEDDING_MODEL_PATH', 'F:/models/all-MiniLM-L6-v2'),
            'reranker_model_path': os.getenv('CONDITIONING_RERANKER_MODEL_PATH', 'F:/models/bge-reranker-large'),
        }

    def _list_literature_pdfs(self, max_files=10):
        literature_dir = self._literature_dir()
        if not os.path.isdir(literature_dir):
            return []
        pdf_files = [
            os.path.join(literature_dir, name)
            for name in sorted(os.listdir(literature_dir))
            if name.lower().endswith('.pdf')
        ]
        return pdf_files[:max_files]

    def _literature_corpus_signature(self, pdf_files):
        signature_items = []
        for path in pdf_files:
            try:
                stat = os.stat(path)
                signature_items.append(f"{os.path.basename(path)}:{int(stat.st_mtime)}:{stat.st_size}")
            except Exception:
                signature_items.append(f"{os.path.basename(path)}:missing")
        return '|'.join(signature_items)

    def _conditioning_history_path(self):
        self._ensure_conditioning_state()
        if self.conditioning_history_file:
            return self.conditioning_history_file
        log_root = getattr(self, 'log_path', './log')
        os.makedirs(log_root, exist_ok=True)
        self.conditioning_history_file = os.path.join(log_root, 'conditioning_history.jsonl')
        return self.conditioning_history_file

    def _append_conditioning_record(self, record):
        history_path = self._conditioning_history_path()
        with open(history_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _parse_human_experience_action(self, action_text, default_pulse_width):
        if not isinstance(action_text, str):
            return None

        tip_match = re.search(r'\bTipShaper\s*\|\s*([-+]?\d+(?:\.\d+)?)\s*n\b', action_text, re.IGNORECASE)
        if tip_match:
            tip_lift = float(tip_match.group(1))
            tip_lift = max(-5.0, min(0.0, tip_lift))
            return {
                'action': 'TipShaper',
                'TipLift': f'{tip_lift}n',
            }

        pulse_match = re.search(
            r'\bPulse\s*\|\s*([-+]?\d+(?:\.\d+)?)\s*V\b(?:\s*\(\s*width\s*=\s*([-+]?\d+(?:\.\d+)?)\s*\))?',
            action_text,
            re.IGNORECASE,
        )
        if pulse_match:
            voltage = float(pulse_match.group(1))
            width = pulse_match.group(2)
            if width is None:
                width_value = float(default_pulse_width)
            else:
                width_value = float(width)
            voltage = max(-10.0, min(10.0, voltage))
            width_value = max(0.001, min(1.0, width_value))
            return {
                'action': 'Pulse',
                'voltage': float(voltage),
                'width': float(width_value),
            }

        return None

    def _plan_signature(self, action_plan):
        if not isinstance(action_plan, dict):
            return ''
        action = str(action_plan.get('action', ''))
        if action == 'Pulse':
            return f"Pulse:{float(action_plan.get('voltage', 0.0))}:{float(action_plan.get('width', 0.05))}"
        if action == 'TipShaper':
            return f"TipShaper:{str(action_plan.get('TipLift', ''))}"
        return ''

    def _load_human_experience_candidates(self, default_pulse_width):
        json_path = self._repair_conditioning_experience_path()
        data = self._load_json_file(json_path)
        if not isinstance(data, dict) or not data:
            return []

        merged_candidates = []
        for key in ('top5_operation_recommendations', 'top5_final_repair_actions'):
            rows = data.get(key, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                action_text = str(row.get('action', ''))
                parsed_plan = self._parse_human_experience_action(action_text, default_pulse_width)
                if parsed_plan is None:
                    continue

                probability = row.get('probability', 0.0)
                try:
                    probability = float(probability)
                except Exception:
                    probability = 0.0

                if probability <= 0.0:
                    try:
                        success = float(row.get('success', 0.0))
                        total = float(row.get('total', 0.0))
                        if total > 0:
                            probability = success / total
                    except Exception:
                        probability = 0.0

                probability = max(0.0, min(1.0, float(probability)))
                merged_candidates.append(
                    {
                        'plan': parsed_plan,
                        'human_probability': probability,
                        'source': key,
                    }
                )

        best_by_signature = {}
        for item in merged_candidates:
            sig = self._plan_signature(item.get('plan', {}))
            if not sig:
                continue
            prev = best_by_signature.get(sig)
            if prev is None or item['human_probability'] > prev['human_probability']:
                best_by_signature[sig] = item

        return list(best_by_signature.values())

    def _evaluation_action_bias_score(self, eval_info, action_name):
        bias = ''
        if isinstance(eval_info, dict):
            bias = str(eval_info.get('recommended_action_bias', '')).strip().lower()

        if bias == 'tipshaper_first':
            return 1.0 if action_name == 'TipShaper' else 0.0
        if bias == 'pulse_first':
            return 1.0 if action_name == 'Pulse' else 0.0
        if bias == 'either':
            return 0.5
        return 0.5

    def _load_recent_conditioning_records(self, limit=30):
        history_path = self._conditioning_history_path()
        if not os.path.exists(history_path):
            return []
        records = []
        with open(history_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        if limit <= 0:
            return records
        return records[-limit:]

    def _get_document_class(self):
        try:
            from langchain_core.documents import Document  # type: ignore
            return Document
        except Exception:
            pass
        try:
            from langchain.schema import Document  # type: ignore
            return Document
        except Exception:
            pass
        return None

    def _get_text_splitter(self, chunk_size, chunk_overlap):
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        except Exception:
            pass
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        except Exception:
            pass
        return None

    def _load_literature_documents_for_rag(self, max_files=10):
        Document = self._get_document_class()
        if Document is None:
            print('[Conditioning RAG] Document class unavailable (langchain not installed).')
            return []

        pdf_files = self._list_literature_pdfs(max_files=max_files)
        docs = []
        if not pdf_files:
            return docs

        fitz = None
        try:
            import fitz as _fitz  # type: ignore
            fitz = _fitz
        except Exception:
            fitz = None

        for pdf_path in pdf_files:
            basename = os.path.basename(pdf_path)
            if fitz is not None:
                try:
                    doc = fitz.open(pdf_path)
                    try:
                        for page_idx in range(len(doc)):
                            page = doc[page_idx]
                            page_text = page.get_text('text')
                            if page_text is None:
                                page_text = ''
                            if not isinstance(page_text, str):
                                page_text = str(page_text)
                            page_text = page_text.strip()
                            if not page_text:
                                continue
                            docs.append(
                                Document(
                                    page_content=page_text,
                                    metadata={
                                        'source': basename,
                                        'pdf_path': pdf_path,
                                        'page': page_idx + 1,
                                    },
                                )
                            )
                    finally:
                        doc.close()
                    continue
                except Exception:
                    pass

            try:
                with open(pdf_path, 'rb') as f:
                    raw = f.read(500000)
                fallback_text = raw.decode('latin-1', errors='ignore').strip()
                if fallback_text:
                    docs.append(
                        Document(
                            page_content=fallback_text,
                            metadata={
                                'source': basename,
                                'pdf_path': pdf_path,
                                'page': 0,
                            },
                        )
                    )
            except Exception:
                continue

        return docs

    def _extract_literature_texts(self, max_files=10, max_chars_per_file=20000):
        literature_dir = self._literature_dir()
        if not os.path.isdir(literature_dir):
            return []

        pdf_files = [
            os.path.join(literature_dir, name)
            for name in sorted(os.listdir(literature_dir))
            if name.lower().endswith('.pdf')
        ][:max_files]

        docs = []

        fitz = None
        try:
            import fitz as _fitz  # type: ignore
            fitz = _fitz
        except Exception:
            fitz = None

        for pdf_path in pdf_files:
            text = ''
            if fitz is not None:
                try:
                    doc = fitz.open(pdf_path)
                    try:
                        for page in doc:
                            page_text = page.get_text('text')
                            if page_text is None:
                                page_text = ''
                            if not isinstance(page_text, str):
                                page_text = str(page_text)
                            text += page_text + '\n'
                            if len(text) >= max_chars_per_file:
                                break
                    finally:
                        doc.close()
                except Exception:
                    text = ''

            if not text:
                try:
                    with open(pdf_path, 'rb') as f:
                        raw = f.read(500000)
                    text = raw.decode('latin-1', errors='ignore')
                except Exception:
                    text = ''

            docs.append(
                {
                    'file': os.path.basename(pdf_path),
                    'text': text[:max_chars_per_file],
                }
            )
        return docs

    def _get_embedding_model(self):
        self._ensure_conditioning_state()
        if self._conditioning_embedding_model is not None:
            return self._conditioning_embedding_model

        cfg = self._rag_config()
        model_path = cfg['embedding_model_path']

        EmbeddingsClass = None
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
            EmbeddingsClass = HuggingFaceEmbeddings
        except Exception:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
                EmbeddingsClass = HuggingFaceEmbeddings
            except Exception:
                EmbeddingsClass = None

        if EmbeddingsClass is None:
            print('[Conditioning RAG] HuggingFaceEmbeddings unavailable.')
            return None

        try:
            self._conditioning_embedding_model = EmbeddingsClass(
                model_name=model_path,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
            )
            print(f'[Conditioning RAG] embedding model loaded: {model_path}')
            return self._conditioning_embedding_model
        except Exception as exc:
            print(f'[Conditioning RAG] embedding model load failed: {exc}')
            return None

    def _get_reranker(self):
        self._ensure_conditioning_state()
        if self._conditioning_reranker is not None:
            return self._conditioning_reranker

        cfg = self._rag_config()
        model_path = cfg['reranker_model_path']
        try:
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder  # type: ignore
            try:
                from langchain_classic.retrievers.document_compressors import CrossEncoderReranker  # type: ignore
            except Exception:
                from langchain.retrievers.document_compressors import CrossEncoderReranker  # type: ignore
        except Exception:
            print('[Conditioning RAG] reranker dependencies unavailable, fallback to lexical ranking.')
            return None

        try:
            cross_encoder = HuggingFaceCrossEncoder(
                model_name=model_path,
                model_kwargs={'device': 'cpu'},
            )
            self._conditioning_reranker = CrossEncoderReranker(model=cross_encoder, top_n=cfg['rerank_top_n'])
            print(f'[Conditioning RAG] reranker model loaded: {model_path}')
            return self._conditioning_reranker
        except Exception as exc:
            print(f'[Conditioning RAG] reranker load failed: {exc}')
            return None

    def _load_or_build_vector_store(self, force_rebuild=False):
        self._ensure_conditioning_state()
        if self._conditioning_vector_store is not None and not force_rebuild:
            return self._conditioning_vector_store

        embedding = self._get_embedding_model()
        if embedding is None:
            return None

        try:
            from langchain_community.vectorstores import FAISS  # type: ignore
        except Exception:
            print('[Conditioning RAG] FAISS vector store unavailable.')
            return None

        cfg = self._rag_config()
        pdf_files = self._list_literature_pdfs(max_files=cfg['max_files'])
        if not pdf_files:
            print('[Conditioning RAG] no pdf files found in literature.')
            return None

        corpus_signature = self._literature_corpus_signature(pdf_files)
        index_path = self._conditioning_faiss_index_path()
        manifest_path = self._conditioning_rag_manifest_path()

        should_rebuild = bool(force_rebuild)
        manifest = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
            except Exception:
                manifest = {}

        if manifest.get('corpus_signature') != corpus_signature:
            should_rebuild = True

        faiss_exists = os.path.exists(f'{index_path}.faiss') and os.path.exists(f'{index_path}.pkl')
        if not faiss_exists:
            should_rebuild = True

        if not should_rebuild:
            try:
                self._conditioning_vector_store = FAISS.load_local(
                    index_path,
                    embedding,
                    allow_dangerous_deserialization=True,
                )
                return self._conditioning_vector_store
            except Exception as exc:
                print(f'[Conditioning RAG] load local index failed, rebuild needed: {exc}')
                should_rebuild = True

        docs = self._load_literature_documents_for_rag(max_files=cfg['max_files'])
        if not docs:
            print('[Conditioning RAG] no readable documents for indexing.')
            return None

        splitter = self._get_text_splitter(cfg['chunk_size'], cfg['chunk_overlap'])
        if splitter is not None:
            chunks = splitter.split_documents(docs)
        else:
            chunks = docs
        for idx, chunk in enumerate(chunks):
            meta = dict(getattr(chunk, 'metadata', {}) or {})
            meta['chunk_id'] = idx
            chunk.metadata = meta

        self._conditioning_vector_store = FAISS.from_documents(chunks, embedding)
        self._conditioning_vector_store.save_local(index_path)

        manifest_data = {
            'corpus_signature': corpus_signature,
            'pdf_count': len(pdf_files),
            'chunk_count': len(chunks),
            'updated_at': time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())),
            'config': cfg,
        }
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

        print(f"[Conditioning RAG] index ready: pdf={len(pdf_files)}, chunks={len(chunks)}")
        return self._conditioning_vector_store

    def _lexical_rerank(self, query, docs, top_n=4):
        tokens = [t for t in re.split(r'\\W+', (query or '').lower()) if len(t) > 1]
        scored = []
        for doc in docs:
            text = str(getattr(doc, 'page_content', '')).lower()
            score = sum(text.count(tok) for tok in tokens)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:max(1, top_n)]]

    def _rerank_documents(self, query, docs, top_n=4):
        if not docs:
            return []
        reranker = self._get_reranker()
        if reranker is None:
            return self._lexical_rerank(query, docs, top_n=top_n)
        try:
            reranked = reranker.compress_documents(docs, query)
            reranked = list(reranked)
            return reranked[:max(1, top_n)]
        except Exception as exc:
            print(f'[Conditioning RAG] rerank failed, fallback to lexical: {exc}')
            return self._lexical_rerank(query, docs, top_n=top_n)

    def _conditioning_rag_search(self, query, force_reindex=False):
        cfg = self._rag_config()
        vector_store = self._load_or_build_vector_store(force_rebuild=force_reindex)
        if vector_store is None:
            return []
        try:
            retrieved = vector_store.similarity_search(query, k=cfg['retrieval_k'])
        except Exception as exc:
            print(f'[Conditioning RAG] similarity search failed: {exc}')
            return []

        reranked = self._rerank_documents(query, retrieved, top_n=cfg['rerank_top_n'])
        contexts = []
        for doc in reranked:
            meta = dict(getattr(doc, 'metadata', {}) or {})
            contexts.append(
                {
                    'source': meta.get('source', 'unknown'),
                    'page': meta.get('page', 'unknown'),
                    'chunk_id': meta.get('chunk_id', 'unknown'),
                    'text': str(getattr(doc, 'page_content', '')),
                }
            )
        return contexts

    def _safe_json_extract(self, text):
        if isinstance(text, list):
            text = ''.join([str(x) for x in text])
        if not isinstance(text, str):
            return {}
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}

    def _kimi_chat_json(self, messages):
        api_key = getattr(self, 'quality_model_path', '')
        if not isinstance(api_key, str) or not api_key.startswith('AIza'):
            api_key = os.getenv('GEMINI_API_KEY', GEMINI_API_KEY)
        if not isinstance(api_key, str) or not api_key:
            raise RuntimeError('Gemini API key is unavailable.')

        model_name = getattr(self, 'gemini_model_path', os.getenv('GEMINI_MODEL_PATH', os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')))
        timeout_seconds = int(getattr(self, 'gemini_timeout_seconds', int(os.getenv('GEMINI_TIMEOUT', 90))))
        retries = int(getattr(self, 'gemini_request_retries', int(os.getenv('GEMINI_RETRIES', 2))))
        backoff = float(getattr(self, 'gemini_retry_backoff_seconds', float(os.getenv('GEMINI_RETRY_BACKOFF', 1.0))))

        set_proxy_from_local_default()
        client = genai.Client(api_key=api_key)

        system_lines = []
        user_lines = []
        for msg in messages if isinstance(messages, list) else []:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get('role', '')).strip().lower()
            content = str(msg.get('content', '') or '')
            if role == 'system':
                system_lines.append(content)
            elif role == 'user':
                user_lines.append(content)

        merged_prompt = (
            'System instruction:\n'
            + '\n\n'.join(system_lines)
            + '\n\nUser instruction:\n'
            + '\n\n'.join(user_lines)
            + '\n\nReturn strict JSON only.'
        )
        gen_config = types.GenerateContentConfig(response_mime_type='application/json', temperature=0.0)

        last_exc = None
        for attempt in range(retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=merged_prompt,
                    config=gen_config,
                )
                content = (getattr(response, 'text', None) or '').strip() or str(response)
                parsed = self._safe_json_extract(content)
                if parsed:
                    return parsed
                return {'raw': content}
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    time.sleep(backoff * (attempt + 1))
        raise RuntimeError(f'Gemini request failed: {last_exc}')

    def _sanitize_learned_ranges(self, learned_ranges):
        pulse = learned_ranges.get('pulse', {}) if isinstance(learned_ranges, dict) else {}
        tip = learned_ranges.get('tipshaper', {}) if isinstance(learned_ranges, dict) else {}

        pulse_values = pulse.get('recommended_voltages', [])
        if not isinstance(pulse_values, list):
            pulse_values = []
        pulse_values = [float(x) for x in pulse_values if isinstance(x, (int, float, str))]

        tip_values = tip.get('recommended_tiplift_n', [])
        if not isinstance(tip_values, list):
            tip_values = []
        tip_values = [float(x) for x in tip_values if isinstance(x, (int, float, str))]

        pulse_min = float(pulse.get('min_v', -6.0))
        pulse_max = float(pulse.get('max_v', -1.0))
        tip_min_n = float(tip.get('min_lift_n', -4.0))
        tip_max_n = float(tip.get('max_lift_n', -1.0))
        pulse_width = float(pulse.get('pulse_width_s', 0.05))

        pulse_min = max(-10.0, min(10.0, pulse_min))
        pulse_max = max(-10.0, min(10.0, pulse_max))
        if pulse_min > pulse_max:
            pulse_min, pulse_max = pulse_max, pulse_min

        tip_min_n = max(-5.0, min(0.0, tip_min_n))
        tip_max_n = max(-5.0, min(0.0, tip_max_n))
        if tip_min_n > tip_max_n:
            tip_min_n, tip_max_n = tip_max_n, tip_min_n

        negative_pulses = [v for v in pulse_values if pulse_min <= v <= pulse_max and v < 0]
        if not negative_pulses:
            candidate = [-1.0, -2.0, -4.0, -6.0, -8.0, -10.0]
            negative_pulses = [v for v in candidate if pulse_min <= v <= pulse_max]
        negative_pulses = sorted(set(negative_pulses), key=lambda x: abs(x))

        tip_candidates = [v for v in tip_values if tip_min_n <= v <= tip_max_n and v < 0]
        if not tip_candidates:
            candidate = [-1.0, -1.5, -2.0, -2.5, -3.0, -4.0, -5.0]
            tip_candidates = [v for v in candidate if tip_min_n <= v <= tip_max_n]
        tip_candidates = sorted(set(tip_candidates), key=lambda x: abs(x))

        pulse_width = max(0.001, min(1.0, pulse_width))

        return {
            'pulse': {
                'min_v': pulse_min,
                'max_v': pulse_max,
                'pulse_width_s': pulse_width,
                'recommended_negative_voltages': negative_pulses,
            },
            'tipshaper': {
                'min_lift_n': tip_min_n,
                'max_lift_n': tip_max_n,
                'recommended_lift_n': tip_candidates,
            },
            'hard_limits': {
                'pulse_abs_max_v': 10.0,
                'tipshaper_min_lift_n': -5.0,
            },
        }

    def conditioning_learning_agent(self, force_refresh=False):
        """
        Conditioning Learning Agent with RAG:
        - Parse PDFs and build/load FAISS index
        - Retrieve relevant chunks by conditioning query and rerank them
        - Feed retrieved context to Kimi for range extraction
        """
        self._ensure_conditioning_state()

        current_literature_info = self._collect_literature_basic_info()
        current_literature_signature = self._literature_signature_from_basic_info(current_literature_info)
        last_literature_info = self._load_json_file(self._conditioning_literature_state_path())
        last_literature_signature = self._literature_signature_from_basic_info(last_literature_info)
        literature_changed_since_last_run = bool(last_literature_signature) and (
            current_literature_signature != last_literature_signature
        )

        if self.conditioning_learned_ranges and not force_refresh:
            return self.conditioning_learned_ranges

        cached_bundle = self._load_json_file(self._conditioning_learning_cache_path())
        cached_signature = str(cached_bundle.get('literature_signature', ''))
        cached_ranges = cached_bundle.get('learned_ranges', {})
        if (
            not force_refresh
            and isinstance(cached_ranges, dict)
            and cached_ranges
            and cached_signature
            and cached_signature == current_literature_signature
        ):
            sanitized_cached = self._sanitize_learned_ranges(cached_ranges)
            cached_meta = cached_ranges.get('learning_meta', {}) if isinstance(cached_ranges, dict) else {}
            sanitized_cached['learning_meta'] = {
                'doc_count': int(current_literature_info.get('pdf_count', 0)),
                'rag_enabled': bool(cached_meta.get('rag_enabled', True)),
                'updated_at': str(cached_meta.get('updated_at', cached_bundle.get('cached_at', ''))),
                'evidence': cached_meta.get('evidence', []),
                'notes': str(cached_meta.get('notes', 'loaded_from_local_cache')),
                'loaded_from_cache': True,
                'literature_changed_since_last_run': False,
            }
            self.conditioning_learned_ranges = sanitized_cached
            self.conditioning_learning_time = sanitized_cached['learning_meta']['updated_at']
            print('[Conditioning Learning Agent] loaded ranges from local cache (literature unchanged).')
            return sanitized_cached

        if literature_changed_since_last_run:
            print('[Conditioning Learning Agent] literature changed since last run, refresh learning ranges.')

        retrieval_query = (
            'STM tip conditioning parameter ranges for pulse voltage and tip dip/tip shaper depth. '
            'Search aliases: voltage pulse, pulsing bias, conditioning voltage, field emission voltage, V_pulse, '
            'indentation depth, vertical plunge, z-displacement, controlled crash, dipping distance. '
            'Need numeric ranges with units V/mV and nm/Å.'
        )
        rag_contexts = self._conditioning_rag_search(
            retrieval_query,
            force_reindex=(force_refresh or literature_changed_since_last_run),
        )

        if rag_contexts:
            docs_block = []
            for idx, item in enumerate(rag_contexts, start=1):
                docs_block.append(
                    f"[Retrieved {idx}] source={item['source']}, page={item['page']}, chunk={item['chunk_id']}\n"
                    f"{item['text']}"
                )
            docs_prompt = '\n\n'.join(docs_block)
            rag_enabled = True
        else:
            docs = self._extract_literature_texts(max_files=self._rag_config()['max_files'], max_chars_per_file=20000)
            docs_block = []
            for idx, doc in enumerate(docs, start=1):
                docs_block.append(f"[Doc {idx}] {doc['file']}\n{doc['text'][:4000]}")
            docs_prompt = '\n\n'.join(docs_block)
            rag_enabled = False

        system_prompt = (
            'You are a scientific STM conditioning knowledge extractor. '
            'Extract practical bias voltage (V) and tip dip (nm) ranges from literature text only. '
            'Note: These parameters, corresponding to pulse and tipshaper in STM control, may appear under various terms. '
            'Return strict JSON only.'
        )
        user_prompt = (
            'Task: From this literature context below, extract ranges for STM tip conditioning. '
            'Return JSON with schema: '
            '{"pulse":{"min_v":number,"max_v":number,"pulse_width_s":number,"recommended_voltages":[number]},'
            '"tipshaper/tip dip":{"min_lift_n":number,"max_lift_n":number,"recommended_tiplift_n":[number]},'
            '"evidence":[{"file":string,"quote":string}],"notes":string}. '
            'Use negative pulse as preferred polarity when literature supports it. '
            'Do not include units in numeric fields.\n\n'
            f'{docs_prompt}'
        )

        learned = {}
        try:
            learned = self._kimi_chat_json(
                [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ]
            )
        except Exception as exc:
            print(f'[Conditioning Learning Agent] Gemini parse failed: {exc}')

        sanitized = self._sanitize_learned_ranges(learned)
        sanitized['learning_meta'] = {
            'doc_count': int(current_literature_info.get('pdf_count', 0)),
            'rag_enabled': rag_enabled,
            'updated_at': time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())),
            'evidence': learned.get('evidence', []) if isinstance(learned, dict) else [],
            'notes': learned.get('notes', '') if isinstance(learned, dict) else '',
            'loaded_from_cache': False,
            'literature_changed_since_last_run': literature_changed_since_last_run,
        }
        self.conditioning_learned_ranges = sanitized
        self.conditioning_learning_time = sanitized['learning_meta']['updated_at']

        cache_payload = {
            'cached_at': sanitized['learning_meta']['updated_at'],
            'literature_signature': current_literature_signature,
            'literature_info': current_literature_info,
            'learned_ranges': sanitized,
        }
        self._save_json_file(self._conditioning_learning_cache_path(), cache_payload)

        print('[Conditioning Learning Agent] learned ranges ready: '
              f"pulse={sanitized['pulse']['recommended_negative_voltages']}, "
              f"tip={sanitized['tipshaper']['recommended_lift_n']}, "
              f"rag_enabled={rag_enabled}")
        return sanitized

    def conditioning_agent_decision(self, eval_info, scan_image, consecutive_failures):
        """
        Conditioning Action Agent decision basis:
        1) Learned ranges from Conditioning Learning Agent + hard limits
        2) Evaluation Agent outputs (label/reason/severity)
        3) Historical conditioning records and outcomes
        """
        if scan_image is None:
            raise ValueError('Conditioning Action Agent requires current scan image.')

        learned_ranges = self.conditioning_learning_agent(force_refresh=False)
        pulse_candidates = learned_ranges['pulse']['recommended_negative_voltages']
        tip_candidates = learned_ranges['tipshaper']['recommended_lift_n']
        pulse_width = learned_ranges['pulse']['pulse_width_s']

        recent_records = self._load_recent_conditioning_records(limit=30)
        recent_failures = 0
        for rec in reversed(recent_records):
            if rec.get('outcome_label') == 'bad':
                recent_failures += 1
            else:
                break

        stage = max(0, int(consecutive_failures)) + min(recent_failures, 3)
        reason = str(eval_info.get('reason', 'bad image')) if isinstance(eval_info, dict) else 'bad image'
        severity = str(eval_info.get('severity', 'unknown')) if isinstance(eval_info, dict) else 'unknown'

        human_candidates = self._load_human_experience_candidates(default_pulse_width=pulse_width)
        if human_candidates:
            human_weight = 0.7
            eval_weight = 0.3
            weighted_candidates = []
            for item in human_candidates:
                plan = dict(item.get('plan', {}))
                action_name = str(plan.get('action', ''))
                human_probability = float(item.get('human_probability', 0.0))
                eval_bias_score = self._evaluation_action_bias_score(eval_info, action_name)
                combined_score = human_weight * human_probability + eval_weight * eval_bias_score
                weighted_candidates.append(
                    {
                        'plan': plan,
                        'combined_score': combined_score,
                        'human_probability': human_probability,
                        'eval_bias_score': eval_bias_score,
                        'source': str(item.get('source', 'unknown')),
                    }
                )

            weighted_candidates.sort(
                key=lambda x: (x['combined_score'], x['human_probability'], x['eval_bias_score']),
                reverse=True,
            )
            selected_idx = min(stage, len(weighted_candidates) - 1)
            selected = weighted_candidates[selected_idx]
            action_plan = dict(selected['plan'])
            action_plan['reason'] = (
                f"{reason} | severity={severity} "
                f"| strategy=human70_eval30 | score={selected['combined_score']:.3f}"
            )
            action_plan['range_source'] = learned_ranges
            action_plan['decision_meta'] = {
                'human_weight': 0.7,
                'evaluation_weight': 0.3,
                'human_probability': selected['human_probability'],
                'evaluation_bias_score': selected['eval_bias_score'],
                'candidate_source': selected['source'],
                'ranked_candidates_count': len(weighted_candidates),
            }
            return action_plan

        if stage < len(pulse_candidates):
            voltage = pulse_candidates[stage]
            return {
                'action': 'Pulse',
                'voltage': float(voltage),
                'width': float(pulse_width),
                'reason': f'{reason} | severity={severity} | strategy=learned_ranges_fallback',
                'range_source': learned_ranges,
            }

        tip_stage = stage - len(pulse_candidates)
        if tip_stage >= len(tip_candidates):
            tip_stage = len(tip_candidates) - 1
        tip_lift_n = tip_candidates[max(0, tip_stage)]
        return {
            'action': 'TipShaper',
            'TipLift': f'{tip_lift_n}n',
            'reason': f'{reason} | severity={severity} | strategy=learned_ranges_fallback',
            'range_source': learned_ranges,
        }

    def execute_conditioning_action(self, action_plan):
        """Execute Conditioning Action Agent plan. TipShaper is always applied at Point Nemo."""
        action = action_plan.get('action')
        if action == 'Pulse':
            voltage = float(action_plan.get('voltage', -2.0))
            width = float(action_plan.get('width', 0.05))
            voltage = max(-10.0, min(10.0, voltage))
            self.BiasPulse(voltage, width=width)
            self.conditioning_tipshaper_streak = 0
            self.navigator_skip_budget = 0
            print(f'[Conditioning Action Agent] Pulse executed: V={voltage}, width={width}s')
            return

        if action == 'TipShaper':
            if self.nemo_nanocoodinate is None:
                raise ValueError('Point Nemo is unavailable, cannot execute TipShaper.')

            tip_lift = str(action_plan.get('TipLift', '-1.5n'))
            m = re.match(r'\s*([-+]?\d+(?:\.\d+)?)\s*n\s*$', tip_lift)
            if m:
                tip_value = float(m.group(1))
                tip_value = max(-5.0, min(0.0, tip_value))
                tip_lift = f'{tip_value}n'

            self.TipXYSet(self.nemo_nanocoodinate[0], self.nemo_nanocoodinate[1])
            time.sleep(1)
            self.TipShaper(TipLift=tip_lift)

            self.conditioning_tipshaper_streak += 1
            if self.conditioning_tipshaper_streak >= 4:
                self.navigator_skip_budget = max(self.navigator_skip_budget, 3)
            elif self.conditioning_tipshaper_streak >= 3:
                self.navigator_skip_budget = max(self.navigator_skip_budget, 2)
            else:
                self.navigator_skip_budget = max(self.navigator_skip_budget, 1)

            print(f'[Conditioning Action Agent] TipShaper executed at Point Nemo: TipLift={tip_lift}')
            print(
                f'[Planning Agent] skip budget updated: budget={self.navigator_skip_budget}, '
                f'tipshaper_streak={self.conditioning_tipshaper_streak}'
            )
            return

        raise ValueError(f'Unsupported conditioning action: {action}')

    def apply_conditioning_agent(self, eval_info, scan_image):
        """End-to-end Conditioning Agent: Learning range -> decide -> execute -> pending record."""
        self._ensure_conditioning_state()
        action_plan = self.conditioning_agent_decision(
            eval_info=eval_info,
            scan_image=scan_image,
            consecutive_failures=self.consecutive_repair_failures,
        )
        self.execute_conditioning_action(action_plan)

        self.pending_conditioning_action = {
            'action_plan': action_plan,
            'eval_before': dict(eval_info) if isinstance(eval_info, dict) else {'reason': 'unknown'},
            'ts': time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())),
        }
        return action_plan

    def update_conditioning_effectiveness(self, current_scan_quality, eval_info=None):
        """
        Evaluate previous conditioning action using next scan result.
        Also persist conditioning history with evaluation result and reason.
        """
        if self.pending_conditioning_action is None:
            return

        if current_scan_quality == 0:
            self.consecutive_repair_failures += 1
            self_outcome = 'bad'
            print('修复无效')
        else:
            self.consecutive_repair_failures = 0
            self.conditioning_tipshaper_streak = 0
            self_outcome = 'good'

        pending = self.pending_conditioning_action
        action_plan = pending.get('action_plan', {})
        before_eval = pending.get('eval_before', {})
        after_eval = dict(eval_info) if isinstance(eval_info, dict) else {}

        record = {
            'timestamp': time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())),
            'action': action_plan.get('action'),
            'action_plan': action_plan,
            'outcome_label': self_outcome,
            'consecutive_repair_failures_after': int(self.consecutive_repair_failures),
            'evaluation_before': before_eval,
            'evaluation_after': after_eval,
            'outcome_reason': after_eval.get('reason', after_eval.get('rationale', '')),
        }
        try:
            self._append_conditioning_record(record)
        except Exception as exc:
            print(f'[Conditioning Agent] failed to persist record: {exc}')

        self.pending_conditioning_action = None

        if self.consecutive_repair_failures >= 20:
            print('\033[31m!!! 连续20次修复无效，请人工介入 !!!\033[0m')


if __name__ == '__main__':
    class _MockConditioningAgent(ConditioningAgentMixin):
        def __init__(self):
            self._root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.log_path = os.path.join(self._root_dir, 'log')
            self.quality_model_path = os.getenv('GEMINI_API_KEY', '')
            self.gemini_model_path = os.getenv('GEMINI_MODEL_PATH', os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'))
            self.gemini_timeout_seconds = 3
            self.gemini_request_retries = 0
            self.gemini_retry_backoff_seconds = 0.1
            self.pending_conditioning_action = None
            self.consecutive_repair_failures = 0
            self.conditioning_tipshaper_streak = 0
            self.navigator_skip_budget = 0
            self.nemo_nanocoodinate = (0.0, 0.0)
            self.executed_actions = []

        def _literature_dir(self):
            return os.path.join(self._root_dir, 'C:\\Users\\17712\\Desktop\\SPM-nanonis_TCP\\literature')

        def _conditioning_rag_dir(self):
            rag_dir = os.path.join(self._root_dir, '.conditioning_rag')
            os.makedirs(rag_dir, exist_ok=True)
            return rag_dir

        def BiasPulse(self, bias, width=0.1):
            self.executed_actions.append({'action': 'Pulse', 'bias': float(bias), 'width': float(width)})
            print(f'[Mock Nanonis] BiasPulse called: bias={bias}, width={width}')

        def TipXYSet(self, x, y):
            self.executed_actions.append({'action': 'TipXYSet', 'x': float(x), 'y': float(y)})
            print(f'[Mock Nanonis] TipXYSet called: x={x}, y={y}')

        def TipShaper(self, TipLift='-1.5n'):
            self.executed_actions.append({'action': 'TipShaper', 'TipLift': str(TipLift)})
            print(f'[Mock Nanonis] TipShaper called: TipLift={TipLift}')

    print('[Conditioning Agent Test] start')
    agent = _MockConditioningAgent()

    fake_eval_bad = {'label': 'bad', 'reason': 'blurred/ghosting', 'severity': 'high'}
    fake_scan_image = [[0]]

    print('[Conditioning Agent Test] decision test ...')
    decision = agent.conditioning_agent_decision(
        eval_info=fake_eval_bad,
        scan_image=fake_scan_image,
        consecutive_failures=0,
    )
    print(f"[Conditioning Agent Test] decision={decision.get('action')}, reason={decision.get('reason', '')}")

    print('[Conditioning Agent Test] execute Pulse ...')
    agent.execute_conditioning_action({'action': 'Pulse', 'voltage': -2.0, 'width': 0.05})
    time.sleep(5)

    print('[Conditioning Agent Test] execute TipShaper ...')
    agent.execute_conditioning_action({'action': 'TipShaper', 'TipLift': '-1.5n'})
    time.sleep(5)

    print('[Conditioning Agent Test] end-to-end apply + update ...')
    action_plan = agent.apply_conditioning_agent(fake_eval_bad, fake_scan_image)
    print(f"[Conditioning Agent Test] applied action={action_plan.get('action')}")
    agent.update_conditioning_effectiveness(current_scan_quality=1, eval_info={'reason': 'image improved'})

    print(f'[Conditioning Agent Test] executed_actions_count={len(agent.executed_actions)}')
    print('[Conditioning Agent Test] done')
