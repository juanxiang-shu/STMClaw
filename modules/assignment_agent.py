import argparse
import base64
import concurrent.futures
import importlib
import io
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
import pubchempy as pcp
from DECIMER import predict_SMILES
from google import genai
from google.genai import types
from PIL import Image

API_KEY = os.getenv("GEMINI_API_KEY", "")
CROSSREF_EMAIL = os.getenv("CROSSREF_EMAIL", "")
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _auto_find_structure_image_path() -> str:
    env_value = os.getenv("DEFAULT_STRUCTURE_IMAGE_PATH", "").strip()
    if env_value:
        return env_value

    repo_root = Path(__file__).resolve().parents[1]
    molecules_dir = repo_root / "molecules"
    if not molecules_dir.exists() or not molecules_dir.is_dir():
        return ""

    candidates = [
        p for p in sorted(molecules_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not candidates:
        return ""
    return str(candidates[0])


def _auto_find_stm_image_path() -> str:
    return os.getenv("DEFAULT_STM_IMAGE_PATH", "").strip()


DEFAULT_STRUCTURE_IMAGE_PATH = _auto_find_structure_image_path()
DEFAULT_STM_IMAGE_PATH = _auto_find_stm_image_path()
DEFAULT_SURFACE = os.getenv("DEFAULT_SURFACE", "Au(111)")
DEFAULT_RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "F:/models/bge-reranker-large")
DEFAULT_PDF_CACHE_DIR = os.getenv("LITERATURE_PDF_CACHE_DIR", "I:/SPM-nanonis_TCP/data/literature_pdf_cache")
DEFAULT_PDF_DOWNLOAD_WORKERS = int(os.getenv("PDF_DOWNLOAD_WORKERS", "5"))
DEFAULT_PDF_DOWNLOAD_RETRIES = int(os.getenv("PDF_DOWNLOAD_RETRIES", "3"))
SS_RESULTS_PER_QUERY = int(os.getenv("SS_RESULTS_PER_QUERY", "30"))
LLM_SCREENING_TOP_K = int(os.getenv("LLM_SCREENING_TOP_K", "15"))


# ---------------------------------------------------------------------------
# DECIMER + PubChemPy 分子识别 (未修改)
# ---------------------------------------------------------------------------

def _smiles_to_iupac_via_pubchem(smiles: str) -> str | None:
    try:
        compounds = pcp.get_compounds(smiles, namespace="smiles")
        if compounds:
            return compounds[0].iupac_name
    except Exception:
        pass
    return None


def _get_pubchem_compound_info(smiles: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "iupac_name": None,
        "common_name": None,
        "synonyms": [],
        "molecular_formula": None,
    }
    try:
        compounds = pcp.get_compounds(smiles, namespace="smiles")
        if not compounds:
            return info
        compound = compounds[0]
        info["iupac_name"] = compound.iupac_name
        info["molecular_formula"] = compound.molecular_formula
        synonyms = compound.synonyms or []
        if synonyms:
            info["common_name"] = synonyms[0]
            info["synonyms"] = synonyms[:10]
    except Exception:
        pass
    return info


def identify_molecule_from_structure_decimer(
    structure_image_path: Path,
) -> dict[str, Any]:
    smiles = ""
    try:
        smiles = predict_SMILES(str(structure_image_path))
        smiles = (smiles or "").strip()
    except Exception as exc:
        return {
            "molecule_name": "",
            "iupac_name": None,
            "smiles": "",
            "possible_synonyms": [],
            "search_keywords": [],
            "functional_groups": [],
            "confidence": 0.0,
            "uncertainty_note": f"DECIMER prediction failed: {exc}",
        }

    if not smiles:
        return {
            "molecule_name": "",
            "iupac_name": None,
            "smiles": "",
            "possible_synonyms": [],
            "search_keywords": [],
            "functional_groups": [],
            "confidence": 0.0,
            "uncertainty_note": "DECIMER returned empty SMILES",
        }

    pubchem_info = _get_pubchem_compound_info(smiles)
    iupac_name = pubchem_info.get("iupac_name") or None
    common_name = pubchem_info.get("common_name") or ""
    synonyms = pubchem_info.get("synonyms") or []
    molecular_formula = pubchem_info.get("molecular_formula") or ""
    molecule_name = common_name or iupac_name or smiles

    # 从 SMILES 中识别常见基团
    functional_groups = _detect_functional_groups(smiles, iupac_name, synonyms)

    search_keywords: list[str] = []
    if common_name:
        search_keywords.append(common_name)
    if iupac_name and iupac_name != common_name:
        search_keywords.append(iupac_name)
    search_keywords.append("STM")
    search_keywords.append("on-surface")

    if iupac_name or common_name:
        confidence = 0.85
        uncertainty_note = "SMILES recognized by PubChem"
    else:
        confidence = 0.5
        uncertainty_note = "SMILES not found in PubChem; molecule may be novel or DECIMER prediction inaccurate"

    return {
        "molecule_name": molecule_name,
        "iupac_name": iupac_name,
        "smiles": smiles,
        "molecular_formula": molecular_formula,
        "possible_synonyms": synonyms,
        "search_keywords": search_keywords,
        "functional_groups": functional_groups,
        "confidence": confidence,
        "uncertainty_note": uncertainty_note,
    }


STRUCTURAL_FAMILY_REFERENCE_SMILES: list[tuple[str, str]] = [
    ("benzene", "c1ccccc1"),
    ("biphenyl", "c1ccccc1-c1ccccc1"),
    ("pyridine", "n1ccccc1"),
    ("thiophene", "c1ccsc1"),
    ("furan", "c1ccoc1"),
    ("naphthalene", "c1ccc2ccccc2c1"),
    ("anthracene", "c1ccc2cc3ccccc3cc2c1"),
    ("phenanthrene", "c1ccc2c(c1)ccc1ccccc12"),
    ("fluorene", "c1ccc2c(c1)Cc1ccccc1-2"),
    ("carbazole", "c1ccc2c(c1)[nH]c1ccccc12"),
]


STRUCTURAL_FUNCTIONAL_GROUP_SMARTS: list[tuple[str, str]] = [
    ("nitrile", "[CX2]#N"),
    ("alkyne", "[CX2]#[CX2]"),
    ("amine", "[NX3;H2,H1;!$(NC=O)]"),
    ("amide", "[NX3][CX3](=[OX1])[#6]"),
    ("imide", "[NX3]([CX3](=[OX1]))[CX3](=[OX1])"),
    ("carboxylic acid", "[CX3](=[OX1])[OX2H1]"),
    ("ester", "[CX3](=[OX1])[OX2][#6]"),
    ("aldehyde", "[CX3H1](=[OX1])[#6]"),
    ("ketone", "[#6][CX3](=[OX1])[#6]"),
    ("ether", "[OD2]([#6])[#6]"),
    ("alcohol", "[OX2H][CX4]"),
    ("halogenated", "[F,Cl,Br,I]"),
]


TEXT_FALLBACK_GROUP_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("porphyrin", ("porphyrin", " tpp ", " h2tpp ", " otpp ")),
    ("phthalocyanine", ("phthalocyanine", " cupc ", " znpc ", " fepc ", " copc ", " nipc ", " h2pc ")),
    ("perylene", ("perylene", "ptcda", "ptcdi")),
    ("pyrene", ("pyrene", "pyrenyl")),
    ("coronene", ("coronene",)),
    ("triphenylene", ("triphenylene",)),
    ("pentacene", ("pentacene",)),
    ("azulene", ("azulene",)),
    ("biphenyl", ("biphenyl", "terphenyl", "quaterphenyl")),
    ("pyridine", ("pyridine", "bipyridine", "pyridyl")),
    ("thiophene", ("thiophene", "bithiophene", "terthiophene", "thienyl")),
    ("furan", ("furan", "furyl")),
    ("naphthalene", ("naphthalene", "naphthyl")),
    ("anthracene", ("anthracene", "anthryl", "anthracenyl")),
    ("phenanthrene", ("phenanthrene",)),
    ("fluorene", ("fluorene", "fluorenyl")),
    ("carbazole", ("carbazole",)),
]


def _dedupe_keep_order_lower(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = (value or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output


def _detect_groups_with_rdkit(smiles: str) -> list[str]:
    try:
        chem = importlib.import_module("rdkit.Chem")
    except Exception:
        return []

    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    found: list[str] = []

    for group_name, reference_smiles in STRUCTURAL_FAMILY_REFERENCE_SMILES:
        reference_mol = chem.MolFromSmiles(reference_smiles)
        if reference_mol is not None and mol.HasSubstructMatch(reference_mol):
            found.append(group_name)

    for group_name, smarts in STRUCTURAL_FUNCTIONAL_GROUP_SMARTS:
        pattern = chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            found.append(group_name)

    return _dedupe_keep_order_lower(found)


def _detect_groups_with_text_fallback(
    iupac_name: str | None,
    synonyms: list[str],
) -> list[str]:
    text_sources = [iupac_name or ""] + [value or "" for value in (synonyms or [])]
    normalized_text = f" {' '.join(text_sources).lower()} "

    found: list[str] = []
    for group_name, keywords in TEXT_FALLBACK_GROUP_PATTERNS:
        for keyword in keywords:
            candidate = keyword.lower()
            if candidate.startswith(" ") or candidate.endswith(" "):
                if candidate in normalized_text:
                    found.append(group_name)
                    break
            else:
                if re.search(rf"\b{re.escape(candidate)}\b", normalized_text):
                    found.append(group_name)
                    break

    return _dedupe_keep_order_lower(found)


def _detect_functional_groups(
    smiles: str,
    iupac_name: str | None,
    synonyms: list[str],
) -> list[str]:
    """优先基于分子结构识别骨架/官能团，名称匹配仅作为 RDKit 缺失时的保底。"""
    structural_matches = _detect_groups_with_rdkit(smiles)
    fallback_matches = _detect_groups_with_text_fallback(iupac_name, synonyms)
    return _dedupe_keep_order_lower(structural_matches + fallback_matches)


# ---------------------------------------------------------------------------
# 阶段一: 查询链生成 + Semantic Scholar 多轮搜索
# ---------------------------------------------------------------------------

def _build_query_chain(
    molecule_info: dict[str, Any],
    surface: str,
) -> list[dict[str, Any]]:
    """
    生成多层次查询链，从精确到宽泛:
      Round 1: 分子名 + 表面 + STM
      Round 2: 基团名 + 表面 + STM
      Round 3: 分子名 + 表面 (无 STM)  /  基团名 + 表面
      Round 4: 分子名 + on-surface  /  基团名 + on-surface
    """
    molecule_name = (molecule_info.get("molecule_name") or "").strip()
    iupac_name = (molecule_info.get("iupac_name") or "").strip()
    synonyms = molecule_info.get("possible_synonyms") or []
    functional_groups = molecule_info.get("functional_groups") or []

    # 收集分子名称变体 (去重)
    mol_names: list[str] = []
    for name in [molecule_name, iupac_name] + list(synonyms[:5]):
        name = (name or "").strip()
        if name and name.lower() not in {n.lower() for n in mol_names}:
            mol_names.append(name)

    # 基团名称
    groups: list[str] = list(dict.fromkeys(functional_groups))  # 保序去重

    surface_clean = surface.strip()

    queries: list[dict[str, Any]] = []

    # === Round 1: 分子名 + 表面 + STM (最精确) ===
    for name in mol_names[:3]:  # 最多用前3个名称变体
        q = f"{name} {surface_clean} STM"
        queries.append({"query": q, "round": 1, "strategy": "mol+surface+STM"})

    # === Round 2: 基团名 + 表面 + STM ===
    for group in groups[:3]:
        q = f"{group} {surface_clean} STM"
        queries.append({"query": q, "round": 2, "strategy": "group+surface+STM"})

    # === Round 3: 分子名/基团 + 表面 (去掉STM, 换 on-surface) ===
    for name in mol_names[:2]:
        q = f"{name} {surface_clean} on-surface"
        queries.append({"query": q, "round": 3, "strategy": "mol+surface+on-surface"})
    for group in groups[:2]:
        q = f"{group} {surface_clean}"
        queries.append({"query": q, "round": 3, "strategy": "group+surface"})

    # === Round 4: 分子名/基团 + on-surface (放宽表面) ===
    for name in mol_names[:2]:
        q = f"{name} on-surface STM"
        queries.append({"query": q, "round": 4, "strategy": "mol+on-surface+STM"})
    for group in groups[:2]:
        q = f"{group} on-surface STM"
        queries.append({"query": q, "round": 4, "strategy": "group+on-surface+STM"})

    # 去重查询 (按 query 文本去重)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in queries:
        key = item["query"].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


def search_papers_semantic_scholar(
    query: str,
    limit: int = 30,
    year_from: int = 2005,
    fields_of_study: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    通过 Semantic Scholar API 的 paper/search (relevance) 端点搜索论文。
    返回标准化的论文列表。
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "paperId,externalIds,title,abstract,year,citationCount,journal,openAccessPdf"
    params: dict[str, Any] = {
        "query": query,
        "limit": min(limit, 100),
        "fields": fields,
    }
    if year_from:
        params["year"] = f"{year_from}-"
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)

    papers: list[dict[str, Any]] = []
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url, params=params)
            if resp.status_code == 429:
                time.sleep(3.0)
                resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        print(f"[WARN] Semantic Scholar search failed for query='{query}': {exc}")
        return []

    for item in (payload.get("data") or []):
        doi = ""
        external_ids = item.get("externalIds") or {}
        if external_ids.get("DOI"):
            doi = external_ids["DOI"]

        pdf_url = ""
        oa_pdf = item.get("openAccessPdf") or {}
        if oa_pdf.get("url"):
            pdf_url = oa_pdf["url"]

        journal_name = ""
        journal_info = item.get("journal") or {}
        if journal_info.get("name"):
            journal_name = journal_info["name"]

        papers.append({
            "paperId": item.get("paperId", ""),
            "title": (item.get("title") or "").strip(),
            "abstract": (item.get("abstract") or "").strip(),
            "doi": doi,
            "year": item.get("year"),
            "citationCount": item.get("citationCount") or 0,
            "journal": journal_name,
            "url": f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
            "pdf_urls": [pdf_url] if pdf_url else [],
        })

    return papers


def search_with_query_chain(
    molecule_info: dict[str, Any],
    surface: str,
    results_per_query: int = SS_RESULTS_PER_QUERY,
    year_from: int = 2005,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    阶段一: 使用查询链在 Semantic Scholar 上进行多轮搜索，
    合并去重，返回候选论文列表。
    """
    query_chain = _build_query_chain(molecule_info, surface)
    all_papers: list[dict[str, Any]] = []
    per_query_stats: list[dict[str, Any]] = []

    for idx, q_info in enumerate(query_chain):
        query_text = q_info["query"]
        round_num = q_info["round"]
        strategy = q_info["strategy"]

        # 控制请求速率，避免被限流
        if idx > 0:
            time.sleep(1.0)

        papers = search_papers_semantic_scholar(
            query=query_text,
            limit=results_per_query,
            year_from=year_from,
        )
        per_query_stats.append({
            "query": query_text,
            "round": round_num,
            "strategy": strategy,
            "results_count": len(papers),
        })
        all_papers.extend(papers)

    # 去重: 优先用 DOI, 其次用 paperId, 最后用 title
    deduped = _dedupe_papers(all_papers)

    # 过滤掉早于 year_from 的论文
    if year_from:
        deduped = [p for p in deduped if (p.get("year") or 9999) >= year_from]

    meta = {
        "query_chain_length": len(query_chain),
        "total_raw_results": len(all_papers),
        "after_dedup": len(deduped),
        "year_from": year_from,
        "per_query_stats": per_query_stats,
    }
    return deduped, meta


def _dedupe_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 DOI > paperId > 标题去重论文列表"""
    seen_doi: set[str] = set()
    seen_pid: set[str] = set()
    seen_title: set[str] = set()
    output: list[dict[str, Any]] = []

    for paper in papers:
        doi = (paper.get("doi") or "").strip().lower()
        pid = (paper.get("paperId") or "").strip()
        title_key = re.sub(r"\s+", " ", (paper.get("title") or "").strip().lower())

        if doi and doi in seen_doi:
            continue
        if pid and pid in seen_pid:
            continue
        if title_key and title_key in seen_title:
            continue

        if doi:
            seen_doi.add(doi)
        if pid:
            seen_pid.add(pid)
        if title_key:
            seen_title.add(title_key)
        output.append(paper)

    return output


# ---------------------------------------------------------------------------
# 阶段二: LLM (Gemini) 对候选论文进行相关性筛选打分
# ---------------------------------------------------------------------------

def screen_papers_with_llm(
    client: genai.Client,
    model_name: str,
    molecule_info: dict[str, Any],
    surface: str,
    candidate_papers: list[dict[str, Any]],
    top_k: int = LLM_SCREENING_TOP_K,
    batch_size: int = 25,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    阶段二: 将候选论文分批发送给 Gemini，让 LLM 对每篇论文打 0-10 的相关性分数。
    返回得分最高的 top_k 篇论文。
    """
    if not candidate_papers:
        return [], {"method": "empty_input", "candidate_count": 0, "selected_count": 0}

    molecule_name = molecule_info.get("molecule_name", "")
    iupac_name = molecule_info.get("iupac_name", "")
    smiles = molecule_info.get("smiles", "")
    functional_groups = molecule_info.get("functional_groups", [])

    all_scored: list[dict[str, Any]] = []

    # 分批处理，每批 batch_size 篇
    for batch_start in range(0, len(candidate_papers), batch_size):
        batch = candidate_papers[batch_start: batch_start + batch_size]
        scored_batch = _score_one_batch(
            client=client,
            model_name=model_name,
            molecule_name=molecule_name,
            iupac_name=iupac_name,
            smiles=smiles,
            functional_groups=functional_groups,
            surface=surface,
            batch=batch,
        )
        all_scored.extend(scored_batch)

        # 请求间隔
        if batch_start + batch_size < len(candidate_papers):
            time.sleep(1.0)

    # 按分数降序排列
    all_scored.sort(key=lambda x: float(x.get("relevance_score", 0)), reverse=True)

    # 取 top_k
    selected = all_scored[:top_k]

    # 添加 rerank_rank
    for idx, paper in enumerate(selected, start=1):
        paper["rerank_rank"] = idx

    meta = {
        "method": "gemini_llm_screening",
        "model": model_name,
        "candidate_count": len(candidate_papers),
        "scored_count": len(all_scored),
        "selected_count": len(selected),
        "batch_size": batch_size,
        "top_k": top_k,
    }
    return selected, meta


def _score_one_batch(
    client: genai.Client,
    model_name: str,
    molecule_name: str,
    iupac_name: str,
    smiles: str,
    functional_groups: list[str],
    surface: str,
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """对一批论文调用 Gemini 进行相关性打分"""
    # 构建论文摘要列表
    paper_entries = []
    for i, paper in enumerate(batch):
        entry = {
            "index": i,
            "title": paper.get("title", ""),
            "abstract": (paper.get("abstract") or "")[:500],  # 截断过长摘要
            "year": paper.get("year"),
            "journal": paper.get("journal", ""),
        }
        paper_entries.append(entry)

    prompt = {
        "task": "Score the relevance of each paper to a specific on-surface STM research topic.",
        "target_molecule": {
            "name": molecule_name,
            "iupac": iupac_name,
            "smiles": smiles,
            "functional_groups": functional_groups,
        },
        "target_surface": surface,
        "scoring_criteria": [
            "10: Paper directly studies the exact same molecule (or its precursor) on the exact target surface using STM/AFM/nc-AFM.",
            "8-9: Paper studies the same molecule on a different metal surface using STM/AFM, or studies a very closely related molecule on the target surface.",
            "6-7: Paper studies a structurally similar molecule (same functional group family) on metal surfaces using STM/AFM.",
            "4-5: Paper is about on-surface synthesis or self-assembly with STM but involves a different class of molecules.",
            "2-3: Paper mentions STM or the molecule but is primarily about DFT calculations, solution chemistry, or electrochemistry.",
            "0-1: Paper is completely unrelated to on-surface STM studies of organic molecules on metal surfaces.",
        ],
        "output_format": {
            "scores": [
                {"index": 0, "score": 0, "reason": "brief reason"},
            ]
        },
        "requirements": [
            "Return ONLY a valid JSON object, no markdown.",
            "Score every paper in the list.",
            "Papers that use scanning tunneling microscopy (STM), atomic force microscopy (AFM), or nc-AFM to image molecules on metal surfaces should score higher.",
            "Papers about purely theoretical/DFT studies without experimental STM should score low.",
            "Papers about electrochemistry, solution chemistry, or biological applications should score very low.",
        ],
        "papers": paper_entries,
    }

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_text(text=json.dumps(prompt, ensure_ascii=False)),
            ],
        )
        response_text = (getattr(response, "text", None) or "").strip() or str(response)
        result = _extract_json_object_safe(response_text)
        scores_list = result.get("scores", [])
    except Exception as exc:
        print(f"[WARN] LLM screening batch failed: {exc}")
        # 失败时给所有论文默认分 5
        scores_list = [{"index": i, "score": 5, "reason": "LLM scoring failed"} for i in range(len(batch))]

    # 将分数合并回论文
    score_map: dict[int, dict[str, Any]] = {}
    for item in scores_list:
        idx = int(item.get("index", -1))
        score_map[idx] = item

    scored_papers: list[dict[str, Any]] = []
    for i, paper in enumerate(batch):
        scored = dict(paper)
        score_info = score_map.get(i, {"score": 5, "reason": "not scored"})
        scored["relevance_score"] = float(score_info.get("score", 5))
        scored["relevance_reason"] = str(score_info.get("reason", ""))
        scored_papers.append(scored)

    return scored_papers


def _extract_json_object_safe(text: str) -> dict[str, Any]:
    """安全地从 LLM 输出中提取 JSON 对象"""
    raw = (text or "").strip()
    if not raw:
        return {}

    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            raw = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        left = raw.find("{")
        right = raw.rfind("}")
        if left == -1 or right == -1 or left >= right:
            return {}
        try:
            return json.loads(raw[left: right + 1])
        except json.JSONDecodeError:
            return {}


# ---------------------------------------------------------------------------
# Crossref: 用于补充 PDF 下载链接
# ---------------------------------------------------------------------------

def enrich_papers_with_crossref_pdf_urls(
    papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """对筛选后的论文，通过 DOI 查 Crossref 补充 PDF 下载链接"""
    enriched = []
    for paper in papers:
        paper_copy = dict(paper)
        doi = (paper_copy.get("doi") or "").strip()
        if doi and not paper_copy.get("pdf_urls"):
            crossref_urls = _get_crossref_pdf_urls_by_doi(doi)
            existing = paper_copy.get("pdf_urls") or []
            paper_copy["pdf_urls"] = _dedupe_urls(existing + crossref_urls)
        enriched.append(paper_copy)
    return enriched


def _get_crossref_pdf_urls_by_doi(doi: str) -> list[str]:
    """通过 DOI 查询 Crossref 获取 PDF 下载链接"""
    url = f"https://api.crossref.org/works/{doi}"
    headers = {
        "User-Agent": f"SPM-nanonis-TCP/1.0 (mailto:{CROSSREF_EMAIL})",
    }
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                return []
            payload = resp.json()
            item = payload.get("message", {})
            return _extract_pdf_urls_from_crossref_item(item)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 以下为原有函数 (未修改)
# ---------------------------------------------------------------------------

def _image_to_vector(image: Image.Image, size: tuple[int, int] = (128, 128)) -> list[float]:
    gray = image.convert("L").resize(size)
    pixels = list(gray.getdata())
    values = [float(v) / 255.0 for v in pixels]
    norm = sum(v * v for v in values) ** 0.5
    if norm <= 1e-12:
        return [0.0 for _ in values]
    return [v / norm for v in values]


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    score = sum(a * b for a, b in zip(vec_a, vec_b))
    return max(-1.0, min(1.0, float(score)))


def _png_bytes_to_data_url(png_bytes: bytes) -> str:
    encoded = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _extract_pdf_urls_from_crossref_item(item: dict[str, Any]) -> list[str]:
    pdf_urls: list[str] = []
    links = item.get("link") or []
    for link in links:
        if not isinstance(link, dict):
            continue
        content_type = str(link.get("content-type", "")).lower()
        url = str(link.get("URL", "")).strip()
        if not url:
            continue
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            pdf_urls.append(url)
    doi = str(item.get("DOI", "")).strip()
    if doi:
        pdf_urls.append(f"https://doi.org/{doi}")
    deduped: list[str] = []
    seen: set[str] = set()
    for u in pdf_urls:
        key = u.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(u)
    return deduped


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "").strip())
    return cleaned[:120] if cleaned else "paper"


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for url in urls:
        value = (url or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


SCI_HUB_MIRRORS = [
    "https://sci-hub.kvnp.top/",
    "https://sci-hub.st/",
    "https://sci-hub.se/",
]


def _fetch_open_access_pdf_urls(doi: str, timeout_seconds: float = 15.0) -> list[str]:
    value = (doi or "").strip()
    if not value:
        return []
    urls: list[str] = []
    for mirror in SCI_HUB_MIRRORS:
        base = mirror if mirror.endswith("/") else mirror + "/"
        urls.append(f"{base}{value}")
    return urls


def _resolve_candidate_pdf_urls_for_paper(paper: dict[str, Any]) -> list[str]:
    existing = [str(u).strip() for u in (paper.get("pdf_urls") or []) if str(u).strip()]
    doi = str(paper.get("doi") or "").strip()
    oa_urls = _fetch_open_access_pdf_urls(doi=doi) if doi else []
    return _dedupe_urls(existing + oa_urls)


def _build_pdf_output_path(cache_dir: Path, paper: dict[str, Any], pdf_url: str) -> Path:
    rank = int(paper.get("rerank_rank") or paper.get("rank") or 0)
    doi = str(paper.get("doi") or "")
    title = str(paper.get("title") or "")
    base_name = _safe_filename(f"{rank:02d}_{doi or title or 'paper'}")
    if not base_name.lower().endswith(".pdf"):
        base_name = f"{base_name}.pdf"
    return cache_dir / base_name


def _extract_pdf_url_from_scihub(html_content: bytes) -> str | None:
    try:
        content = html_content.decode("utf-8", errors="ignore")
        iframe_match = re.search(r'<iframe[^>]+src=["\']([^"\']+)["\']', content, re.IGNORECASE)
        if iframe_match:
            pdf_url = iframe_match.group(1)
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url
            return pdf_url
        location_match = re.search(r'location\.href\s*=\s*["\']([^"\']+)["\']', content)
        if location_match:
            pdf_url = location_match.group(1)
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url
            return pdf_url
        button_match = re.search(r'onclick=\s*["\'][^"\']*location\.href=["\']([^"\']+)["\']', content)
        if button_match:
            pdf_url = button_match.group(1)
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url
            return pdf_url
    except Exception:
        pass
    return None


def _try_download_pdf(
    client: httpx.Client,
    url: str,
    out_path: Path,
    timeout_seconds: float = 25.0,
) -> tuple[bool, str]:
    try:
        resp = client.get(url, timeout=timeout_seconds, follow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:
        return False, f"request_failed: {exc}"
    body = resp.content
    content_type = str(resp.headers.get("Content-Type", "")).lower()
    looks_like_pdf = body.startswith(b"%PDF")
    if "pdf" in content_type or looks_like_pdf:
        try:
            out_path.write_bytes(body)
        except Exception as exc:
            return False, f"write_failed: {exc}"
        return True, "ok"
    if "text/html" in content_type or body.startswith(b"<!DOCTYPE") or body.startswith(b"<html"):
        pdf_url = _extract_pdf_url_from_scihub(body)
        if pdf_url:
            try:
                pdf_resp = client.get(pdf_url, timeout=timeout_seconds, follow_redirects=True)
                pdf_resp.raise_for_status()
                pdf_body = pdf_resp.content
                if not pdf_body.startswith(b"%PDF"):
                    return False, f"extracted_url_not_pdf: {pdf_url[:100]}"
                out_path.write_bytes(pdf_body)
                return True, "ok_via_scihub"
            except Exception as exc:
                return False, f"pdf_download_failed: {exc}"
        else:
            return False, "no_pdf_url_found_in_scihub_page"
    return False, f"unexpected_content_type: {content_type or 'unknown'}"


def download_top_paper_pdfs(
    papers: list[dict[str, Any]],
    cache_dir: str,
    max_papers: int = 6,
    workers: int = DEFAULT_PDF_DOWNLOAD_WORKERS,
    retries_per_url: int = DEFAULT_PDF_DOWNLOAD_RETRIES,
    max_urls_per_paper: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def _download_one_paper(paper: dict[str, Any]) -> dict[str, Any]:
        rank = int(paper.get("rerank_rank") or paper.get("rank") or 0)
        title = paper.get("title", "")
        doi = paper.get("doi", "")
        paper_url = paper.get("url", "")
        urls = _resolve_candidate_pdf_urls_for_paper(paper)[: max(1, max_urls_per_paper)]
        if not urls:
            return {"ok": False, "rank": rank, "title": title, "doi": doi,
                    "paper_url": paper_url, "reason": "no_candidate_pdf_url", "attempts": 0}
        out_path = _build_pdf_output_path(out_dir, paper, urls[0])
        url_errors: list[dict[str, Any]] = []
        attempts = 0
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/129.0.0.0 Safari/537.36"
            )
        }
        with httpx.Client(headers=headers, follow_redirects=True) as dl_client:
            for pdf_url in urls:
                last_status = "not_attempted"
                for attempt in range(max(1, retries_per_url)):
                    attempts += 1
                    ok, status = _try_download_pdf(client=dl_client, url=pdf_url, out_path=out_path)
                    last_status = status
                    if ok:
                        return {"ok": True, "rank": rank, "title": title, "doi": doi,
                                "paper_url": paper_url, "pdf_url": pdf_url,
                                "pdf_path": str(out_path), "download_status": status,
                                "attempts": attempts, "candidate_url_count": len(urls)}
                    if attempt < max(1, retries_per_url) - 1:
                        time.sleep(1.0 + 0.6 * attempt)
                url_errors.append({"url": pdf_url, "status": last_status})
        return {"ok": False, "rank": rank, "title": title, "doi": doi,
                "paper_url": paper_url, "reason": "all_urls_failed",
                "attempts": attempts, "candidate_url_count": len(urls), "url_errors": url_errors}

    candidate_papers = papers[:max_papers]
    max_workers = max(1, min(int(workers), len(candidate_papers) if candidate_papers else 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_one_paper, paper) for paper in candidate_papers]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if bool(result.get("ok")):
                downloaded.append({
                    "rank": result.get("rank", 0), "title": result.get("title", ""),
                    "doi": result.get("doi", ""), "paper_url": result.get("paper_url", ""),
                    "pdf_url": result.get("pdf_url", ""), "pdf_path": result.get("pdf_path", ""),
                    "download_status": result.get("download_status", "ok"),
                    "attempts": int(result.get("attempts", 0)),
                    "candidate_url_count": int(result.get("candidate_url_count", 0)),
                })
            else:
                failures.append(result)
    downloaded.sort(key=lambda item: int(item.get("rank") or 0))
    total_attempts = sum(int(item.get("attempts", 0)) for item in downloaded) + \
                     sum(int(item.get("attempts", 0)) for item in failures)
    return downloaded, {
        "cache_dir": str(out_dir), "candidate_papers": min(len(papers), max_papers),
        "workers": max_workers, "retries_per_url": max(1, retries_per_url),
        "max_urls_per_paper": max(1, max_urls_per_paper),
        "download_attempts": total_attempts,
        "downloaded_count": len(downloaded), "failed_count": len(failures), "failures": failures,
    }


def _extract_pdf_page_images(
    pdf_path: Path, max_pages: int = 12, dpi: int = 140,
) -> list[dict[str, Any]]:
    try:
        import fitz
    except Exception:
        return []
    page_images: list[dict[str, Any]] = []
    zoom = max(1.0, float(dpi) / 72.0)
    matrix = fitz.Matrix(zoom, zoom)
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return page_images
    try:
        for page_index in range(min(len(doc), max_pages)):
            try:
                pix = doc[page_index].get_pixmap(matrix=matrix, alpha=False)
                png_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(png_bytes)).convert("L")
                vector = _image_to_vector(image)
                page_images.append({"page": page_index + 1, "png_bytes": png_bytes, "vector": vector})
            except Exception:
                continue
    finally:
        doc.close()
    return page_images


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        import fitz
    except Exception:
        return 0
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return 0
    try:
        return len(doc)
    except Exception:
        return 0
    finally:
        doc.close()


def match_stm_image_with_pdf_pages(
    stm_image_path: Path, downloaded_pdfs: list[dict[str, Any]],
    max_pages_per_pdf: int = 12, top_k_pages: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not stm_image_path.exists() or not stm_image_path.is_file():
        return [], {"method": "invalid_stm_image", "candidate_pdf_count": len(downloaded_pdfs),
                    "candidate_page_count": 0, "selected_count": 0}
    try:
        stm_image = Image.open(stm_image_path).convert("L")
    except Exception:
        return [], {"method": "stm_open_failed", "candidate_pdf_count": len(downloaded_pdfs),
                    "candidate_page_count": 0, "selected_count": 0}
    target_vector = _image_to_vector(stm_image)
    candidates: list[dict[str, Any]] = []
    per_pdf_metrics: list[dict[str, Any]] = []
    targeted_page_count = 0
    extracted_page_count = 0
    opened_pdf_count = 0
    for paper in downloaded_pdfs:
        pdf_path = Path(str(paper.get("pdf_path") or ""))
        if not pdf_path.exists() or not pdf_path.is_file():
            per_pdf_metrics.append({"pdf_path": str(pdf_path), "title": paper.get("title", ""),
                                    "rank": paper.get("rank", 0), "targeted_pages": 0,
                                    "extracted_pages": 0, "success_rate": 0.0, "status": "missing_file"})
            continue
        page_count = _pdf_page_count(pdf_path)
        target_pages = min(max(0, page_count), max_pages_per_pdf)
        targeted_page_count += target_pages
        page_images = _extract_pdf_page_images(pdf_path=pdf_path, max_pages=max_pages_per_pdf)
        if page_images:
            opened_pdf_count += 1
        extracted = len(page_images)
        extracted_page_count += extracted
        rate = float(extracted) / float(target_pages) if target_pages > 0 else 0.0
        per_pdf_metrics.append({"pdf_path": str(pdf_path), "title": paper.get("title", ""),
                                "rank": paper.get("rank", 0), "targeted_pages": target_pages,
                                "extracted_pages": extracted, "success_rate": round(rate, 4),
                                "status": "ok" if extracted > 0 else "no_page_extracted"})
        for item in page_images:
            score = _cosine_similarity(target_vector, item["vector"])
            candidates.append({"rank": paper.get("rank", 0), "title": paper.get("title", ""),
                                "doi": paper.get("doi", ""), "pdf_url": paper.get("pdf_url", ""),
                                "pdf_path": str(pdf_path), "page": int(item["page"]),
                                "similarity": float(score), "png_bytes": item["png_bytes"]})
    candidates.sort(key=lambda row: row["similarity"], reverse=True)
    selected = candidates[:max(1, top_k_pages)] if candidates else []
    matches: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        match = dict(row)
        match["page_match_rank"] = idx
        match["image_data_url"] = _png_bytes_to_data_url(row["png_bytes"])
        match.pop("png_bytes", None)
        matches.append(match)
    global_rate = float(extracted_page_count) / float(targeted_page_count) if targeted_page_count > 0 else 0.0
    return matches, {
        "method": "cosine_on_pdf_page_images", "candidate_pdf_count": len(downloaded_pdfs),
        "opened_pdf_count": opened_pdf_count, "candidate_page_count": len(candidates),
        "selected_count": len(matches), "targeted_page_count": targeted_page_count,
        "extracted_page_count": extracted_page_count,
        "pdf_to_image_success_rate": round(global_rate, 4), "per_pdf_metrics": per_pdf_metrics,
    }


def set_proxy_from_local_default() -> None:
    proxy_url = "socks5h://127.0.0.1:7890"
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["ALL_PROXY"] = proxy_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Gemini + online paper search to judge molecule/STM/surface matching."
    )
    parser.add_argument("--structure-image", "--structure-path", type=str,
                        default=DEFAULT_STRUCTURE_IMAGE_PATH, help="Path to molecule structure image file.")
    parser.add_argument("--stm-image", "--stm-path", type=str,
                        default=DEFAULT_STM_IMAGE_PATH, help="Path to STM scan image file.")
    parser.add_argument("--surface", type=str, default=DEFAULT_SURFACE, help="Target surface type, e.g. Au(111).")
    parser.add_argument("--model", type=str, default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                        help="Gemini model name.")
    parser.add_argument("--out", type=str, default="", help="Optional output JSON path.")
    parser.add_argument("--reranker-model", type=str, default=DEFAULT_RERANKER_MODEL_PATH,
                        help="Local reranker model path (legacy, not used in new pipeline).")
    return parser.parse_args()


def read_image_as_part(image_path: Path) -> types.Part:
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    image_bytes = image_path.read_bytes()
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


def extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Model returned empty text")
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            raw = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        left = raw.find("{")
        right = raw.rfind("}")
        if left == -1 or right == -1 or left >= right:
            raise
        return json.loads(raw[left: right + 1])


def strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").replace("\n", " ").strip()


def judge_match_with_gemini(
    client: genai.Client, model_name: str, structure_part: types.Part, stm_part: types.Part,
    surface: str, molecule_info: dict[str, Any],
    top_papers: list[dict[str, Any]], top_pdf_page_matches: list[dict[str, Any]],
) -> dict[str, Any]:
    analysis_prompt = {
        "task": "Assess whether provided STM image and molecule structure are matched on the same target metal surface.",
        "target_surface": surface,
        "inputs": {
            "molecule_info": molecule_info,
            "top_papers": top_papers,
            "top_pdf_page_matches": [
                {"page_match_rank": item.get("page_match_rank", 0),
                 "paper_rank": item.get("rank", 0), "title": item.get("title", ""),
                 "doi": item.get("doi", ""), "page": item.get("page", 0),
                 "similarity": item.get("similarity", 0.0)}
                for item in top_pdf_page_matches
            ],
        },
        "decision_definition": {
            "match": "The STM pattern and molecule identity are consistent with literature evidence on the target surface.",
            "mismatch": "Evidence suggests inconsistency with target surface or molecule assignment.",
            "uncertain": "Insufficient or conflicting evidence.",
        },
        "required_output_fields": {
            "target_surface": "string", "judgement": "match|mismatch|uncertain",
            "overall_match": True, "confidence": 0.0, "evidence_summary": "string",
            "supporting_papers": [{"rank": 0, "title": "string", "doi": "string",
                                   "why_relevant": "string", "supports_match": True}],
            "possible_error_sources": ["string"], "next_steps": ["string"],
        },
        "requirements": [
            "Use only provided papers and images as evidence.",
            "supporting_papers should cite rank/title/doi from top_papers list.",
            "If literature PDF page images are provided, prioritize visual evidence from those pages when deciding STM match.",
            "Return JSON only.",
        ],
    }
    contents: list[types.Part] = [
        types.Part.from_text(text=json.dumps(analysis_prompt, ensure_ascii=False, indent=2)),
        types.Part.from_text(text="[Image A] Molecule structure diagram"), structure_part,
        types.Part.from_text(text="[Image B] STM scan image"), stm_part,
    ]
    if top_pdf_page_matches:
        contents.append(types.Part.from_text(
            text="[Image C..] Literature PDF pages most similar to current STM. Use as visual references."))
        for item in top_pdf_page_matches:
            contents.append(types.Part.from_text(
                text=(f"literature_page_match rank={item.get('page_match_rank', 0)}, "
                      f"paper_rank={item.get('rank', 0)}, page={item.get('page', 0)}, "
                      f"similarity={float(item.get('similarity', 0.0)):.4f}, "
                      f"title={item.get('title', '')}, doi={item.get('doi', '')}")))
            data_url = str(item.get("image_data_url") or "")
            if data_url.startswith("data:image/png;base64,"):
                encoded = data_url.split(",", 1)[1]
                try:
                    png_bytes = base64.b64decode(encoded)
                    contents.append(types.Part.from_bytes(data=png_bytes, mime_type="image/png"))
                except Exception:
                    continue
    response = client.models.generate_content(model=model_name, contents=contents)
    response_text = (getattr(response, "text", None) or "").strip() or str(response)
    return extract_json_object(response_text)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_match_pipeline(
    structure_path: Path,
    stm_path: Path,
    surface: str,
    model_name: str,
    reranker_model_path: str = DEFAULT_RERANKER_MODEL_PATH,
    pdf_cache_dir: str = DEFAULT_PDF_CACHE_DIR,
) -> dict[str, Any]:
    # Step 1: 分子识别
    molecule_info = identify_molecule_from_structure_decimer(
        structure_image_path=structure_path,
    )

    # Step 2: Gemini client (用于 LLM 筛选和最终判断)
    client = genai.Client(api_key=API_KEY)
    structure_part = read_image_as_part(structure_path)
    stm_part = read_image_as_part(stm_path)

    # Step 3: 阶段一 — 查询链 + Semantic Scholar 多轮搜索
    candidate_papers, search_meta = search_with_query_chain(
        molecule_info=molecule_info,
        surface=surface,
        results_per_query=SS_RESULTS_PER_QUERY,
        year_from=2005,
    )

    # Step 4: 阶段二 — LLM 筛选打分
    top_papers, screening_meta = screen_papers_with_llm(
        client=client,
        model_name=model_name,
        molecule_info=molecule_info,
        surface=surface,
        candidate_papers=candidate_papers,
        top_k=LLM_SCREENING_TOP_K,
    )

    # Step 5: 用 Crossref 补充 PDF 下载链接
    top_papers = enrich_papers_with_crossref_pdf_urls(top_papers)

    # Step 6: PDF 下载
    downloaded_pdfs, literature_pdf_download_meta = download_top_paper_pdfs(
        papers=top_papers,
        cache_dir=pdf_cache_dir,
        max_papers=6,
    )

    # Step 7: STM 图像与 PDF 页面匹配
    top_pdf_page_matches, literature_pdf_image_match_meta = match_stm_image_with_pdf_pages(
        stm_image_path=stm_path,
        downloaded_pdfs=downloaded_pdfs,
        max_pages_per_pdf=12,
        top_k_pages=5,
    )

    # Step 8: 最终判断
    judgement = judge_match_with_gemini(
        client=client,
        model_name=model_name,
        structure_part=structure_part,
        stm_part=stm_part,
        surface=surface,
        molecule_info=molecule_info,
        top_papers=top_papers,
        top_pdf_page_matches=top_pdf_page_matches,
    )

    return {
        "input": {
            "structure_image": str(structure_path),
            "stm_image": str(stm_path),
            "surface": surface,
            "model": model_name,
        },
        "molecule_identification": molecule_info,
        "search_chain_meta": search_meta,
        "llm_screening_meta": screening_meta,
        "top_papers": top_papers,
        "literature_pdf_download_meta": literature_pdf_download_meta,
        "downloaded_pdfs": downloaded_pdfs,
        "literature_pdf_image_match_meta": literature_pdf_image_match_meta,
        "top_pdf_page_matches": top_pdf_page_matches,
        "match_judgement": judgement,
    }


def main() -> None:
    args = parse_args()
    set_proxy_from_local_default()

    if not args.structure_image:
        raise ValueError(
            "No structure image found. Set --structure-image or place at least one image in ./molecules."
        )
    if not args.stm_image:
        raise ValueError("No STM image found. Set --stm-image or export DEFAULT_STM_IMAGE_PATH.")

    structure_path = Path(args.structure_image)
    stm_path = Path(args.stm_image)
    surface = args.surface.strip()
    model_name = args.model

    final_result = run_match_pipeline(
        structure_path=structure_path,
        stm_path=stm_path,
        surface=surface,
        model_name=model_name,
        pdf_cache_dir=DEFAULT_PDF_CACHE_DIR,
    )

    literature_pdf_image_match_meta = final_result.get("literature_pdf_image_match_meta", {})
    print(
        "[QUANT_METRIC_TEST] pdf_to_image_success_rate="
        + str(literature_pdf_image_match_meta.get("pdf_to_image_success_rate", 0.0))
        + ", targeted_pages="
        + str(literature_pdf_image_match_meta.get("targeted_page_count", 0))
        + ", extracted_pages="
        + str(literature_pdf_image_match_meta.get("extracted_page_count", 0))
        + ", candidate_pdf_count="
        + str(literature_pdf_image_match_meta.get("candidate_pdf_count", 0))
        + ", opened_pdf_count="
        + str(literature_pdf_image_match_meta.get("opened_pdf_count", 0))
    )

    console_result = dict(final_result)
    console_result.pop("search_chain_meta", None)
    result_text = json.dumps(console_result, ensure_ascii=False, indent=2)
    print(result_text)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()