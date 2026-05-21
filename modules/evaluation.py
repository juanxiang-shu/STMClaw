import base64
import io
import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from google import genai
from google.genai import types
from PIL import Image


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
SUPPORTED_PDF_EXTENSIONS = ('.pdf',)


def set_proxy_from_local_default() -> None:
    proxy_url = 'socks5h://127.0.0.1:7890'
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    os.environ['ALL_PROXY'] = proxy_url


def _data_url_to_gemini_part(data_url: str):
    value = (data_url or '').strip()
    if not value.startswith('data:') or ',' not in value:
        return None
    try:
        header, encoded = value.split(',', 1)
        mime_type = 'image/png'
        if ';base64' in header and ':' in header:
            mime_type = header.split(':', 1)[1].split(';', 1)[0] or 'image/png'
        image_bytes = base64.b64decode(encoded)
        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    except Exception:
        return None


def _openai_style_content_to_gemini_parts(content_blocks):
    parts = []
    for block in content_blocks or []:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get('type', '')).strip().lower()
        if block_type == 'text':
            text = str(block.get('text', '') or '')
            if text:
                parts.append(types.Part.from_text(text=text))
        elif block_type == 'image_url':
            image_url = block.get('image_url')
            if isinstance(image_url, dict):
                url = str(image_url.get('url', '') or '')
                part = _data_url_to_gemini_part(url)
                if part is not None:
                    parts.append(part)
    return parts


def _load_image(image_data):
    if isinstance(image_data, str):
        return Image.open(image_data).convert('L')
    if isinstance(image_data, np.ndarray):
        return Image.fromarray(image_data).convert('L')
    raise TypeError('image_data must be image path or numpy array')


def _image_to_data_url(image, size=(208, 208)):
    image = image.resize(size)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{base64_image}'


def _safe_parse_json_block(text):
    if isinstance(text, list):
        text = ''.join([str(x) for x in text])
    if not isinstance(text, str):
        return {}

    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    fence_match = re.search(r'\{[\s\S]*\}', text)
    if fence_match:
        try:
            return json.loads(fence_match.group(0))
        except Exception:
            pass

    return {}


def _label_from_probability(probability, threshold):
    return 'good' if probability > threshold else 'bad'


def _vector_from_image(image, size=(64, 64)):
    arr = np.asarray(image.resize(size), dtype=np.float32)
    vec = arr.flatten()
    vec = vec - float(np.mean(vec))
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec = vec / norm
    return vec


def _cosine_similarity(vec_a, vec_b):
    return float(np.dot(vec_a, vec_b))


def _iter_image_files(folder_path, max_count):
    if not os.path.isdir(folder_path):
        return []

    files = []
    for name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, name)
        if os.path.isfile(file_path) and name.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
            files.append(file_path)
        if len(files) >= max_count:
            break
    return files


def _iter_pdf_files(folder_path, max_count):
    if not os.path.isdir(folder_path):
        return []

    files = []
    for name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, name)
        if os.path.isfile(file_path) and name.lower().endswith(SUPPORTED_PDF_EXTENSIONS):
            files.append(file_path)
        if len(files) >= max_count:
            break
    return files


@lru_cache(maxsize=1)
def _build_literature_page_index(max_pdf_count=10, max_pages_per_pdf=10, dpi=150):
    """
    Build a lightweight visual index from ./literature PDF pages.
    Returns list of dict items with keys: pdf_path, page, data_url, vector.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return []

    root_dir = os.path.dirname(__file__)
    literature_dir = os.path.normpath(os.path.join(root_dir, '..', 'literature'))
    pdf_files = _iter_pdf_files(literature_dir, max_pdf_count)
    if not pdf_files:
        return []

    index = []
    zoom = max(1.0, float(dpi) / 72.0)
    matrix = fitz.Matrix(zoom, zoom)

    for pdf_path in pdf_files:
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            continue

        try:
            page_count = min(len(doc), max_pages_per_pdf)
            for page_idx in range(page_count):
                try:
                    pix = doc[page_idx].get_pixmap(matrix=matrix, alpha=False)
                    png_bytes = pix.tobytes('png')
                    image = Image.open(io.BytesIO(png_bytes)).convert('L')
                    page_data_url = _image_to_data_url(image, size=(208, 208))
                    page_vector = _vector_from_image(image, size=(64, 64))
                    index.append(
                        {
                            'pdf_path': pdf_path,
                            'page': page_idx + 1,
                            'data_url': page_data_url,
                            'vector': page_vector,
                        }
                    )
                except Exception:
                    continue
        finally:
            doc.close()

    return index


def _build_literature_retrieval_content(target_image, top_k=4):
    """Retrieve top-k visually similar pages from literature index for prompt grounding."""
    index = _build_literature_page_index()
    if not index:
        empty_content: List[Dict[str, object]] = [
            {
                'type': 'text',
                'text': 'No retrievable PDF pages found in ./literature, continue with local examples only.'
            }
        ]
        return empty_content

    target_vector = _vector_from_image(target_image, size=(64, 64))
    scored_pages = []
    for item in index:
        score = _cosine_similarity(target_vector, item['vector'])
        scored_pages.append((score, item))

    scored_pages.sort(key=lambda x: x[0], reverse=True)
    top_items = scored_pages[:max(1, top_k)]

    content: List[Dict[str, object]] = [
        {
            'type': 'text',
            'text': 'Retrieved visual references from ./literature PDFs (most similar pages first):'
        }
    ]

    for rank, (score, item) in enumerate(top_items, start=1):
        pdf_name = os.path.basename(item['pdf_path'])
        page = item['page']
        content.append(
            {
                'type': 'text',
                'text': f'literature match #{rank}: {pdf_name}, page {page}, similarity={score:.3f}'
            }
        )
        content.append({'type': 'image_url', 'image_url': {'url': item['data_url']}})

    return content


@lru_cache(maxsize=1)
def _build_fewshot_examples_content(max_per_class=12):
    """Build a reusable multimodal few-shot block from EvaluationCNN/good and EvaluationCNN/bad."""
    root_dir = os.path.dirname(__file__)
    good_dir = os.path.join(root_dir, 'learning', 'good')
    bad_dir = os.path.join(root_dir, 'learning', 'bad')

    content: List[Dict[str, object]] = [
        {
            'type': 'text',
            'text': (
                'Reference examples for STM image quality classification are provided below. '
                'Examples under [GOOD EXAMPLES] are the positive standard; '
                'examples under [BAD EXAMPLES] are the negative standard. '
                'Use these few-shot examples as the primary decision basis and keep your own free-form assumptions minimal.'
            )
        }
    ]

    good_files = _iter_image_files(good_dir, max_per_class)
    bad_files = _iter_image_files(bad_dir, max_per_class)

    if good_files:
        content.append({'type': 'text', 'text': '[GOOD EXAMPLES]'})
        for idx, file_path in enumerate(good_files, start=1):
            image = _load_image(file_path)
            content.append({'type': 'text', 'text': f'good example #{idx}'})
            content.append({'type': 'image_url', 'image_url': {'url': _image_to_data_url(image, size=(208, 208))}})

    if bad_files:
        content.append({'type': 'text', 'text': '[BAD EXAMPLES]'})
        for idx, file_path in enumerate(bad_files, start=1):
            image = _load_image(file_path)
            content.append({'type': 'text', 'text': f'bad example #{idx}'})
            content.append({'type': 'image_url', 'image_url': {'url': _image_to_data_url(image, size=(208, 208))}})

    if not good_files and not bad_files:
        content.append(
            {
                'type': 'text',
                'text': (
                    'No local few-shot examples were found at ./Evaluation/learning/good or ./Evaluation/learning/bad. '
                    'Proceed with generic STM quality evaluation.'
                )
            }
        )

    return content


def _extract_probability(text):
    if isinstance(text, list):
        text = ''.join([str(x) for x in text])
    if not isinstance(text, str):
        return 0.0

    text = text.strip()

    try:
        data = json.loads(text)
        value = float(data.get('good_probability', 0.0))
        return max(0.0, min(1.0, value))
    except Exception:
        pass

    fence_match = re.search(r'\{[\s\S]*\}', text)
    if fence_match:
        try:
            data = json.loads(fence_match.group(0))
            value = float(data.get('good_probability', 0.0))
            return max(0.0, min(1.0, value))
        except Exception:
            pass

    num_match = re.search(r'(0(?:\.\d+)?|1(?:\.0+)?)', text)
    if num_match:
        value = float(num_match.group(1))
        return max(0.0, min(1.0, value))

    return 0.0


def _extract_structured_diagnosis(text, probability, threshold):
    data = _safe_parse_json_block(text)
    if not data:
        fallback_label = _label_from_probability(probability, threshold)
        return {
            'good_probability': probability,
            'label': fallback_label,
            'confidence': probability,
            'defect_tags': [],
            'severity': 'unknown',
            'recommended_action_bias': '' if fallback_label == 'good' else 'pulse_first',
            'rationale': 'fallback parser used',
        }

    parsed_probability = data.get('good_probability', probability)
    try:
        parsed_probability = float(parsed_probability)
    except Exception:
        parsed_probability = probability
    parsed_probability = max(0.0, min(1.0, parsed_probability))

    label = str(data.get('label', _label_from_probability(parsed_probability, threshold))).lower().strip()
    if label not in ('good', 'bad'):
        label = _label_from_probability(parsed_probability, threshold)

    confidence = data.get('confidence', parsed_probability)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = parsed_probability
    confidence = max(0.0, min(1.0, confidence))

    defect_tags = data.get('defect_tags', [])
    if not isinstance(defect_tags, list):
        defect_tags = []
    defect_tags = [str(x) for x in defect_tags]

    severity = str(data.get('severity', 'unknown')).strip().lower()
    if severity not in ('mild', 'moderate', 'severe', 'unknown'):
        severity = 'unknown'

    recommended_action_bias = str(data.get('recommended_action_bias', 'pulse_first')).strip().lower()
    if label == 'good':
        recommended_action_bias = ''
    elif severity == 'severe':
        recommended_action_bias = 'tipshaper_first'
    elif recommended_action_bias not in ('pulse_first', 'tipshaper_first', 'either'):
        recommended_action_bias = 'pulse_first'

    return {
        'good_probability': parsed_probability,
        'label': label,
        'confidence': confidence,
        'defect_tags': defect_tags,
        'severity': severity,
        'recommended_action_bias': recommended_action_bias,
        'rationale': str(data.get('rationale', '')),
    }


def evaluate_image_quality(image_data, model_path=None, return_prompt_debug=False):
    """
    Enhanced Evaluation Agent:
    - Uses local few-shot examples (./Evaluation/good and ./Evaluation/bad)
    - Retrieves similar pages from ./literature PDFs as additional grounding
    - Returns structured diagnosis for downstream Conditioning Agent
    """
    api_key = GEMINI_API_KEY
    if isinstance(model_path, str) and model_path.startswith('AIza'):
        api_key = model_path

    if not api_key:
        raise RuntimeError('GEMINI_API_KEY is empty in code.')

    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    quality_threshold = float(os.getenv('EVAL_GOOD_THRESHOLD', '0.7'))
    set_proxy_from_local_default()

    image = _load_image(image_data)
    image_data_url = _image_to_data_url(image)
    fewshot_content = _build_fewshot_examples_content(max_per_class=13)
    # literature_content = _build_literature_retrieval_content(image, top_k=4)

    user_content: List[Dict[str, object]] = list(fewshot_content)
    # user_content.extend(literature_content)
    user_content.append(
        {
            'type': 'text',
            'text': (
                'Now evaluate the target STM scan image. '
                'Prioritize similarity to the provided few-shot examples as the main criterion. '
                'if the target is broadly consistent with good examples in overall morphology, continuity, and interpretability, '
                'prefer classifying it as good even when there are minor imperfections or mild local defects. '
                'Reserve bad mainly for clear, repeated, or analysis-blocking defects. '
                'Avoid making extra assumptions beyond the examples. '
                'Return ONLY JSON with this schema: '
                '{'
                '"good_probability": <0..1>, '
                '"label": "good" or "bad", '
                '"confidence": <0..1>, '
                '"defect_tags": ["double_tip"|"drift"|"noise"|"contamination"|"lost_contact"|"blur"|"artifact"], '
                '"severity": "mild"|"moderate"|"severe", '
                '"recommended_action_bias": "pulse_first"|"tipshaper_first"|"either", '
                '"rationale": "short reason"'
                '}. '
                'In rationale, briefly explain which few-shot patterns the target most resembles.'
            )
        }
    )

    user_content.append({'type': 'image_url', 'image_url': {'url': image_data_url}})

    try:
        client = genai.Client(api_key=api_key)
        gemini_parts = [
            types.Part.from_text(
                text=(
                    'You are an STM scan image quality evaluator. '
                    'Classify mainly by matching the target to provided few-shot good/bad examples. '
                    'Apply a practical boundary: tolerable small artifacts should not dominate '
                    'the decision when the image remains structurally useful and close to good references. '
                    'Use bad only when defects are substantial, persistent, or clearly dominant. '
                    'Do not over-rely on external heuristics or speculative reasoning. '
                    'Provide structured diagnosis in JSON only.'
                )
            )
        ]
        gemini_parts.extend(_openai_style_content_to_gemini_parts(user_content))
        response = client.models.generate_content(
            model=model_name,
            contents=gemini_parts,
        )

        content = (getattr(response, 'text', None) or '').strip() or str(response)
        probability = _extract_probability(content)
        diagnosis = _extract_structured_diagnosis(content, probability, quality_threshold)
        # if return_prompt_debug:
        #     diagnosis['prompt_debug'] = {
        #         'fewshot_blocks': len(fewshot_content),
        #         'literature_blocks': len(literature_content),
        #     }

        if return_prompt_debug:
            diagnosis['prompt_debug'] = {
                'fewshot_blocks': len(fewshot_content),
                # 'literature_blocks': len(literature_content),
                'last_instruction_text': user_content[-2].get('text', '')
            }
        return diagnosis
    except Exception as exc:
        print(f'[Gemini Evaluation] request failed: {exc}')
        fallback_probability = 0.0
        return {
            'good_probability': fallback_probability,
            'label': _label_from_probability(fallback_probability, quality_threshold),
            'confidence': 0.0,
            'defect_tags': [],
            'severity': 'unknown',
            'recommended_action_bias': 'pulse_first',
            'rationale': f'request_failed: {exc}',
        }



def predict_image_quality(image_data, model_path=None):
    diagnosis = evaluate_image_quality(image_data, model_path=model_path, return_prompt_debug=True)
    return float(diagnosis.get('good_probability', 0.0))



if __name__ =='__main__':
    # Example usage:
    model_path = None
    image_path = r"E:\1-Work\8-STM+LLM\test.png"
    result = predict_image_quality(image_path, model_path)
    print('This is a {} picture.'.format('good' if result >= 0.5 else 'bad'))
    print('Good probability: {:.2f}'.format(result))
    print('Detailed diagnosis:', evaluate_image_quality(image_path, model_path=model_path, return_prompt_debug=True))
