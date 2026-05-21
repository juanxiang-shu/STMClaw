# pyright: reportAttributeAccessIssue=false

import ast
import json
import os
import re
import time
from urllib import error

import numpy as np
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')


def set_proxy_from_local_default() -> None:
    proxy_url = 'socks5h://127.0.0.1:7890'
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    os.environ['ALL_PROXY'] = proxy_url


class NavigatorPathAgentMixin:
    """Planning Agent methods extracted from Mustard_AI_Nanonis."""

    def configure_navigator_agent(self, mode='custom', step_pix=None, custom_points=None, start_points=None):
        """Configure Planning Agent path strategy.

        mode: 'custom' (default). Legacy modes 'spiral' and 'snake' are deprecated and mapped to 'custom'.
        step_pix: pixel spacing for path generation.
        custom_points: list of (x_pix, y_pix), only used when mode='custom'.
        start_points: optional first-two-point hint [(x1,y1), (x2,y2)] for the planner.
        """
        mode = str(mode).strip().lower()
        if mode in ('spiral', 'snake'):
            print(f"[Planning Agent] legacy mode='{mode}' is deprecated; using 'custom' instead.")
            mode = 'custom'
        if mode not in ('custom',):
            raise ValueError("Unsupported navigator mode. Use 'custom'.")

        self.navigator_mode = mode
        if step_pix is not None:
            self.navigator_step_pix = max(1, int(step_pix))
        elif self.navigator_step_pix is None:
            self.navigator_step_pix = max(1, int(self.scan_square_edge))

        if mode == 'custom':
            if not custom_points:
                raise ValueError("custom_points is required when mode='custom'.")
            self.navigator_custom_points = self._sanitize_waypoints(custom_points)
            if not self.navigator_custom_points:
                raise ValueError('No valid custom waypoints within boundary.')
        elif custom_points is not None:
            self.navigator_custom_points = self._sanitize_waypoints(custom_points)

        if start_points is None:
            self.navigator_start_points = []
        else:
            self.navigator_start_points = self._sanitize_waypoints(start_points)[:2]

        self.navigator_reset()
        print(
            f"[Planning Agent] mode={self.navigator_mode}, "
            f"step_pix={self.navigator_step_pix}, custom_points={len(self.navigator_custom_points)}, "
            f"start_points={self.navigator_start_points}"
        )

    def _safe_parse_json_block(self, text):
        """Parse the first JSON object from raw model output."""
        if isinstance(text, list):
            text = ''.join([str(x) for x in text])
        if not isinstance(text, str):
            return {}

        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {}

    def _navigator_system_prompt(self):
        """System prompt: coordinate definition, constraints, and output schema."""
        left, right, top, bottom = self._navigator_scan_bounds()
        return (
            "You are Planning Agent for STM scan path planning. "
            "You must output ONLY JSON and follow hard constraints. "
            "Coordinate definition: use image pixel coordinates (x_pix, y_pix). "
            f"Plane pixel size is {self.plane_size}x{self.plane_size}. "
            f"Scanable boundary is rectangle: x in [{left}, {right}], y in [{top}, {bottom}]. "
            f"The scan_edge is {self.Scan_edge}. "
            "Please determine the coordinates of the first scan point according to the scan path requested by the user. "
            "If user asks for unsafe/out-of-bound path, adjust to valid in-bound path. "
            "The Planning Agent interprets path semantics such as spiral-like or serpentine shapes; custom waypoints are also allowed. "
            "Return JSON schema exactly: "
            "{"
            "\"path_type\": \"spiral\"|\"snake\"|\"custom\", "
            "\"step_pix\": <int or null>, "
            "\"waypoints\": [{\"x_pix\": int, \"y_pix\": int}, ...], "
            "\"reason\": \"short explanation\""
            "}. "
            "Rules: "
            "- If user explicitly specifies step length, return step_pix as that exact pixel step. "
            "- If user does NOT specify step length, use default step equal to scan_square_edge. "
            "- waypoints can be empty because runtime may generate dense coverage path from starting points and step. "
            "- If the user specifies a starting point (e.g., 'top-left corner'), the first element in 'waypoints' should be that starting coordinate. For a snake path, the rest of the waypoints can be empty, as they will be generated. "
            "- Never output text outside JSON."
        )

    def _navigator_user_prompt(self, user_instruction):
        """User prompt: pass user's natural-language path description."""
        return (
            "User path requirement (natural language):\n"
            f"{user_instruction}\n"
            "Plan an executable scan path under the system constraints and output JSON only."
        )

    def _path_system_prompt(self):
        """System prompt for Path Agent: focus on path-shape semantics to concrete waypoints."""
        left, right, top, bottom = self._navigator_scan_bounds()
        return (
            "You are Planning Agent for STM path-shape understanding and coordinate generation. "
            "You must output ONLY JSON. "
            "Coordinate definition: image pixel coordinates (x_pix, y_pix). "
            f"Plane pixel size is {self.plane_size}x{self.plane_size}. "
            f"Scanable boundary is rectangle: x in [{left}, {right}], y in [{top}, {bottom}]. "
            f"Scan_edge is {self.Scan_edge}. "
            "Input includes first two start points from Starting Planning Agent. "
            "Your task: infer requested path shape (especially Chinese words like 回型/回字形), "
            "then generate safe in-bound waypoints that follow that shape and cover the scanable area. "
            "Return JSON schema exactly: "
            "{"
            "\"path_type\": \"snake\"|\"spiral\"|\"custom\", "
            "\"step_pix\": <int or null>, "
            "\"waypoints\": [{\"x_pix\": int, \"y_pix\": int}, ...], "
            "\"reason\": \"short explanation\""
            "}. "
            "Rules: "
            "- First waypoint should match the first start point when possible. "
            "- Keep all points in boundary. "
            "- Never output text outside JSON."
        )

    def _path_codegen_system_prompt(self):
        """System prompt for Path Agent code generation mode."""
        left, right, top, bottom = self._navigator_scan_bounds()
        return (
            "You are Planning Agent for STM. Output ONLY JSON. "
            "Generate Python code that computes path coordinates from runtime context. "
            "Coordinate definition: image pixel (x_pix, y_pix). "
            f"Boundary: x in [{left}, {right}], y in [{top}, {bottom}]. "
            "Code must define exactly one function: generate_path(context). "
            "Function must return list of (x_pix, y_pix). "
            "Context fields: plane_size, scan_edge, left, right, top, bottom, step_pix, start_points. "
            "Keep all returned points in boundary. "
            "Return JSON schema exactly: "
            "{"
            "\"path_type\": \"custom\"|\"snake\"|\"spiral\", "
            "\"step_pix\": <int or null>, "
            "\"python_code\": \"string, must include def generate_path(context): ...\", "
            "\"reason\": \"short explanation\""
            "}. "
            "Do not include markdown code fences. Never output text outside JSON."
        )

    def _path_user_prompt(self, user_instruction, start_points):
        """User prompt for Path Agent with runtime context and starting points."""
        left, right, top, bottom = self._navigator_scan_bounds()
        return (
            "User path requirement (natural language):\n"
            f"{user_instruction}\n"
            "Runtime context:\n"
            f"- plane_size: {self.plane_size}\n"
            f"- Scan_edge: {self.Scan_edge}\n"
            f"- scanable boundary: x in [{left}, {right}], y in [{top}, {bottom}]\n"
            f"- start_points(first two): {start_points}\n"
            "Return path_type + step_pix + waypoints as JSON only."
        )

    def _path_codegen_user_prompt(self, user_instruction, start_points, step_pix):
        """User prompt for Path Agent code generation mode."""
        left, right, top, bottom = self._navigator_scan_bounds()
        return (
            "User path requirement (natural language):\n"
            f"{user_instruction}\n"
            "Runtime context for code execution:\n"
            f"- plane_size: {self.plane_size}\n"
            f"- scan_edge: {self.Scan_edge}\n"
            f"- left: {left}, right: {right}, top: {top}, bottom: {bottom}\n"
            f"- step_pix: {int(step_pix)}\n"
            f"- start_points(first two): {start_points}\n"
            "Return JSON with python_code only, no markdown fences."
        )

    def _starting_system_prompt(self):
        """System prompt for determining first two scan points from user intent."""
        left, right, top, bottom = self._navigator_scan_bounds()
        return (
            "You are Starting Planning Agent for STM scan initialization. "
            "You must output ONLY JSON. "
            "Coordinate definition: image pixel coordinates (x_pix, y_pix). "
            f"Scan boundary: x in [{left}, {right}], y in [{top}, {bottom}]. "
            f"Scan step in pixel is typically scan_square_edge={self.scan_square_edge}. "
            "Task: infer the first two scan-center points from user natural-language instruction. "
            "If user clearly specifies corner start and row direction, follow it. "
            "If user only describes shape but not start corner, use center as first point and choose a valid nearby second point. "
            "The distance between the first and second point defines the step candidate in pixel. "
            "If user explicitly specifies step, use that step in these two points. "
            "If user does not specify step, use default step equal to scan_square_edge. "
            "Clamp all points into boundary. "
            "Return JSON schema exactly: "
            "{"
            "\"start_points\": [{\"x_pix\": int, \"y_pix\": int}, {\"x_pix\": int, \"y_pix\": int}], "
            "\"reason\": \"short explanation\""
            "}. "
            "Rules: start_points must contain exactly two points and the two points should be different. "
            "Never output text outside JSON."
        )

    def _starting_user_prompt(self, user_instruction):
        """User prompt for first-two-point planning."""
        left, right, top, bottom = self._navigator_scan_bounds()
        return (
            "User path requirement (natural language):\n"
            f"{user_instruction}\n"
            "Additional runtime info:\n"
            f"- scanable boundary: x in [{left}, {right}], y in [{top}, {bottom}]\n"
            f"- scan edge (physical): {self.Scan_edge}\n"
            f"- step_pix preference: {self.scan_square_edge}\n"
            "Return first two scan points as JSON only."
        )

    def _extract_user_step_pix(self, user_instruction):
        """Try to extract an explicit user-defined step from instruction text."""
        text = str(user_instruction or '').lower()

        pix_match = re.search(r'(\d+(?:\.\d+)?)\s*(px|pix|pixel)\b', text)
        if pix_match:
            return max(1, int(round(float(pix_match.group(1))))), 'user_explicit_pixel'

        si_match = re.search(r'(\d+(?:\.\d+)?)\s*([numk]?)(?:m)\b', text)
        if si_match:
            raw = f"{si_match.group(1)}{si_match.group(2)}m"
            try:
                pix_val = int(round(self.convert(raw) * 1e9))
                return max(1, pix_val), 'user_explicit_si'
            except Exception:
                pass

        nm_match = re.search(r'(\d+(?:\.\d+)?)\s*n(?:m)?\b', text)
        if nm_match:
            try:
                pix_val = int(round(float(nm_match.group(1))))
                return max(1, pix_val), 'user_explicit_nm'
            except Exception:
                pass

        return None, 'not_specified'

    def _infer_step_from_start_points(self, start_points):
        """Infer step from first two starting points if available."""
        if not isinstance(start_points, list) or len(start_points) < 2:
            return None
        p1, p2 = start_points[0], start_points[1]
        try:
            dx = abs(int(p2[0]) - int(p1[0]))
            dy = abs(int(p2[1]) - int(p1[1]))
        except Exception:
            return None
        step = max(dx, dy)
        return step if step > 0 else None

    def _resolve_effective_step_pix(self, plan, start_points, user_instruction):
        """Resolve final step with precedence: user explicit > start-points > plan > default."""
        user_step, user_src = self._extract_user_step_pix(user_instruction)
        if user_step is not None:
            return user_step, user_src

        step_from_start = self._infer_step_from_start_points(start_points)
        if step_from_start is not None:
            return step_from_start, 'starting_points_delta'

        try:
            plan_step = plan.get('step_pix', None) if isinstance(plan, dict) else None
            if plan_step is not None:
                return max(1, int(plan_step)), 'navigator_plan'
        except Exception:
            pass

        return max(1, int(self.scan_square_edge)), 'default_scan_square_edge'

    def _build_kimi_chat_payload(self, model_name, system_prompt, user_prompt, prefer_json=False):
        """Build request payload with model-specific parameter compatibility."""
        payload = {
            'model': model_name,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'prefer_json': bool(prefer_json),
        }

        allow_temperature = os.getenv('GEMINI_ALLOW_TEMPERATURE', '0').strip().lower() in ('1', 'true', 'yes')
        if allow_temperature:
            payload['temperature'] = float(os.getenv('GEMINI_TEMPERATURE', '0'))

        return payload

    def _mask_api_key(self, api_key):
        """Return a masked key for debug logs without leaking full secret."""
        if not isinstance(api_key, str) or not api_key:
            return '<empty>'
        if len(api_key) <= 10:
            return api_key[:2] + '***'
        return api_key[:6] + '...' + api_key[-4:]

    # def _instruction_prefers_spiral(self, user_instruction):
    #     """Heuristic: detect instructions preferring spiral-like trajectories (e.g., 回字形)."""
    #     text = str(user_instruction or '').strip().lower()
    #     if not text:
    #         return False
    #     keywords = (
    #         '回字', '回型', '回形', '方形螺旋', '矩形螺旋',
    #         'spiral', 'square spiral', 'rectangular spiral'
    #     )
    #     return any(key in text for key in keywords)

    def _get_kimi_runtime_config(self, purpose='navigator'):
        """Resolve request config, preferring main-class attributes over environment defaults."""
        default_model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

        if purpose == 'codegen':
            model_name = str(getattr(self, 'gemini_model_codegen', os.getenv('GEMINI_MODEL_CODEGEN', default_model)))
        elif purpose == 'path':
            model_name = str(getattr(self, 'gemini_model_path', os.getenv('GEMINI_MODEL_PATH', default_model)))
        else:
            model_name = str(getattr(self, 'gemini_model_navigator', os.getenv('GEMINI_MODEL_NAVIGATOR', default_model)))

        timeout_seconds = int(getattr(self, 'gemini_timeout_seconds', int(os.getenv('GEMINI_TIMEOUT', '90'))))
        retries = int(getattr(self, 'gemini_request_retries', int(os.getenv('GEMINI_RETRIES', '2'))))
        backoff_seconds = float(getattr(self, 'gemini_retry_backoff_seconds', float(os.getenv('GEMINI_RETRY_BACKOFF', '1.0'))))

        return {
            'model_name': model_name,
            'timeout_seconds': max(1, timeout_seconds),
            'retries': max(0, retries),
            'backoff_seconds': max(0.0, backoff_seconds),
        }

    def _request_kimi_chat_json(self, api_key, payload, config, log_tag):
        """Send chat completion request with timeout retry and return parsed JSON dict."""
        timeout_seconds = int(config['timeout_seconds'])
        retries = int(config['retries'])
        backoff_seconds = float(config['backoff_seconds'])
        max_attempts = retries + 1
        set_proxy_from_local_default()
        model_name = str(payload.get('model', config.get('model_name', 'gemini-2.5-flash')))
        system_prompt = str(payload.get('system_prompt', '') or '')
        user_prompt = str(payload.get('user_prompt', '') or '')
        prefer_json = bool(payload.get('prefer_json', False))
        temperature = payload.get('temperature', None)

        config_kwargs = {}
        if prefer_json:
            config_kwargs['response_mime_type'] = 'application/json'
        if temperature is not None:
            config_kwargs['temperature'] = float(temperature)

        gen_config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        client = genai.Client(api_key=api_key)

        for attempt in range(1, max_attempts + 1):
            try:
                print(
                    f'[{log_tag}] sending request... '
                    f'model={model_name}, timeout={timeout_seconds}s, '
                    f'attempt={attempt}/{max_attempts}, provider=gemini'
                )
                full_prompt = (
                    'System instruction:\n'
                    + system_prompt
                    + '\n\nUser instruction:\n'
                    + user_prompt
                    + '\n\nReturn JSON only.'
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=gen_config,
                )
                content = (getattr(response, 'text', None) or '').strip() or str(response)
                return {
                    'choices': [
                        {
                            'message': {
                                'content': content
                            }
                        }
                    ]
                }
            except TimeoutError as exc:
                if attempt < max_attempts:
                    wait_s = backoff_seconds * attempt
                    print(f'[{log_tag}] timeout on attempt {attempt}/{max_attempts}: {repr(exc)}; retry in {wait_s:.1f}s')
                    if wait_s > 0:
                        time.sleep(wait_s)
                    continue
                raise
            except Exception as exc:
                if attempt < max_attempts:
                    wait_s = backoff_seconds * attempt
                    print(f'[{log_tag}] transient error on attempt {attempt}/{max_attempts}: {repr(exc)}; retry in {wait_s:.1f}s')
                    if wait_s > 0:
                        time.sleep(wait_s)
                    continue
                raise

        raise RuntimeError(f'[{log_tag}] request failed after {max_attempts} attempts.')

    def _resolve_gemini_api_key(self, model_path):
        if isinstance(model_path, str) and model_path.startswith('AIza'):
            return model_path

        current = getattr(self, 'quality_model_path', None)
        if isinstance(current, str) and current.startswith('AIza'):
            return current

        env_key = os.getenv('GEMINI_API_KEY', '').strip()
        if env_key:
            return env_key

        return GEMINI_API_KEY

    def path_agent_from_text(self, user_instruction, start_points=None, model_path=None):
        """Use LLM Path Agent to map path-shape language to concrete waypoints."""
        user_instruction = str(user_instruction).strip()
        safe_start_points = self._sanitize_waypoints(start_points or [])[:2]
        if not user_instruction:
            return {}

        api_key = self._resolve_gemini_api_key(model_path)
        if not api_key:
            print('[Path Agent] API key is empty; skip path-agent request')
            return {}

        cfg = self._get_kimi_runtime_config(purpose='path')
        model_name = cfg['model_name']
        timeout_seconds = cfg['timeout_seconds']

        payload = self._build_kimi_chat_payload(
            model_name=model_name,
            system_prompt=self._path_system_prompt(),
            user_prompt=self._path_user_prompt(user_instruction, safe_start_points),
            prefer_json=True,
        )

        try:
            result = self._request_kimi_chat_json(api_key=api_key, payload=payload, config=cfg, log_tag='Path Agent')
            raw_content = result['choices'][0]['message']['content']
            parsed = self._safe_parse_json_block(raw_content)
            if not isinstance(parsed, dict):
                return {}
        except Exception as exc:
            print(f'[Path Agent] request failed: {repr(exc)}; use runtime fallback')
            return {}

        raw_points = []
        for item in parsed.get('waypoints', []) if isinstance(parsed, dict) else []:
            if isinstance(item, dict):
                raw_points.append((item.get('x_pix'), item.get('y_pix')))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                raw_points.append((item[0], item[1]))

        safe_points = self._sanitize_waypoints(raw_points)
        out = {
            'path_type': str(parsed.get('path_type', '')).strip().lower(),
            'step_pix': parsed.get('step_pix', None),
            'waypoints': safe_points,
            'python_code': str(parsed.get('python_code', '') or ''),
            'reason': str(parsed.get('reason', '')).strip(),
        }
        print(
            f"[Path Agent] parsed: path_type={out['path_type']}, "
            f"step_pix={out['step_pix']}, waypoints={len(out['waypoints'])}"
        )
        return out

    def path_agent_codegen_from_text(self, user_instruction, start_points=None, step_pix=None, model_path=None):
        """Ask Path Agent to generate Python path code, execute it safely, and return waypoints."""
        user_instruction = str(user_instruction).strip()
        safe_start_points = self._sanitize_waypoints(start_points or [])[:2]
        if not user_instruction:
            return {}

        if step_pix is None:
            step_pix = max(1, int(self.scan_square_edge))

        api_key = self._resolve_gemini_api_key(model_path)
        if not api_key:
            print('[Path Agent Codegen] API key is empty; skip codegen request')
            return {}

        cfg = self._get_kimi_runtime_config(purpose='codegen')
        model_name = cfg['model_name']
        timeout_seconds = cfg['timeout_seconds']

        payload = self._build_kimi_chat_payload(
            model_name=model_name,
            system_prompt=self._path_codegen_system_prompt(),
            user_prompt=self._path_codegen_user_prompt(user_instruction, safe_start_points, step_pix),
            prefer_json=True,
        )

        try:
            result = self._request_kimi_chat_json(api_key=api_key, payload=payload, config=cfg, log_tag='Path Agent Codegen')
            raw_content = result['choices'][0]['message']['content']
            parsed = self._safe_parse_json_block(raw_content)
            if not isinstance(parsed, dict):
                return {}
        except Exception as exc:
            print(f'[Path Agent Codegen] request failed: {repr(exc)}; skip codegen')
            return {}

        code_str = str(parsed.get('python_code', '') or '').strip()
        points = self._execute_path_codegen(
            code_str=code_str,
            start_points=safe_start_points,
            step_pix=int(step_pix),
        )
        out = {
            'path_type': str(parsed.get('path_type', 'custom')).strip().lower(),
            'step_pix': parsed.get('step_pix', int(step_pix)),
            'waypoints': points,
            'python_code': code_str,
            'reason': str(parsed.get('reason', '')).strip(),
        }
        print(
            f"[Path Agent Codegen] parsed: path_type={out['path_type']}, "
            f"step_pix={out['step_pix']}, code_len={len(code_str)}, waypoints={len(points)}"
        )
        return out

    def _validate_path_codegen_ast(self, code_str):
        """Static AST validation for LLM-generated code before execution."""
        if not code_str or not isinstance(code_str, str):
            raise ValueError('python_code is empty.')

        tree = ast.parse(code_str, mode='exec')
        forbidden = (
            ast.Import,
            ast.ImportFrom,
            ast.With,
            ast.Try,
            ast.Raise,
            ast.ClassDef,
            ast.Global,
            ast.Nonlocal,
            ast.Delete,
            ast.Lambda,
            ast.Yield,
            ast.Await,
        )
        for node in ast.walk(tree):
            if isinstance(node, forbidden):
                raise ValueError(f'Forbidden syntax in path code: {type(node).__name__}')

        fn_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if len(fn_defs) != 1 or fn_defs[0].name != 'generate_path':
            raise ValueError('python_code must define exactly one function named generate_path(context).')

        return tree

    def _execute_path_codegen(self, code_str, start_points, step_pix):
        """Execute validated path code in restricted globals and sanitize outputs."""
        try:
            tree = self._validate_path_codegen_ast(code_str)
        except Exception as exc:
            print(f'[Path Agent Codegen] AST validation failed: {repr(exc)}')
            return []

        safe_builtins = {
            'range': range,
            'len': len,
            'min': min,
            'max': max,
            'abs': abs,
            'int': int,
            'float': float,
            'round': round,
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'set': set,
            'enumerate': enumerate,
            'zip': zip,
        }

        exec_globals = {'__builtins__': safe_builtins, 'np': np}
        exec_locals = {}
        try:
            compiled = compile(tree, filename='<path_agent_codegen>', mode='exec')
            exec(compiled, exec_globals, exec_locals)
            func = exec_locals.get('generate_path') or exec_globals.get('generate_path')
            if not callable(func):
                raise ValueError('generate_path is not callable')
        except Exception as exc:
            print(f'[Path Agent Codegen] compile/exec failed: {repr(exc)}')
            return []

        left, right, top, bottom = self._navigator_scan_bounds()
        context = {
            'plane_size': int(self.plane_size),
            'scan_edge': str(self.Scan_edge),
            'left': int(left),
            'right': int(right),
            'top': int(top),
            'bottom': int(bottom),
            'step_pix': int(step_pix),
            'start_points': [tuple(p) for p in (start_points or [])[:2]],
        }

        try:
            result = func(context)
        except Exception as exc:
            print(f'[Path Agent Codegen] runtime failed: {repr(exc)}')
            return []

        if not isinstance(result, (list, tuple)):
            print('[Path Agent Codegen] runtime returned non-list points')
            return []

        capped = list(result)[:250000]
        safe_points = self._sanitize_waypoints(capped)
        return safe_points

    def starting_navigator_from_text(self, user_instruction, model_path=None):
        """Use LLM to infer first two scan points from natural-language instruction."""
        user_instruction = str(user_instruction).strip()
        print(
            f"[Starting Planning Agent] input received: len={len(user_instruction)}, "
            f"preview={repr(user_instruction[:80])}"
        )
        if not user_instruction:
            print('[Starting Planning Agent] empty user_instruction; skip request and fallback to default start')
            return []

        api_key = self._resolve_gemini_api_key(model_path)
        print(
            f"[Starting Planning Agent] api_key source={'model_path' if isinstance(model_path, str) and model_path.startswith('AIza') else 'resolved_gemini_key'}, "
            f"present={bool(api_key)}, masked={self._mask_api_key(api_key)}"
        )
        if not api_key:
            print('[Starting Planning Agent] API key is empty; fallback to default start')
            return []

        cfg = self._get_kimi_runtime_config(purpose='navigator')
        model_name = cfg['model_name']
        timeout_seconds = cfg['timeout_seconds']

        payload = self._build_kimi_chat_payload(
            model_name=model_name,
            system_prompt=self._starting_system_prompt(),
            user_prompt=self._starting_user_prompt(user_instruction),
            prefer_json=True,
        )
        print(f"[Starting Planning Agent] payload prepared: keys={list(payload.keys())}")

        try:
            result = self._request_kimi_chat_json(api_key=api_key, payload=payload, config=cfg, log_tag='Starting Planning Agent')
            raw_content = result['choices'][0]['message']['content']
            parsed = self._safe_parse_json_block(raw_content)
        except Exception as exc:
            print(f'[Starting Planning Agent] request failed: {repr(exc)}; fallback to default start')
            return []

        raw_points = []
        for item in parsed.get('start_points', []) if isinstance(parsed, dict) else []:
            if isinstance(item, dict):
                raw_points.append((item.get('x_pix'), item.get('y_pix')))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                raw_points.append((item[0], item[1]))

        safe_points = self._sanitize_waypoints(raw_points)
        print(f"[Starting Planning Agent] raw_points={len(raw_points)}, safe_points={safe_points[:2]}")
        if len(safe_points) >= 2:
            return safe_points[:2]

        return []

    def plan_navigator_from_text(self, user_instruction, model_path=None):
        """Use LLM to translate natural-language path description into navigator config."""
        user_instruction = str(user_instruction).strip()
        print(
            f"[Planning Agent] input received: len={len(user_instruction)}, "
            f"preview={repr(user_instruction[:80])}"
        )
        if not user_instruction:
            raise ValueError('user_instruction cannot be empty.')

        self.navigator_user_instruction = user_instruction

        api_key = self._resolve_gemini_api_key(model_path)

        if not api_key:
            raise RuntimeError('Planning Agent API key is empty.')

        cfg = self._get_kimi_runtime_config(purpose='navigator')
        model_name = cfg['model_name']
        timeout_seconds = cfg['timeout_seconds']

        payload = self._build_kimi_chat_payload(
            model_name=model_name,
            system_prompt=self._navigator_system_prompt(),
            user_prompt=self._navigator_user_prompt(user_instruction),
            prefer_json=True,
        )
        print(f"[Planning Agent] payload prepared: keys={list(payload.keys())}")

        try:
            result = self._request_kimi_chat_json(api_key=api_key, payload=payload, config=cfg, log_tag='Planning Agent')
            raw_content = result['choices'][0]['message']['content']
            plan = self._safe_parse_json_block(raw_content)
            print(f"[Planning Agent] plan parsed type={type(plan).__name__}, keys={list(plan.keys()) if isinstance(plan, dict) else 'N/A'}")
        except Exception as exc:
            print(f'[Planning Agent] request failed: {repr(exc)}; fallback to custom empty plan')
            plan = {'path_type': 'custom', 'step_pix': None, 'waypoints': [], 'reason': f'request_failed: {exc}'}

        start_points = self.starting_navigator_from_text(user_instruction=user_instruction, model_path=model_path)
        pre_step, _ = self._resolve_effective_step_pix(
            plan=plan,
            start_points=start_points,
            user_instruction=user_instruction,
        )
        path_agent_plan = self.path_agent_codegen_from_text(
            user_instruction=user_instruction,
            start_points=start_points,
            step_pix=pre_step,
            model_path=model_path,
        )
        if not path_agent_plan.get('waypoints'):
            path_agent_plan = self.path_agent_from_text(
                user_instruction=user_instruction,
                start_points=start_points,
                model_path=model_path,
            )
        applied = self.apply_navigator_plan(plan, start_points=start_points, path_agent_plan=path_agent_plan)
        self.navigator_last_plan = applied
        print(f"[Planning Agent] applied plan: {applied}")
        return applied

    def apply_navigator_plan(self, plan, start_points=None, path_agent_plan=None):
        """Validate LLM plan and apply to runtime navigator state."""
        if not isinstance(plan, dict):
            plan = {}
        if not isinstance(path_agent_plan, dict):
            path_agent_plan = {}

        if start_points is None:
            start_points = []
        self.navigator_start_points = self._sanitize_waypoints(start_points)[:2]

        nav_path_type = str(plan.get('path_type', 'custom')).strip().lower()
        path_type = str(path_agent_plan.get('path_type', nav_path_type)).strip().lower()
        waypoints = path_agent_plan.get('waypoints', []) if path_agent_plan.get('waypoints') else plan.get('waypoints', [])
        reason_parts = []
        if str(plan.get('reason', '')).strip():
            reason_parts.append(f"navigator={str(plan.get('reason', '')).strip()}")
        if str(path_agent_plan.get('reason', '')).strip():
            reason_parts.append(f"path_agent={str(path_agent_plan.get('reason', '')).strip()}")
        reason = ' | '.join(reason_parts)

        effective_step_pix, step_source = self._resolve_effective_step_pix(
            plan=path_agent_plan if path_agent_plan else plan,
            start_points=self.navigator_start_points,
            user_instruction=self.navigator_user_instruction,
        )
        print(f"[Planning Agent] step resolved: {effective_step_pix} (source={step_source})")

        raw_points = []
        if isinstance(waypoints, list):
            for item in waypoints:
                if isinstance(item, dict):
                    raw_points.append((item.get('x_pix'), item.get('y_pix')))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    raw_points.append((item[0], item[1]))

        safe_points = self._sanitize_waypoints(raw_points)
        runtime_path_type = path_type
        path_source = 'path_agent_waypoints' if path_agent_plan.get('waypoints') else 'runtime_generator'
        if len(safe_points) < 2:
            if len(self.navigator_start_points) >= 2:
                safe_points = self.navigator_start_points[:2]
            elif len(self.navigator_start_points) == 1:
                safe_points = self.navigator_start_points[:1]
            else:
                left, right, top, bottom = self._navigator_scan_bounds()
                safe_points = [((left + right) // 2, (top + bottom) // 2)]
            path_source = 'fallback_default'

        self.configure_navigator_agent(
            mode='custom',
            step_pix=effective_step_pix,
            custom_points=safe_points,
            start_points=self.navigator_start_points,
        )

        return {
            'path_type': runtime_path_type,
            'runtime_mode': 'custom',
            'step_pix': int(self.navigator_step_pix),
            'step_source': step_source,
            'waypoints_count': len(self.navigator_custom_points),
            'start_points': self.navigator_start_points,
            'path_agent_used': bool(path_agent_plan),
            'path_source': path_source,
            'reason': reason,
        }

    def navigator_reset(self):
        """Reset Planning Agent runtime buffers when task starts or mode changes."""
        self.navigator_snake_points = []
        self.navigator_snake_index = 0
        self.navigator_custom_index = 0

    def _consume_navigator_skip_budget(self):
        """Consume and clear one-shot skip budget set by Conditioning Agent TipShaper."""
        extra = max(0, int(self.navigator_skip_budget))
        self.navigator_skip_budget = 0
        return extra

    def _point_in_bounds(self, point):
        """Check if point is inside current scan boundary."""
        left, right, top, bottom = self._navigator_scan_bounds()
        x, y = int(round(point[0])), int(round(point[1]))
        return left <= x <= right and top <= y <= bottom

    def _clamp_point_to_bounds(self, point):
        """Clamp a point into current scan boundary."""
        left, right, top, bottom = self._navigator_scan_bounds()
        x = int(round(point[0]))
        y = int(round(point[1]))
        x = min(max(x, left), right)
        y = min(max(y, top), bottom)
        return (x, y)

    def _sanitize_waypoints(self, waypoints):
        """Convert waypoints to unique in-bound integer pixel points."""
        safe_points = []
        seen = set()
        for point in waypoints:
            if point is None or len(point) < 2:
                continue
            try:
                clamped = self._clamp_point_to_bounds((float(point[0]), float(point[1])))
            except Exception:
                continue
            if clamped not in seen:
                safe_points.append(clamped)
                seen.add(clamped)
        return safe_points

    def _navigator_scan_bounds(self):
        """Return scanable rectangle in pixel coordinates."""
        left = round(self.plane_size / 2 * (1 - self.real_scan_factor))
        right = round(self.plane_size / 2 * (1 + self.real_scan_factor))
        top = round(self.plane_size / 2 * (1 - self.real_scan_factor))
        bottom = round(self.plane_size / 2 * (1 + self.real_scan_factor))
        return left, right, top, bottom

    # def _build_snake_points(self, step_pix=None, start_points=None):
    #     """Build a deterministic serpentine (snake) waypoint list in scanable area."""
    #     left, right, top, bottom = self._navigator_scan_bounds()
    #     step = max(1, int(self.navigator_step_pix if step_pix is None else step_pix))

    #     xs = np.arange(left, right + 1, step, dtype=int)
    #     ys = np.arange(top, bottom + 1, step, dtype=int)
    #     if xs.size == 0:
    #         xs = np.array([left], dtype=int)
    #     if ys.size == 0:
    #         ys = np.array([top], dtype=int)
    #     if xs[-1] != right:
    #         xs = np.append(xs, right)
    #     if ys[-1] != bottom:
    #         ys = np.append(ys, bottom)

    #     points = []
    #     for row_idx, y in enumerate(ys):
    #         row_xs = xs if row_idx % 2 == 0 else xs[::-1]
    #         for x in row_xs:
    #             points.append((int(x), int(y)))

    #     if not points:
    #         return [(round(self.plane_size / 2), round(self.plane_size / 2))]

    #     local_start_points = self.navigator_start_points if start_points is None else start_points
    #     if len(local_start_points) >= 1:
    #         start_target = local_start_points[0]
    #     else:
    #         start_target = (round(self.plane_size / 2), round(self.plane_size / 2))

    #     start_idx = min(
    #         range(len(points)),
    #         key=lambda i: (points[i][0] - start_target[0]) ** 2 + (points[i][1] - start_target[1]) ** 2,
    #     )
    #     ordered = points[start_idx:] + points[:start_idx]

    #     if len(local_start_points) >= 2 and len(ordered) > 1:
    #         second_target = local_start_points[1]
    #         dist_forward = (ordered[1][0] - second_target[0]) ** 2 + (ordered[1][1] - second_target[1]) ** 2
    #         dist_backward = (ordered[-1][0] - second_target[0]) ** 2 + (ordered[-1][1] - second_target[1]) ** 2
    #         if dist_backward < dist_forward:
    #             ordered = [ordered[0]] + list(reversed(ordered[1:]))

    #     return ordered

    def navigator_next_point(self):
        """Planning Agent decision: output next scan-center in pixel coordinates."""
        if self.navigator_mode != 'custom':
            print(f"[Planning Agent] unsupported mode '{self.navigator_mode}'; falling back to custom.")
            self.navigator_mode = 'custom'

        if self.navigator_mode == 'custom':
            if not self.navigator_custom_points:
                if self.navigator_start_points:
                    return self.navigator_start_points[0]
                return (round(self.plane_size / 2), round(self.plane_size / 2))

            if len(self.circle_list) == 0:
                self.navigator_custom_index = 1
                return self.navigator_custom_points[0]

            step_advance = 1 + self._consume_navigator_skip_budget()
            idx = min(self.navigator_custom_index, len(self.navigator_custom_points) - 1)
            candidate = self.navigator_custom_points[idx]
            self.navigator_custom_index = min(self.navigator_custom_index + step_advance, len(self.navigator_custom_points))
            return candidate

        raise ValueError(f'Unsupported navigator mode: {self.navigator_mode}')


if __name__ == '__main__':
    class _NavigatorPathAgentSmokeTest(NavigatorPathAgentMixin):
        def __init__(self):
            self.plane_size = 1000
            self.real_scan_factor = 0.8
            self.Scan_edge = '30n'
            self.scan_square_edge = 30

            self.navigator_mode = 'custom'
            self.navigator_step_pix = self.scan_square_edge
            self.navigator_snake_points = []
            self.navigator_snake_index = 0
            self.navigator_custom_points = []
            self.navigator_custom_index = 0
            self.navigator_user_instruction = ''
            self.navigator_last_plan = {}
            self.navigator_start_points = []
            self.navigator_skip_budget = 0

            self.quality_model_path = os.getenv('GEMINI_API_KEY', '')
            self.gemini_model_navigator = os.getenv('GEMINI_MODEL_NAVIGATOR', os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'))
            self.gemini_model_path = os.getenv('GEMINI_MODEL_PATH', os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'))
            self.gemini_model_codegen = os.getenv('GEMINI_MODEL_CODEGEN', os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'))
            self.gemini_timeout_seconds = int(os.getenv('GEMINI_TIMEOUT', '90'))
            self.gemini_request_retries = int(os.getenv('GEMINI_RETRIES', '2'))
            self.gemini_retry_backoff_seconds = float(os.getenv('GEMINI_RETRY_BACKOFF', '1.0'))

            self.circle_list = []
            self.navigator_reset()

    nanonis = _NavigatorPathAgentSmokeTest()
    instruction = (
        'The scan shall start from the center of the scan area and proceed outward in '
        'rectangular spiral loops until the entire scannable area is covered.'
    )

    applied_plan = nanonis.plan_navigator_from_text(user_instruction=instruction)
    print('[Navigator Test] applied_plan:')
    print(json.dumps(applied_plan, ensure_ascii=False, indent=2))

    print('[Navigator Test] generated coordinates:')
    if nanonis.navigator_mode == 'custom' and nanonis.navigator_custom_points:
        for idx, point in enumerate(nanonis.navigator_custom_points, start=1):
            print(f'{idx:03d}: ({int(point[0])}, {int(point[1])})')
    else:
        sample_points = 30
        sampled = []
        for _ in range(sample_points):
            point = nanonis.navigator_next_point()
            px = int(round(point[0]))
            py = int(round(point[1]))
            sampled.append((px, py))
            nanonis.circle_list.append([px, py, int(nanonis.scan_square_edge), 1])
        for idx, point in enumerate(sampled, start=1):
            print(f'{idx:03d}: ({point[0]}, {point[1]})')
