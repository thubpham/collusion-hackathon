import json
import re


class LLMClient:
    def _clean_response(self, response: str, model: str) -> str:
        response = re.sub(r'^(Assistant:|AI:|Human:|User:)', '', response).strip()
        response = re.sub(r'\n(Assistant:|AI:|Human:|User:).*$', '', response, flags=re.MULTILINE | re.DOTALL)
        for stop_phrase in ["Human:", "User:", "\n\n---", "```\n\n", "<|", "[INST]"]:
            if stop_phrase in response:
                response = response.split(stop_phrase)[0].strip()
        if response.strip().startswith('{') or '```json' in response:
            response = self._extract_clean_json(response)
        return response.strip()

    def _extract_clean_json(self, response: str) -> str:
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        for stop_phrase in ["Human:", "User:", "Assistant:"]:
            if stop_phrase in response:
                response = response.split(stop_phrase)[0].strip()
        json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response)
        if json_match:
            json_str = json_match.group(1)
            json_str = re.sub(r':\s*(?!true\b|false\b|null\b)([A-Za-z][A-Za-z0-9]*)', r': "\1"', json_str)
            json_str = re.sub(r'\[\s*(?!true\b|false\b|null\b)([A-Za-z0-9]+)\s*,\s*(?!true\b|false\b|null\b)([A-Za-z0-9]+)\s*\]', r'["\1", "\2"]', json_str)
            json_str = re.sub(r'\[\s*(?!true\b|false\b|null\b)([A-Za-z0-9]+)\s*\]', r'["\1"]', json_str)
            try:
                parsed = json.loads(json_str)
                return json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                return json_str
        return response.strip()
