"""Recording fake of an OpenAI-compatible chat server, for planner-request parity checks.

Every POST body is appended to <log> as one JSON line: the sha256 of the body, then per message part
the text (sha256 + length + head) or the image (sha256 of the decoded JPEG bytes). Replies are a pure
function of the request text, so two code versions that send the same requests get the same replies.

    python fake_llm.py <port> <log.jsonl>
    export SPARK_LLM_BACKEND=openai OPENAI_BASE_URL=http://127.0.0.1:<port>/v1 OPENAI_API_KEY=EMPTY
"""
import base64
import hashlib
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def reply_for(text: str) -> str:
    if 'For each concept below' in text:
        block = text.split('Concepts:\n', 1)[1].split('\n\nOutput JSON', 1)[0]
        concepts = re.findall(r'^- (.+)$', block, re.M)
        return json.dumps({c: [c] for c in concepts})
    if 'ANSWER: <number>' in text:
        return 'ANSWER: 1'
    if 'ANSWER:' in text:
        return 'ANSWER: yes'
    labels = re.findall(r'^  ([^\n:]+): \(-?\d', text, re.M)
    if labels:
        pick, place = labels[0], labels[-1]
        return ('```yaml\ntask: parity\ntree:\n  type: sequence\n  children:\n'
                f'    - type: move_to_keypoint\n      params: {{keypoint_label: "{pick}"}}\n'
                '    - type: grasp\n      params: {force: 100}\n'
                '    - type: move_relative\n      params: {dz: 0.15}\n'
                f'    - type: move_to_keypoint\n      params: {{keypoint_label: "{place}", offset_z: 0.1}}\n'
                '    - type: release\n```')
    return '[]'


def summarize(body: bytes) -> dict:
    req = json.loads(body)
    parts = []
    for m in req.get('messages', []):
        content = m['content'] if isinstance(m['content'], list) else [{'type': 'text', 'text': m['content']}]
        for p in content:
            if p['type'] == 'text':
                t = p['text'].encode()
                parts.append({'text_sha256': _sha(t), 'len': len(t), 'head': p['text'][:80]})
            else:
                jpg = base64.b64decode(p['image_url']['url'].split(',', 1)[1])
                parts.append({'image_sha256': _sha(jpg)})
    return {'body_sha256': _sha(body), 'model': req.get('model'), 'temperature': req.get('temperature'),
            'parts': parts}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        rec = summarize(body)
        with open(self.server.log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        text = '\n'.join(p['text'] for m in json.loads(body)['messages']
                         for p in (m['content'] if isinstance(m['content'], list) else [{'type': 'text', 'text': m['content']}])
                         if p['type'] == 'text')
        out = json.dumps({
            'id': 'fake', 'object': 'chat.completion', 'created': 0, 'model': rec['model'],
            'choices': [{'index': 0, 'finish_reason': 'stop',
                         'message': {'role': 'assistant', 'content': reply_for(text)}}],
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
        }).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    srv = HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler)
    srv.log_path = sys.argv[2]
    srv.serve_forever()
