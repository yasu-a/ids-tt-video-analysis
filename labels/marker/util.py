import hashlib
import json
import os.path
import re


def extract_video_name_from_json_path(path):
    _, name = os.path.split(path)
    name, _ = os.path.splitext(name)
    m = re.match(r'\S+', name)
    if not m:
        raise ValueError('invalid json path', path)
    return m.group(0)


def normalized_json_text(json_root):
    norm = json.dumps(
        json_root,
        indent=2,
        sort_keys=True,
        ensure_ascii=True
    ).encode('latin-1')
    return norm


def json_to_md5_digest(json_root):
    norm = normalized_json_text(json_root)
    hash_value: str = hashlib.md5(norm).hexdigest()
    return hash_value
