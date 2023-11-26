from typing import NamedTuple
import app_logging
from frozendict import frozendict

logger = app_logging.create_logger(__name__)


def dict_ambiguous_get(dct, k):
    if k in dct:
        return dct[k]
    k = k.replace('-', '_')
    return dct[k]


class LabelJsonMeta(NamedTuple):
    author: frozendict
    video_name: str
    hash_digest: str

    @property
    def author_hash_digest(self):
        # TODO: update when full data collected
        string = str(sorted(self.author.items(), key=lambda x: x[0]))
        import hashlib
        return hashlib.md5(string.encode('utf-8')).hexdigest()

    @classmethod
    def from_json(cls, json_root: dict):
        return cls(
            author=frozendict(json_root['author']),
            video_name=dict_ambiguous_get(json_root, 'video-name'),
            hash_digest=dict_ambiguous_get(json_root, 'hash-digest')
        )

    def to_json(self) -> dict:
        return self._asdict() | dict(author_hash_digest=self.author_hash_digest)


class LabelJsonFrame(NamedTuple):
    fi: int
    label: str
    tags: tuple[str, ...]

    @classmethod
    def from_json(cls, json_root):
        def _raise():
            raise ValueError('invalid json format', json_root)

        if not isinstance(json_root, dict):
            _raise()
        if set(json_root.keys()) != {'fi', 'label', 'tags'}:
            _raise()
        fi, label, tags = json_root['fi'], json_root['label'], json_root['tags']
        if not isinstance(fi, int):
            _raise()
        if not isinstance(label, str):
            _raise()
        if not isinstance(tags, list):
            _raise()
        if not all(isinstance(tag, str) for tag in tags):
            _raise()

        return cls(
            fi=fi,
            label=label,
            tags=tuple(tags)
        )

    def to_json(self):
        return self._asdict()


class LabelJsonFrames(NamedTuple):
    frames: list[LabelJsonFrame]

    @classmethod
    def from_json(cls, json_root: dict):
        if isinstance(json_root, dict):
            json_root = list(json_root.values())
        if not isinstance(json_root, list):
            raise ValueError('invalid json format')

        entries = [
            LabelJsonFrame.from_json(item)
            for item in json_root
        ]

        return cls(
            frames=entries
        )

    def to_json(self) -> list[dict]:
        return sorted(
            (entry.to_json() for entry in self.frames),
            key=lambda entry: entry['fi']
        )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.frames[i]

    def __iter__(self):
        yield from self.frames


class LabelJson(NamedTuple):
    meta: LabelJsonMeta
    frames: LabelJsonFrames

    @classmethod
    def from_json(cls, json_root: dict):
        return cls(
            meta=LabelJsonMeta.from_json(json_root['meta']),
            frames=LabelJsonFrames.from_json(json_root['frames'])
        )

    def to_json(self) -> dict:
        return dict(
            meta=self.meta.to_json(),
            frames=self.frames.to_json()
        )
