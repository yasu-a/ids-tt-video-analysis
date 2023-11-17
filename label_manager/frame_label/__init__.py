#
# from labels import common
#
#
# def json_to_python_hash(json_root):
#     md5_digest = json_to_md5_digest(json_root)
#     return hash(md5_digest)
#
#
# def load_md5_and_json(file_path) -> tuple[str, Any]:
#     with open(file_path, 'r') as f:
#         json_root = json.load(f)
#         hash_value: str = json_to_md5_digest(json_root)
#     return hash_value, json_root
#
#
# class VideoMarkerEntry(NamedTuple):
#     parent: 'VideoMarker'
#     frame_index: int
#     name: str
#     tags: tuple[str]
#
#
# class VideoMarker:
#     def __init__(self, src_json_root, src_path=None):
#         self.__src_path = src_path
#         self.__json_root = self._normalize_json(src_json_root)
#         self.__markers = self.__json_root['markers']
#         self.__tags = self.__json_root['tags']
#         self.__frame_indexes = np.array(sorted(self.__markers.keys()))
#         self.__frame_index_to_entry: dict[int, VideoMarkerEntry] = {
#             i: VideoMarkerEntry(
#                 parent=self,
#                 frame_index=i,
#                 name=self.__markers[i],
#                 tags=self.__tags[i]
#             ) for i in self.__frame_indexes
#         }
#
#     @property
#     def frame_index_array(self):
#         return np.array(self.__frame_indexes)
#
#     @functools.cached_property
#     def ordered_marker_names(self) -> tuple[str]:
#         return tuple(sorted(entry.name for entry in self.entries()))
#
#     @property
#     def marker_name_index_array(self):
#         name_to_name_index = {name: i for i, name in enumerate(self.ordered_marker_names)}
#         return np.array([
#             name_to_name_index[entry.name]
#             for entry in self.entries()
#         ])
#
#     # marker_name: str -> frame_index: int
#     def frame_indexes_grouped_by_marker_name(self) -> dict[str, np.ndarray]:
#         dct = collections.defaultdict(list)
#         for entry in self.entries():
#             dct[entry.name].append(entry.frame_index)
#         return {
#             marker_name: np.array(frame_index_lst)
#             for marker_name, frame_index_lst in dct.items()
#         }
#
#     @functools.cached_property
#     def __hash(self):
#         return json_to_python_hash(self.__json_root)
#
#     @functools.cached_property
#     def __ordering_key(self):
#         return json_to_md5_digest(self.__json_root)
#
#     def __hash__(self):
#         return self.__hash
#
#     def __lt__(self, other):
#         return self.__ordering_key < other.__ordering_key
#
#     def __repr__(self):
#         return f'VideoMarker({self.__src_path!r})'
#
#     @classmethod
#     def _normalize_json(cls, json_root):
#         def parse_key_to_int(d: dict):
#             try:
#                 return {int(k): v for k, v in d.items()}
#             except ValueError:
#                 raise ValueError('invalid marker json format')
#
#         markers = parse_key_to_int(json_root['markers'])
#         tags = parse_key_to_int(json_root['tags'])
#
#         # check frame_indexes
#         if not set(markers.keys()) == set(tags.keys()):
#             raise ValueError('invalid marker json format')
#
#         # check json structure
#         # TODO: check all structures
#         if not all(isinstance(v, list) for v in tags.values()):
#             raise ValueError('invalid marker json format')
#
#         return dict(
#             markers=markers,
#             tags=tags
#         )
#
#     def dump(self, path=None):
#         path = path or self.__src_path
#         assert path is not None, path
#         with open(path, 'w') as f:
#             json.dump(self.__json_root, f, indent=2, sort_keys=True)
#
#     @classmethod
#     def load(cls, path):
#         with open(path, 'r') as f:
#             json_root = json.load(f)
#         return cls(src_json_root=json_root, src_path=path)
#
#     def __getitem__(self, frame_index) -> Optional[VideoMarkerEntry]:
#         return self.__markers.get(frame_index)
#
#     def keys(self) -> Iterator[int]:  # frame index
#         yield from self.__frame_index_to_entry.keys()
#
#     def entries(self) -> Iterator[VideoMarkerEntry]:
#         yield from self.__frame_index_to_entry.values()
#
#     def find_neighbour_frame_index(self, index):
#         a = np.abs(self.__frame_indexes - index)
#         return self.__frame_indexes[np.argmin(a)]
#
#
# def import_json(json_path):
#     video_name = get_video_name_from_json_path(json_path)
#
#     dir_path = common.resolve_data_path(_root_path, video_name)
#     os.makedirs(dir_path, exist_ok=True)
#
#     hash_value, json_root = load_md5_and_json(json_path)
#     file_path = os.path.join(dir_path, hash_value + '.json')
#
#     if os.path.exists(file_path):
#         return json_path, None
#
#     VideoMarker(json_root, file_path).dump()
#
#     return json_path, file_path
#
#
# def iter_marker_dir_path_and_video_names():
#     root_dir = common.resolve_data_path(_root_path)
#     for video_name in os.listdir(root_dir):
#         yield os.path.join(root_dir, video_name), video_name
#
#
# @dataclass(frozen=True)
# class VideoMarkerSet:
#     video_name: str
#     marker_set: tuple[VideoMarker]
#
#     @classmethod
#     def create_full_set(cls) -> dict[str, 'VideoMarkerSet']:
#         dct: dict[str, list[VideoMarker]] = collections.defaultdict(list[VideoMarker])
#         for dir_path, video_name in iter_marker_dir_path_and_video_names():
#             for json_name in os.listdir(dir_path):
#                 json_path = os.path.join(dir_path, json_name)
#                 marker = VideoMarker.load(json_path)
#                 dct[video_name].append(marker)
#         cls(video_name='aa', marker_set=tuple())
#         return {
#             video_name: cls(
#                 video_name=video_name,
#                 marker_set=tuple(marker_lst)
#             ) for video_name, marker_lst in dct.items()
#         }
#
#     def calculate_meaningful_margin(self) -> float:
#         margins = []
#         for m in self.marker_set:
#             frame_indexes = np.array(list(m.keys()))
#             diff = np.diff(frame_indexes)
#             import scipy.stats
#             kernel = scipy.stats.gaussian_kde(diff, bw_method=.03)
#             import matplotlib.pyplot as plt
#             _, x, _ = plt.hist(diff, bins=256, density=True)
#             plt.plot(x, kernel(x))
#             plt.show()
#             margin = (diff.mean() - diff.std() * 2) / 2
#             margins.append(margin)
#         return min(margins)
#
#     # FIXME: not working
#     def classify(self):
#         meaningful_margin = self.calculate_meaningful_margin()
#
#         @dataclass
#         class Cluster:
#             points: list[VideoMarkerEntry]
#             v_min: float
#             v_max: float
#
#             @classmethod
#             def create_instance(cls):
#                 return Cluster(
#                     points=[],
#                     v_min=None,
#                     v_max=None
#                 )
#
#             def add_point(self, entry: VideoMarkerEntry):
#                 self.points.append(entry)
#                 a = np.array([m.frame_index for m in self.points])
#                 self.v_min = a.min()
#                 self.v_max = a.max()
#
#             def is_neighbour(self, entry: VideoMarkerEntry):
#                 return self.v_min - meaningful_margin <= entry.frame_index \
#                     <= self.v_max + meaningful_margin
#
#         clusters: list[Cluster] = []
#
#         def find_nearest_cluster(e: VideoMarkerEntry) -> Optional[Cluster]:
#             for c in clusters:
#                 if c.is_neighbour(e):
#                     return c
#             return None
#
#         for m in self.marker_set:
#             for e in m.entries():
#                 c = find_nearest_cluster(e)
#                 if c is None:
#                     c = Cluster.create_instance()
#                     clusters.append(c)
#                 c.add_point(e)
#
#         groups: list[list[VideoMarkerEntry]] \
#             = [list(c.points) for c in clusters if len(c.points) > 1]
#         rest_entries \
#             = [list(c.points) for c in clusters if len(c.points) <= 1]
#
#         return groups, rest_entries
#
#
# class MarkerDataSet:
#     def __init__(self):
#         self.__marker_set = VideoMarkerSet.create_full_set()
#         for k, v in self.__marker_set.items():
#             print(k)
#             pprint(v.classify()[0])
