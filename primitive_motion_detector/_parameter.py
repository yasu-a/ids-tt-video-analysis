from dataclasses import dataclass


@dataclass(frozen=True)
class PMDetectorParameter:
    # 極大点の算出で考慮する差分画像の最小輝度（差分画像にかけるマスクの条件）
    diff_minimum_luminance: float = 0.1

    # 与えられた画像にかける平均値フィルタの大きさ`画像サイズ // mean_filter_size_factor`
    mean_filter_size_factor: int = 32

    # 輝度極大点を求めるときに`skimage.feature.peak_local_max`の引数`min_distance`に
    # 与える値`画像サイズ // local_max_distance_factor`
    local_max_distance_factor: int = 32

    # 輝度極大点を求めるときに極大点の対象とする最小輝度
    motion_local_max_thresh: float = 0.03

    # キーフレームの相互マッチングのときに，cos距離がこの値以下のマッチングが対象となる。
    # cos距離は`1 - np.clip(cos_distance, 0, 1)`で算出され，0に近いほうが距離が近い。
    mutual_match_max_cos_distance: float = 0.3

    # 切り出すキーフレームの大きさで，中心点から±`key_image_size`の範囲が切り出される。
    # 実際に切り出されるキーフレームの大きさは`key_image_size * 2 + 1`
    key_image_size: int = 32

    # これ以上の動き（正規化済み）を持つモーションはエラーとして除外する
    max_velocity_normalized: float = 1.5

    # Motion-centroid-correctionを行うかどうか
    enable_motion_correction: bool = True

    # Motion-centroid-correctionでテンプレートマッチの結果にかける円フィルタの半径」
    # もとのキーフレームの矩形の大きさ//2に対する比率
    centroid_correction_template_match_filter_radius_ratio: float = 0.5
