# === Metashape 球面画像 → キューブマップ → PostShot(COLMAP) エクスポーター ===
# unified_fixed_v002.py をベースに PostShot 互換性を確保し、UIを日本語化
# バージョン: 1.0

import os
import re
import struct
import threading
import numpy as np
import cv2
import Metashape
import concurrent.futures
import time
import gc

# === PyQt5 プログレスバー（利用可能な場合） ===
_USE_GUI = False
try:
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QProgressBar, QApplication,
    )
    from PyQt5.QtCore import Qt
    _USE_GUI = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
}

CUBE_FACES = ["front", "right", "left", "top", "down", "back"]
_CUBE_SUFFIXES = tuple(f"_{f}" for f in CUBE_FACES)

# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """コンソールにタイムスタンプ付きログを出力"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _sanitize_filename(name: str) -> str:
    """ファイル名に使えない文字を除去"""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(". ")


def rotation_matrix_to_quaternion(R: np.ndarray) -> list:
    """回転行列 → クォータニオン (qw, qx, qy, qz) COLMAP規約"""
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qw, qx, qy, qz], dtype=float)
    quat /= np.linalg.norm(quat)
    return quat.tolist()


# ---------------------------------------------------------------------------
# 画像 I/O（マルチバイトパス対応）
# ---------------------------------------------------------------------------

def read_image_safe(path: str):
    """画像を安全に読み込む（Windows のマルチバイトパス対応）"""
    try:
        if os.name == "nt":
            with open(path, "rb") as f:
                buf = bytearray(f.read())
            return cv2.imdecode(np.asarray(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.imread(path)
    except (OSError, cv2.error) as e:
        _log(f"画像読み込みエラー: {path} — {e}")
        return None


def save_image_safe(image, path: str, params=None) -> bool:
    """画像を安全に保存する"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.name == "nt":
            ext = os.path.splitext(path)[1].lower()
            ok, buf = cv2.imencode(ext, image, params or [])
            if ok:
                with open(path, "wb") as f:
                    f.write(buf)
                return True
            return False
        return cv2.imwrite(path, image, params or [])
    except (OSError, cv2.error) as e:
        _log(f"画像保存エラー: {path} — {e}")
        return False


# ---------------------------------------------------------------------------
# エクイレクタングラー → キューブマップ面 変換
# ---------------------------------------------------------------------------

def equirect_to_cubemap_face(equirect_image, face_name: str,
                             face_size: int, fov: float = 90,
                             overlap: float = 10):
    """エクイレクタングラー画像からキューブマップの1面を生成"""
    if equirect_image is None:
        return None

    eq_h, eq_w = equirect_image.shape[:2]
    effective_fov = fov + overlap
    half_fov = np.radians(effective_fov / 2)
    tan_hf = np.tan(half_fov)

    coords = np.mgrid[0:face_size, 0:face_size].astype(np.float32)
    x_n = (coords[1] - face_size / 2) / (face_size / 2)
    y_n = -(coords[0] - face_size / 2) / (face_size / 2)

    xp = x_n * tan_hf
    yp = y_n * tan_hf
    zp = np.ones_like(xp)

    # 各面の方向マッピング（右手座標系: Y上, Z前, X右）
    dirs = {
        "front":  ( xp,  yp,  zp),
        "back":   (-xp,  yp, -zp),
        "right":  ( zp,  yp, -xp),
        "left":   (-zp,  yp,  xp),
        "top":    ( xp, -zp,  yp),
        "down":   ( xp,  zp, -yp),
    }
    if face_name not in dirs:
        _log(f"不明な面: {face_name}")
        return None

    xw, yw, zw = dirs[face_name]
    norm = np.sqrt(xw**2 + yw**2 + zw**2)
    xw, yw, zw = xw / norm, yw / norm, zw / norm

    lon = np.arctan2(xw, zw)
    lat = np.arcsin(np.clip(yw, -1, 1))

    eq_x = ((lon + np.pi) / (2 * np.pi)) * eq_w
    eq_y = ((np.pi / 2 - lat) / np.pi) * eq_h

    eq_x = eq_x % eq_w
    eq_y = np.clip(eq_y, 0, eq_h - 1)

    return cv2.remap(
        equirect_image,
        eq_x.astype(np.float32),
        eq_y.astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP,
    )


# ---------------------------------------------------------------------------
# カラー付きポイントクラウド抽出
# ---------------------------------------------------------------------------

def extract_colored_point_cloud(chunk, max_points=None) -> dict:
    """Metashapeのタイポイントからカラー付きスパースポイントクラウドを抽出"""
    _log("カラー付きスパースポイントクラウドを抽出中...")

    if not chunk.tie_points:
        _log("スパースポイントクラウドが存在しません")
        return {}

    points3D = {}
    total = len(chunk.tie_points.points)
    step = max(1, total // max_points) if max_points and total > max_points else 1
    valid_count = 0
    colored_count = 0

    for idx, point in enumerate(chunk.tie_points.points):
        if not point.valid:
            continue
        if step > 1 and idx % step != 0:
            continue

        valid_count += 1
        pid = idx + 1  # COLMAP は 1-based

        coord = point.coord
        xyz = [float(coord.x), float(coord.y), float(coord.z)]

        rgb = [128, 128, 128]
        found = False

        # ポイントカラーの取得を試行
        try:
            if hasattr(point, "color") and point.color is not None:
                c = point.color
                if hasattr(c, "__len__") and len(c) >= 3:
                    vals = list(c[:3])
                    mx = max(vals)
                    if mx <= 1.0:
                        rgb = [int(255 * v) for v in vals]
                    elif mx <= 255.0:
                        rgb = [int(v) for v in vals]
                    else:
                        rgb = [int(255 * v / mx) for v in vals]
                    found = True
                elif hasattr(c, "r") and hasattr(c, "g") and hasattr(c, "b"):
                    r, g, b = c.r, c.g, c.b
                    if max(r, g, b) <= 1.0:
                        rgb = [int(255 * r), int(255 * g), int(255 * b)]
                    else:
                        rgb = [int(r), int(g), int(b)]
                    found = True
        except (TypeError, ValueError, AttributeError):
            pass

        # トラック経由のフォールバック
        if not found:
            try:
                if (hasattr(chunk.tie_points, "tracks")
                        and idx < len(chunk.tie_points.tracks)):
                    tc = chunk.tie_points.tracks[idx].color
                    if tc is not None and hasattr(tc, "__len__") and len(tc) >= 3:
                        vals = list(tc[:3])
                        mx = max(vals)
                        if mx <= 1.0:
                            rgb = [int(255 * v) for v in vals]
                        else:
                            rgb = [int(v) for v in vals]
                        found = True
            except (TypeError, ValueError, AttributeError, IndexError):
                pass

        rgb = [max(0, min(255, int(v))) for v in rgb]
        if found and not (rgb[0] == rgb[1] == rgb[2] and 120 <= rgb[0] <= 135):
            colored_count += 1

        error = 0.0
        try:
            if hasattr(point, "error"):
                error = float(point.error)
        except (TypeError, ValueError):
            pass

        points3D[pid] = {
            "xyz": xyz,
            "rgb": rgb,
            "error": error,
            "image_ids": [],
            "point2D_idxs": [],
        }

    ratio = colored_count / valid_count if valid_count > 0 else 0
    _log(f"抽出完了: {len(points3D)}点 (カラー: {colored_count}, {ratio:.0%})")
    if ratio < 0.3:
        _log("カラー付き点が少ないです。Metashapeで Tools > Tie Points > Colorize を実行してください")

    return points3D


# ---------------------------------------------------------------------------
# COLMAP バイナリ書き込み（little-endian 固定）
# ---------------------------------------------------------------------------

def write_cameras_binary(cameras: dict, path: str) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cid, cam in cameras.items():
            f.write(struct.pack("<I", cid))
            f.write(struct.pack("<i", cam["model_id"]))
            f.write(struct.pack("<Q", cam["width"]))
            f.write(struct.pack("<Q", cam["height"]))
            for p in cam["params"]:
                f.write(struct.pack("<d", p))


def write_images_binary(images: dict, path: str) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for iid, img in images.items():
            f.write(struct.pack("<I", iid))
            for v in img["qvec"]:
                f.write(struct.pack("<d", v))
            for v in img["tvec"]:
                f.write(struct.pack("<d", v))
            f.write(struct.pack("<I", img["camera_id"]))
            f.write(img["name"].encode("utf-8") + b"\x00")
            n_pts = len(img["xys"])
            if len(img["point3D_ids"]) != n_pts:
                raise ValueError(
                    f"xys/point3D_ids 長さ不一致: {img['name']}")
            f.write(struct.pack("<Q", n_pts))
            for xy, p3d_id in zip(img["xys"], img["point3D_ids"]):
                f.write(struct.pack("<d", xy[0]))
                f.write(struct.pack("<d", xy[1]))
                # point3D_id は -1 (対応なし) を許容するため signed
                f.write(struct.pack("<q", p3d_id))


def write_points3D_binary(points3D: dict, path: str) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points3D)))
        for pid, pt in points3D.items():
            f.write(struct.pack("<Q", pid))
            for c in pt["xyz"]:
                f.write(struct.pack("<d", c))
            for c in pt["rgb"]:
                f.write(struct.pack("<B", c))
            f.write(struct.pack("<d", pt["error"]))
            n_obs = len(pt["image_ids"])
            if len(pt["point2D_idxs"]) != n_obs:
                raise ValueError(
                    f"image_ids/point2D_idxs 長さ不一致: point {pid}")
            f.write(struct.pack("<Q", n_obs))
            for im_id, p2d_idx in zip(pt["image_ids"], pt["point2D_idxs"]):
                f.write(struct.pack("<I", im_id))
                f.write(struct.pack("<I", p2d_idx))


# ---------------------------------------------------------------------------
# プログレストラッカー（PyQt5 GUI / コンソール フォールバック）
# ---------------------------------------------------------------------------

class ProgressDialog:
    """PyQt5 プログレスバー付きダイアログ。利用不可時はコンソール出力"""

    def __init__(self, title: str = "PostShot エクスポート"):
        self.start_time = time.time()
        self._cancelled = False
        self._dialog = None
        self._bar = None
        self._label = None
        self._time_label = None

        if _USE_GUI:
            try:
                # QApplicationが存在しない場合はMetashape内で既に存在するはず
                if QApplication.instance() is None:
                    _log("QApplication未初期化のためコンソールモードで動作します")
                    return
                self._dialog = QDialog()
                self._dialog.setWindowTitle(title)
                self._dialog.setMinimumWidth(480)
                self._dialog.setWindowFlags(
                    self._dialog.windowFlags() | Qt.WindowStaysOnTopHint
                )

                layout = QVBoxLayout()
                layout.setSpacing(12)
                layout.setContentsMargins(20, 20, 20, 20)

                self._label = QLabel("準備中...")
                self._label.setStyleSheet("font-size: 13px;")
                layout.addWidget(self._label)

                self._bar = QProgressBar()
                self._bar.setRange(0, 100)
                self._bar.setValue(0)
                self._bar.setTextVisible(True)
                self._bar.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #ccc;
                        border-radius: 4px;
                        text-align: center;
                        height: 22px;
                        background: #f0f0f0;
                    }
                    QProgressBar::chunk {
                        background: #3d7dd8;
                        border-radius: 3px;
                    }
                """)
                layout.addWidget(self._bar)

                self._time_label = QLabel("")
                self._time_label.setStyleSheet(
                    "font-size: 11px; color: #666;"
                )
                layout.addWidget(self._time_label)

                bottom = QHBoxLayout()
                cancel_btn = QPushButton("キャンセル")
                cancel_btn.setFixedWidth(100)
                cancel_btn.clicked.connect(self._on_cancel)
                bottom.addStretch()
                bottom.addWidget(cancel_btn)
                layout.addLayout(bottom)

                self._dialog.setLayout(layout)
                self._dialog.show()
                QApplication.processEvents()
            except Exception as e:
                _log(f"GUI初期化エラー（コンソールモードで継続）: {e}")
                self._dialog = None

    def _on_cancel(self):
        self._cancelled = True
        if self._dialog:
            self._dialog.close()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def update(self, percent: int, message: str = "") -> None:
        percent = max(0, min(100, percent))
        elapsed = time.time() - self.start_time
        if percent > 0:
            remaining = max(0, elapsed * (100 / percent) - elapsed)
            if remaining < 60:
                time_str = f"残り約 {remaining:.0f}秒"
            else:
                time_str = f"残り約 {remaining / 60:.1f}分"
        else:
            time_str = ""

        if self._dialog and self._bar:
            try:
                self._bar.setValue(percent)
                if self._label:
                    self._label.setText(message)
                if self._time_label:
                    self._time_label.setText(time_str)
                QApplication.processEvents()
            except RuntimeError:
                self._dialog = None

        _log(f"[{percent:3d}%] {message}  {time_str}")

    def close(self):
        if self._dialog:
            try:
                self._dialog.close()
            except RuntimeError:
                pass


# ---------------------------------------------------------------------------
# Metashape キューブマップカメラ作成
# ---------------------------------------------------------------------------

# キューブマップ面の方向定義
_FACE_DIRECTIONS = {
    "front": {
        "forward": [0, 0, 1],
        "up": [0, 1, 0],
        "right": [1, 0, 0],
    },
    "back": {
        "forward": [0, 0, -1],
        "up": [0, 1, 0],
        "right": [-1, 0, 0],
    },
    "right": {
        "forward": [1, 0, 0],
        "up": [0, 1, 0],
        "right": [0, 0, -1],
    },
    "left": {
        "forward": [-1, 0, 0],
        "up": [0, 1, 0],
        "right": [0, 0, 1],
    },
    "top": {
        "forward": [0, 1, 0],
        "up": [0, 0, -1],
        "right": [1, 0, 0],
    },
    "down": {
        "forward": [0, -1, 0],
        "up": [0, 0, 1],
        "right": [1, 0, 0],
    },
}


def create_cubemap_cameras(chunk, spherical_camera, face_image_paths: dict,
                           face_size: int, overlap: float = 10) -> list:
    """球面カメラに対応する6面のキューブマップカメラをMetashapeに作成"""
    camera_center = spherical_camera.center
    base_rotation = spherical_camera.transform.rotation()
    effective_fov = 90 + overlap
    focal = face_size / (2 * np.tan(np.radians(effective_fov / 2)))

    # 既存センサーの再利用を試みる
    sensor = None
    for s in chunk.sensors:
        if (s.type == Metashape.Sensor.Type.Frame
                and s.width == face_size and s.height == face_size
                and abs(s.calibration.f - focal) < 1.0):
            sensor = s
            break

    if sensor is None:
        sensor = chunk.addSensor()
        sensor.label = f"CubeMap_{face_size}px_fov{effective_fov:.0f}"
        sensor.type = Metashape.Sensor.Type.Frame
        sensor.width = face_size
        sensor.height = face_size
        cal = sensor.calibration
        cal.f = focal
        cal.cx = cal.cy = 0.0
        cal.k1 = cal.k2 = cal.k3 = 0.0
        cal.p1 = cal.p2 = 0.0

    created = []
    for face_name, img_path in face_image_paths.items():
        if face_name not in _FACE_DIRECTIONS:
            continue
        try:
            cam = chunk.addCamera()
            cam.label = f"{spherical_camera.label}_{face_name}"
            cam.sensor = sensor
            cam.photo = Metashape.Photo()
            cam.photo.path = img_path

            d = _FACE_DIRECTIONS[face_name]
            w_fwd = base_rotation * Metashape.Vector(d["forward"])
            w_up = base_rotation * Metashape.Vector(d["up"])
            w_rt = base_rotation * Metashape.Vector(d["right"])

            rot = Metashape.Matrix([
                [w_rt.x, w_rt.y, w_rt.z],
                [w_up.x, w_up.y, w_up.z],
                [w_fwd.x, w_fwd.y, w_fwd.z],
            ])
            cam.transform = (Metashape.Matrix.Translation(camera_center)
                             * Metashape.Matrix.Rotation(rot))
            created.append(cam)
        except (AttributeError, RuntimeError) as e:
            _log(f"カメラ作成エラー ({face_name}): {e}")
    return created


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def process_spherical_to_postshot(
    chunk,
    output_folder: str,
    face_size: int = None,
    overlap: float = 10,
    file_format: str = "jpg",
    quality: int = 95,
    max_points: int = 50000,
    camera_threads: int = None,
    progress: ProgressDialog = None,
) -> bool:
    """球面カメラ → キューブマップ → COLMAP(PostShot互換) エクスポート"""
    _process_start = time.time()

    def _update(pct, msg):
        if progress:
            if progress.cancelled:
                raise InterruptedError("ユーザーによりキャンセルされました")
            progress.update(pct, msg)
        else:
            _log(f"[{pct:3d}%] {msg}")

    _log("PostShot エクスポートを開始します")

    # フォルダ作成
    os.makedirs(output_folder, exist_ok=True)
    images_folder = os.path.join(output_folder, "images")
    os.makedirs(images_folder, exist_ok=True)
    sparse_folder = os.path.join(output_folder, "sparse", "0")
    os.makedirs(sparse_folder, exist_ok=True)

    # --- ステージ1: カメラ分析 (5%) ---
    _update(5, "球面カメラを検出中...")

    spherical_cameras = []
    existing_cube = []
    for cam in chunk.cameras:
        if cam.transform and cam.photo and cam.enabled:
            if any(cam.label.endswith(s) for s in _CUBE_SUFFIXES):
                existing_cube.append(cam)
            elif (hasattr(cam, "sensor") and cam.sensor
                  and hasattr(cam.sensor, "type")
                  and cam.sensor.type == Metashape.Sensor.Type.Spherical):
                spherical_cameras.append(cam)
            elif not any(cam.label.endswith(s) for s in _CUBE_SUFFIXES):
                # センサー型が不明な場合もラベルベースで判定
                spherical_cameras.append(cam)

    _log(f"検出: 球面カメラ {len(spherical_cameras)}台, "
         f"既存キューブマップ {len(existing_cube)}台")

    if not spherical_cameras:
        _log("球面カメラが見つかりません。カメラのアライメントを確認してください。")
        return False

    if existing_cube:
        _log(f"既存キューブマップカメラ {len(existing_cube)}台を削除中...")
        for cam in existing_cube:
            chunk.remove(cam)

    # --- ステージ2: ポイントクラウド抽出 (15%) ---
    _update(15, "スパースポイントクラウドを抽出中...")
    points3D = extract_colored_point_cloud(chunk, max_points=max_points)

    # --- ステージ3: パラメータ準備 (20%) ---
    _update(20, "変換パラメータを準備中...")

    if camera_threads is None:
        camera_threads = min(len(spherical_cameras), os.cpu_count() or 1)

    if file_format.lower() in ("jpg", "jpeg"):
        save_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        file_ext = "jpg"
    elif file_format.lower() == "png":
        save_params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, 10 - quality // 10)]
        file_ext = "png"
    else:
        save_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        file_ext = "jpg"

    # ファイル名重複チェック用（スレッドセーフ）
    used_filenames = set()
    _filename_lock = threading.Lock()

    def _unique_filename(base_label, face_name):
        base = _sanitize_filename(f"{base_label}_{face_name}")
        candidate = f"{base}.{file_ext}"
        with _filename_lock:
            if candidate not in used_filenames:
                used_filenames.add(candidate)
                return candidate
            i = 1
            while True:
                candidate = f"{base}_{i}.{file_ext}"
                if candidate not in used_filenames:
                    used_filenames.add(candidate)
                    return candidate
                i += 1

    # --- ステージ4: キューブマップ面の生成 (20-60%) ---
    def _process_one_camera(cam_data):
        cam_idx, sph_cam = cam_data
        result = {"camera": sph_cam, "face_images": {},
                  "face_size_actual": None, "error": None}
        try:
            img = read_image_safe(sph_cam.photo.path)
            if img is None:
                result["error"] = "画像を読み込めません"
                return result

            eq_h, eq_w = img.shape[:2]
            if face_size is None:
                sz = min(max(eq_w // 4, 512), 2048)
                sz = 2 ** int(np.log2(sz) + 0.5)
            else:
                sz = face_size

            result["face_size_actual"] = sz

            for fn in CUBE_FACES:
                try:
                    face_img = equirect_to_cubemap_face(img, fn, sz,
                                                        fov=90, overlap=overlap)
                    if face_img is None:
                        continue
                    fname = _unique_filename(sph_cam.label, fn)
                    out_path = os.path.join(images_folder, fname)
                    if save_image_safe(face_img, out_path, save_params):
                        result["face_images"][fn] = out_path
                    del face_img
                except (cv2.error, OSError, ValueError) as e:
                    _log(f"面生成エラー ({sph_cam.label}/{fn}): {e}")
            del img
            gc.collect()
        except (OSError, RuntimeError) as e:
            result["error"] = str(e)
        return result

    _update(20, f"{len(spherical_cameras)}台のカメラからキューブマップを生成中...")

    all_results = []
    if camera_threads <= 1:
        for i, cam in enumerate(spherical_cameras):
            pct = 20 + int((i / len(spherical_cameras)) * 40)
            _update(pct, f"キューブマップ生成: {cam.label} "
                         f"({i + 1}/{len(spherical_cameras)})")
            all_results.append(_process_one_camera((i, cam)))
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=camera_threads) as ex:
            futs = {ex.submit(_process_one_camera, (i, c)): c
                    for i, c in enumerate(spherical_cameras)}
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                done += 1
                pct = 20 + int((done / len(spherical_cameras)) * 40)
                _update(pct, f"キューブマップ生成完了: "
                             f"{futs[fut].label} ({done}/{len(spherical_cameras)})")
                try:
                    all_results.append(fut.result())
                except Exception as e:
                    all_results.append({
                        "camera": futs[fut], "face_images": {},
                        "face_size_actual": None, "error": str(e),
                    })

    # 6面揃っている結果のみ採用
    successful = [r for r in all_results
                  if not r["error"] and len(r["face_images"]) == len(CUBE_FACES)]
    partial = [r for r in all_results
               if not r["error"] and 0 < len(r["face_images"]) < len(CUBE_FACES)]

    if partial:
        _log(f"{len(partial)}台のカメラで一部の面生成に失敗 — スキップします")
    if not successful:
        _log("有効なキューブマップを生成できませんでした")
        return False

    total_faces = sum(len(r["face_images"]) for r in successful)
    _log(f"キューブマップ生成完了: {total_faces}面 ({len(successful)}台)")

    # --- ステージ5: Metashapeカメラ作成 (60-75%) ---
    _update(60, "Metashapeにキューブマップカメラを作成中...")
    all_new_cameras = []
    for i, r in enumerate(successful):
        pct = 60 + int((i / len(successful)) * 15)
        _update(pct, f"カメラ作成: {r['camera'].label}")
        try:
            cams = create_cubemap_cameras(
                chunk, r["camera"], r["face_images"],
                r["face_size_actual"], overlap)
            all_new_cameras.extend(cams)
        except (AttributeError, RuntimeError) as e:
            _log(f"カメラ作成エラー ({r['camera'].label}): {e}")

    _log(f"Metashapeカメラ作成完了: {len(all_new_cameras)}台")

    # --- ステージ6: COLMAP構造作成 (75-90%) ---
    _update(75, "COLMAPデータを構築中...")

    cameras_colmap = {}
    images_colmap = {}
    cam_key_to_id = {}
    next_cam_id = 1
    next_img_id = 1

    for r in successful:
        center = r["camera"].center
        center_np = np.array([center.x, center.y, center.z])
        base_rot = r["camera"].transform.rotation()
        sz = r["face_size_actual"]
        eff_fov = 90 + overlap
        focal = sz / (2 * np.tan(np.radians(eff_fov / 2)))
        cx = cy = sz / 2.0

        key = (sz, sz, focal, cx, cy)
        if key not in cam_key_to_id:
            cam_key_to_id[key] = next_cam_id
            cameras_colmap[next_cam_id] = {
                "model_id": CAMERA_MODEL_IDS["PINHOLE"],
                "width": sz,
                "height": sz,
                "params": [focal, focal, cx, cy],
            }
            cam_id = next_cam_id
            next_cam_id += 1
        else:
            cam_id = cam_key_to_id[key]

        for fn, img_path in r["face_images"].items():
            if fn not in _FACE_DIRECTIONS:
                continue

            d = _FACE_DIRECTIONS[fn]
            base_mat = np.array([
                [base_rot[0, 0], base_rot[0, 1], base_rot[0, 2]],
                [base_rot[1, 0], base_rot[1, 1], base_rot[1, 2]],
                [base_rot[2, 0], base_rot[2, 1], base_rot[2, 2]],
            ])

            w_fwd = base_mat @ np.array(d["forward"])
            w_up = base_mat @ np.array(d["up"])
            w_rt = base_mat @ np.array(d["right"])

            # COLMAP の R は world-to-camera
            R_w2c = np.array([w_rt, w_up, w_fwd])
            tvec = -R_w2c @ center_np
            qvec = rotation_matrix_to_quaternion(R_w2c)

            images_colmap[next_img_id] = {
                "qvec": qvec,
                "tvec": tvec.tolist(),
                "camera_id": cam_id,
                "name": os.path.basename(img_path),
                "xys": [],
                "point3D_ids": [],
            }
            next_img_id += 1

    # --- ステージ7: ファイル書き込み (90-98%) ---
    _update(90, "COLMAPファイルを書き込み中...")

    write_cameras_binary(cameras_colmap,
                         os.path.join(sparse_folder, "cameras.bin"))
    _log(f"cameras.bin 保存完了 ({len(cameras_colmap)}タイプ)")

    _update(93, "images.bin を書き込み中...")
    write_images_binary(images_colmap,
                        os.path.join(sparse_folder, "images.bin"))
    _log(f"images.bin 保存完了 ({len(images_colmap)}枚)")

    _update(96, "points3D.bin を書き込み中...")
    write_points3D_binary(points3D,
                          os.path.join(sparse_folder, "points3D.bin"))
    _log(f"points3D.bin 保存完了 ({len(points3D)}点)")

    # --- ステージ8: README生成 (98-100%) ---
    _update(98, "READMEを生成中...")

    sample = successful[0]
    sz_actual = sample["face_size_actual"]
    eff_fov = 90 + overlap
    focal_actual = sz_actual / (2 * np.tan(np.radians(eff_fov / 2)))
    colored = sum(1 for p in points3D.values() if p["rgb"] != [128, 128, 128])
    c_ratio = colored / len(points3D) if points3D else 0

    with open(os.path.join(output_folder, "README.txt"), "w",
              encoding="utf-8") as f:
        f.write("PostShot 用 COLMAP エクスポート\n")
        f.write("=" * 40 + "\n\n")
        f.write("使い方:\n")
        f.write("  このフォルダを PostShot にドラッグ＆ドロップしてください。\n")
        f.write("  Camera Poses と Sparse Points が自動検出されます。\n\n")
        f.write("フォルダ構造:\n")
        f.write("  images/          キューブマップ画像\n")
        f.write("  sparse/0/        COLMAP バイナリデータ\n")
        f.write("    cameras.bin    カメラ内部パラメータ\n")
        f.write("    images.bin     カメラ位置・姿勢\n")
        f.write("    points3D.bin   スパースポイントクラウド\n\n")
        f.write("パラメータ:\n")
        f.write(f"  球面カメラ数: {len(spherical_cameras)}\n")
        f.write(f"  生成面数:     {total_faces}\n")
        f.write(f"  面サイズ:     {sz_actual}px\n")
        f.write(f"  オーバーラップ: {overlap}°\n")
        f.write(f"  実効FOV:      {eff_fov}°\n")
        f.write(f"  焦点距離:     {focal_actual:.2f}px\n")
        f.write(f"  画像形式:     {file_format.upper()} (品質: {quality})\n")
        f.write(f"  3D点数:       {len(points3D)} (カラー: {c_ratio:.0%})\n")

    _update(100, "エクスポート完了")
    elapsed = time.time() - _process_start
    _log(f"完了: {output_folder} ({elapsed / 60:.1f}分)")
    return True


# ---------------------------------------------------------------------------
# GUI エントリポイント（Metashapeメニューから実行）
# ---------------------------------------------------------------------------

def main():
    doc = Metashape.app.document
    chunk = doc.chunk

    if not chunk:
        Metashape.app.messageBox("チャンクが見つかりません。\n"
                                 "プロジェクトを開いてチャンクを選択してください。")
        return

    # カメラ検出
    spherical = []
    existing_cube = []
    for cam in chunk.cameras:
        if cam.transform and cam.photo and cam.enabled:
            if any(cam.label.endswith(s) for s in _CUBE_SUFFIXES):
                existing_cube.append(cam)
            else:
                spherical.append(cam)

    if not spherical:
        Metashape.app.messageBox(
            "球面カメラが見つかりません。\n\n"
            "カメラのアライメント (Align Cameras) を\n"
            "実行してから再度お試しください。"
        )
        return

    # 確認ダイアログ
    msg = (f"PostShot 用エクスポート\n\n"
           f"検出されたカメラ:\n"
           f"  球面カメラ: {len(spherical)}台\n")
    if existing_cube:
        msg += f"  既存キューブマップ: {len(existing_cube)}台 (削除されます)\n"
    msg += (f"\n生成される画像: 最大 {len(spherical) * 6}枚\n\n"
            f"続行しますか?")

    if not Metashape.app.getBool(msg):
        return

    # 出力先選択
    output_folder = Metashape.app.getExistingDirectory(
        "PostShot エクスポート先フォルダを選択")
    if not output_folder:
        return

    # オーバーラップ設定
    try:
        overlap = Metashape.app.getFloat("オーバーラップ (度):", 10.0)
    except (TypeError, ValueError):
        overlap = 10.0
    if overlap is None:
        overlap = 10.0

    # 面サイズ設定
    try:
        size_choice = Metashape.app.getInt(
            "面サイズ:\n"
            "1 — 自動 (推奨)\n"
            "2 — 1024px\n"
            "3 — 2048px\n"
            "4 — 4096px",
            1, 1, 4)
    except (TypeError, ValueError):
        size_choice = 1

    face_size = {2: 1024, 3: 2048, 4: 4096}.get(size_choice)

    # ポイントクラウド制限
    max_points = None
    if chunk.tie_points and len(chunk.tie_points.points) > 50000:
        if Metashape.app.getBool(
                f"スパースポイントクラウドに "
                f"{len(chunk.tie_points.points):,}点あります。\n"
                f"50,000点に制限して高速化しますか?"):
            max_points = 50000

    # スレッド数
    cpu = os.cpu_count() or 1
    cam_threads = min(len(spherical), max(1, cpu // 2))

    # 最終確認
    final = (f"エクスポート設定の確認\n\n"
             f"出力先: {output_folder}\n"
             f"球面カメラ: {len(spherical)}台\n"
             f"面サイズ: {'自動' if face_size is None else f'{face_size}px'}\n"
             f"オーバーラップ: {overlap}°\n"
             f"画像形式: JPEG (品質 95)\n"
             f"スレッド数: {cam_threads}\n\n"
             f"処理を開始しますか?")

    if not Metashape.app.getBool(final):
        return

    # 実行
    prog = ProgressDialog("PostShot エクスポート")

    try:
        ok = process_spherical_to_postshot(
            chunk=chunk,
            output_folder=output_folder,
            face_size=face_size,
            overlap=overlap,
            file_format="jpg",
            quality=95,
            max_points=max_points,
            camera_threads=cam_threads,
            progress=prog,
        )

        prog.close()

        if ok:
            elapsed = time.time() - prog.start_time
            Metashape.app.messageBox(
                f"エクスポートが完了しました。\n\n"
                f"処理時間: {elapsed / 60:.1f}分\n"
                f"出力先: {output_folder}\n\n"
                f"PostShot でこのフォルダを\n"
                f"ドラッグ＆ドロップしてください。"
            )
        else:
            Metashape.app.messageBox(
                "エクスポートに失敗しました。\n"
                "コンソール出力を確認してください。"
            )

    except InterruptedError:
        prog.close()
        _log("ユーザーによりキャンセルされました")
        Metashape.app.messageBox("処理がキャンセルされました。")

    except Exception as e:
        prog.close()
        import traceback
        _log(f"エラー: {e}")
        _log(traceback.format_exc())
        Metashape.app.messageBox(f"エラーが発生しました:\n{e}")


if __name__ == "__main__":
    main()
