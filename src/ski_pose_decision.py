#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
from uuid import uuid4
import pickle

from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment


import torch
if 'weights_only' in torch.load.__code__.co_varnames:   # only on Torch>=2.6
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)  # RTMPose .pth requires pickle
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load


# --- MMPose (RTMPose) ---
# pip install mmpose==1.3.2 mmengine==0.10.4 mmcv==2.1.0

from mmpose.apis import init_model as mmpose_init_model
from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples

try:
    from mmaction.apis import init_recognizer
    import torch.nn.functional as F
    _HAVE_MMACTION = True
except Exception:
    _HAVE_MMACTION = False

# --- exports / sequence constants ---
SEQ_LEN = 64          # frames per clip (≈ 3 seconds)
STRIDE = 8           
MAX_SEQ_BACKLOG = 512
SKEL_EXPORT_DIR = "./export_skeleton_clips"
DEFAULT_CLASS_ID = -1 

# --- utility filters (introduced mainly in ckpt 1 & 2) ---
class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt) if dt > 0 else 1.0

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = 0.0
            return x
        dt = t - self.t_prev
        self.t_prev = t
        if dt <= 0:
            return self.x_prev

        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1 - alpha_d) * (self.dx_prev if self.dx_prev is not None else 0.0)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


# simple constant-velocity Kalman filter for a single (x,y) 
class JointKF:
    def __init__(self):
        self.x = np.zeros((4, 1), dtype=np.float32)  # [x, y, vx, vy]
        self.P = np.eye(4, dtype=np.float32) * 1.0
        self.Q = np.eye(4, dtype=np.float32) * 0.05
        self.R = np.eye(2, dtype=np.float32) * 2.0
        self.F = np.eye(4, dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.initialized = False

    def predict(self, dt):
        if not self.initialized:
            return None
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].ravel()


    def update(self, z):
        if z is None:
            return
        z = np.array(z, dtype=np.float32).reshape(2, 1)
        if not self.initialized:
            self.x[:2] = z
            self.initialized = True
            return
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ self.x)
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P



# --- core analyzer ---
class SkiPoseAnalyzer:
    def __init__(self,
                 yolo_model='yolov8m.pt',
                 rtmpose_cfg='rtmpose-m_8xb256-420e_body8-256x192.py',
                 rtmpose_ckpt='rtmpose-m_simcc-body7_pt-body7_420e-256x192.pth',
                 confidence=0.5, bbox_padding=0.25, imgsz=960, enable_logging=True,
                 oks_sigma_body=(0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72,
                                 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89),
                 enable_export=True,
                 action_cfg=None,
                 action_ckpt=None):
        self.confidence = confidence
        self.bbox_padding = bbox_padding
        self.imgsz = imgsz
        self.enable_logging = enable_logging
        self.enable_export = enable_export

        # Tracks: each id in dict
        self.tracks = {}
        self.next_id = 0
        self.max_miss = 20                     # frames needed to keep track alive during occlusion
        self.low_score_keep = 0.1 
        self.history_len = 5

        self.oks_sigma = np.array(oks_sigma_body, dtype=np.float32)
        self.prev_time = None

        self.frame_logs = []
        self.session_stats = {'start_time': None, 'end_time': None, 'processing_fps': 0, 'total_frames': 0}
        self.input_file = None
        self._vhw = None

        # --- Init models ---
        yolo_path = Path('../yolos') / yolo_model
        if not yolo_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {yolo_path.resolve()}")
        self.detector = YOLO(str(yolo_path.resolve()))

        # expect RTMPose config/ckpt under ../mmpose/
        mmpose_dir = Path('../mmpose')
        cfg = mmpose_dir / rtmpose_cfg
        ckpt = mmpose_dir / rtmpose_ckpt
        if not cfg.exists() or not ckpt.exists():
            print(cfg)
            print(ckpt)
            raise FileNotFoundError("RTMPose config/ckpt not found under '../mmpose/'.")
        self.pose_model = mmpose_init_model(str(cfg), str(ckpt), device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
        # GPU  is definitely needed for training

        # ---- action recognizer (PoseC3D, optional) ----
        self.action_cfg = action_cfg
        self.action_ckpt = action_ckpt
        self.action_model = None
        if _HAVE_MMACTION and action_cfg and action_ckpt and os.path.exists(action_cfg) and os.path.exists(action_ckpt):
            try:
                self.action_model = init_recognizer(
                    action_cfg, action_ckpt,
                    device=('cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
                )
                self.action_model.eval()
                print("[info] PoseC3D action model loaded.")
            except Exception as e:
                print("[warn] failed to load action model:", e)
                self.action_model = None

    # --- association via OKS ---
    @staticmethod
    def _bbox_area(b):
        x1, y1, x2, y2 = b
        return max(1.0, float((x2 - x1) * (y2 - y1)))

    def oks(self, kpts_a, kpts_b, area):
        kpts_a = np.asarray(kpts_a, dtype=np.float32)
        kpts_b = np.asarray(kpts_b, dtype=np.float32)
        v = (kpts_a[:, 2] > 0.1) & (kpts_b[:, 2] > 0.1)
        if not np.any(v):
            return 0.0

        d2 = (kpts_a[v, 0] - kpts_b[v, 0]) ** 2 + (kpts_a[v, 1] - kpts_b[v, 1]) ** 2
        s2 = (2 * self.oks_sigma[v]) ** 2
        e = np.exp(-d2 / (s2 * (area + np.finfo(np.float32).eps)))
        return float(np.mean(e))

    # --- Pose estimation with RTMPose ---
    def infer_pose_batch(self, frame, bboxes):
        """
        RTMPose inference:
        - Accepts Nx4 xyxy int/float boxes
        - Converts to list[np.ndarray(shape=(4,), dtype=float32)]
        """
        if bboxes is None or len(bboxes) == 0:
            return []

        # ensure list of float32 np.ndarrays of shape (4,)
        boxes_list = []
        for b in bboxes:
            b = np.asarray(b, dtype=np.float32).reshape(-1)
            if b.shape[0] != 4:
                continue
            boxes_list.append(b)
        if len(boxes_list) == 0:
            return [None] * len(bboxes)

        try:
            # preferred path: batched inference
            data_samples = inference_topdown(
                self.pose_model,
                frame,
                boxes_list,
                bbox_format='xyxy'
            )
        except Exception:
            # fallback: run one-by-one
            data_samples = []
            for b in boxes_list:
                ds = inference_topdown(
                    self.pose_model, frame, [b], bbox_format='xyxy'
                )
                data_samples.extend(ds)

        # data_samples length should match boxes_list
        kpt_list = []

        for ds in data_samples:
            try:
                inst = ds.pred_instances
                keypoints = getattr(inst, 'keypoints', None)
                scores = getattr(inst, 'keypoint_scores', None)
                if keypoints is None or len(keypoints) == 0:
                    kpt_list.append(None)
                    continue
                kps = keypoints[0]
                sc = scores[0] if scores is not None and len(scores) > 0 else np.ones((kps.shape[0],), dtype=np.float32) * 0.5
                kpts = [(int(kps[i, 0]), int(kps[i, 1]), float(sc[i])) for i in range(kps.shape[0])]
                kpt_list.append(kpts)
            except Exception:
                kpt_list.append(None)

        if len(kpt_list) < len(boxes_list):
            kpt_list += [None] * (len(boxes_list) - len(kpt_list))
        return kpt_list

    # --- tracking ---
    def step(self, frame):
        t0 = time.time()
        if self.session_stats['start_time'] is None:
            self.session_stats['start_time'] = t0

        # 1: detect people
        dets = self.detector(frame, imgsz=self.imgsz, conf=min(self.confidence, 0.6), iou=0.6, classes=[0], verbose=False)[0]
        xyxy = dets.boxes.xyxy.cpu().numpy().astype(int) if dets.boxes is not None else np.zeros((0, 4), dtype=int)
        confs = dets.boxes.conf.cpu().numpy() if dets.boxes is not None else np.zeros((0,), dtype=float)

        keep = confs >= self.low_score_keep
        xyxy = xyxy[keep]
        confs = confs[keep]

        # 2: pose estimation (batched)
        poses = []
        if len(xyxy) > 0:
            poses = self.infer_pose_batch(frame, xyxy)
        else:
            poses = []

        # 3: build cost matrix via OKS
        active_ids = [tid for tid, tr in self.tracks.items() if tr['miss'] < self.max_miss]
        M, N = len(active_ids), len(xyxy)
        cost = np.ones((M, N), dtype=np.float32)

        for i, tid in enumerate(active_ids):
            tr = self.tracks[tid]
            for j in range(N):
                if poses[j] is None:
                    cost[i, j] = 1.0
                    continue
                area = self._bbox_area(xyxy[j])
                oks_val = self.oks(tr['kpts_pred'], poses[j], area)
                cost[i, j] = 1.0 - oks_val

        assigned_tracks = {}
        used_det = set()
        if M > 0 and N > 0:
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] < 0.6:  # OKS >= 0.4 association threshold
                    assigned_tracks[active_ids[i]] = j
                    used_det.add(j)

        # 4: update matched tracks
        now = time.time()
        dt = 1/30.0 if self.prev_time is None else max(1e-3, now - self.prev_time)
        self.prev_time = now

        for tid, j in assigned_tracks.items():
            kpts = poses[j]
            bbox = tuple(map(int, xyxy[j]))
            self._update_track(tid, kpts, bbox, dt)
            self.tracks[tid]['miss'] = 0
            self.tracks[tid]['conf'] = float(confs[j])

        # 5: create new tracks for unused detections with valid pose
        for j in range(N):
            if j in used_det:
                continue
            if poses[j] is None:
                continue
            tid = self.next_id
            self.next_id += 1
            self._init_track(tid, poses[j], tuple(map(int, xyxy[j])), now)

        # 6: for unmatched (occluded) tracks: predict with KF
        for tid, tr in self.tracks.items():
            if tid in assigned_tracks:
                continue
            tr['miss'] += 1
            pred = []
            for k in range(len(tr['kfs'])):
                p = tr['kfs'][k].predict(dt)
                if p is None:
                    px, py = tr['kpts_pred'][k][0], tr['kpts_pred'][k][1]
                else:
                    px, py = float(p[0]), float(p[1])
                pred.append((int(px), int(py), tr['kpts_pred'][k][2] * 0.95))
            tr['kpts_pred'] = pred
            tr['history'].append({'bbox': tr['bbox'], 'kpts': tr['kpts_pred']})
            if len(tr['history']) > self.history_len:
                tr['history'].popleft()

            # --- added: buffer predicted joints too ---
            k_arr = np.array([[px, py] for (px, py, vv) in pred], dtype=np.float32)
            s_arr = np.array([vv for (_, _, vv) in pred], dtype=np.float32)
            tr['kp_buf'].append(k_arr)
            tr['ks_buf'].append(s_arr)
            tr['frame_idx'] += 1

        # 7: optional export of sliding-window skeleton clips
        if self._vhw is not None:
            vw, vh = self._vhw
            for tid, tr in self.tracks.items():
                if self.enable_export and tr['conf'] >= 0.2:
                    _ = self._maybe_export_clip(tr, vw, vh, label=DEFAULT_CLASS_ID)
                if self.action_model is not None and len(tr['kp_buf']) >= SEQ_LEN:
                    pred = self._infer_action(tr, vw, vh)
                    if pred is not None:
                        tr['action_pred'] = pred

        # 8: collect outputs
        results = []
        for tid, tr in list(self.tracks.items()):
            if tr['miss'] > self.max_miss * 2:
                del self.tracks[tid]
                continue
            results.append({
                'track_id': tid,
                'bbox': tr['bbox'],
                'confidence': tr['conf'],
                'landmarks': tr['kpts_pred'],
                'held': tr['miss'] > 0,
                'held_pose': tr['miss'] > 0,
                'action_pred': tr.get('action_pred', None)
            })

        t1 = time.time()
        self.session_stats['total_frames'] += 1
        fps = 1.0 / max(1e-6, (t1 - t0))
        self.session_stats['processing_fps'] = 0.9 * self.session_stats['processing_fps'] + 0.1 * fps if self.session_stats['processing_fps'] else fps
        return results

    def _init_track(self, tid, kpts, bbox, now_t):
        kfs = [JointKF() for _ in kpts]
        euros = [(OneEuro(1.2, 0.02, 1.0), OneEuro(1.2, 0.02, 1.0)) for _ in kpts]
        for i, (x, y, v) in enumerate(kpts):
            kfs[i].update((x, y))
        self.tracks[tid] = {
            'id': tid,
            'bbox': bbox,
            'kpts_pred': kpts,
            'kfs': kfs,
            'euros': euros,
            'miss': 0,
            'conf': 1.0,
            'history': deque(maxlen=self.history_len),
            't_last': now_t,

            # ---- NEW buffers ----
            'kp_buf': deque(maxlen=MAX_SEQ_BACKLOG),  # [K,2] per frame
            'ks_buf': deque(maxlen=MAX_SEQ_BACKLOG),  # [K] per frame (scores)
            'frame_idx': 0,
            'last_export_at': -9999,
        }

        # seed buffers with initial observation
        k_arr = np.array([[x, y] for (x, y, v) in kpts], dtype=np.float32)
        s_arr = np.array([v for (_, _, v) in kpts], dtype=np.float32)
        self.tracks[tid]['kp_buf'].append(k_arr)
        self.tracks[tid]['ks_buf'].append(s_arr)

    def _update_track(self, tid, kpts, bbox, dt):
        tr = self.tracks[tid]
        tr['bbox'] = bbox
        smoothed = []
        tnow = time.time()
        for i, (x, y, v) in enumerate(kpts):
            tr['kfs'][i].predict(dt)
            tr['kfs'][i].update((x, y))
            est = tr['kfs'][i].x[:2].ravel()
            ex = tr['euros'][i][0](float(est[0]), tnow)
            ey = tr['euros'][i][1](float(est[1]), tnow)
            smoothed.append((int(ex), int(ey), float(v)))
        tr['kpts_pred'] = smoothed
        tr['history'].append({'bbox': bbox, 'kpts': smoothed})
        if len(tr['history']) > self.history_len:
            tr['history'].popleft()

        # --- append to buffers ---
        k_arr = np.array([[x, y] for (x, y, v) in smoothed], dtype=np.float32)
        s_arr = np.array([v for (_, _, v) in smoothed], dtype=np.float32)
        tr['kp_buf'].append(k_arr)
        tr['ks_buf'].append(s_arr)
        tr['frame_idx'] += 1

    # --- export a PoseC3D skeleton clip ---
    def _maybe_export_clip(self, tr, video_w, video_h, label=DEFAULT_CLASS_ID):
        if len(tr['kp_buf']) < SEQ_LEN:
            return None
        if tr['frame_idx'] - tr['last_export_at'] < STRIDE:
            return None

        kp_seq = list(tr['kp_buf'])[-SEQ_LEN:]
        ks_seq = list(tr['ks_buf'])[-SEQ_LEN:]
        T = len(kp_seq)
        if T < SEQ_LEN:
            return None
        K = kp_seq[0].shape[0]
        N = 1

        keypoint = np.stack(kp_seq, axis=0).reshape(T, N, K, 2).astype(np.float32)
        keypoint_score = np.stack(ks_seq, axis=0).reshape(T, N, K).astype(np.float32)

        # deterministic naming with frame range
        end_frame = tr['frame_idx'] - 1
        start_frame = end_frame - (SEQ_LEN - 1)
        clip_base = f"{self.input_file.stem}_tid{tr['id']}_s{start_frame}_e{end_frame}"
        out_path = os.path.join(SKEL_EXPORT_DIR, clip_base + ".npz")

        np.savez_compressed(
            out_path,
            keypoint=keypoint,
            keypoint_score=keypoint_score,
            total_frames=np.int32(T),
            img_shape=np.array([video_h, video_w], dtype=np.int32),
            label=np.int32(label),
            meta=np.array([tr['id']], dtype=np.int32)
        )
        tr['last_export_at'] = tr['frame_idx']

        # write manifest row
        if hasattr(self, "_manifest_path") and hasattr(self, "_fps"):
            start_sec = start_frame / max(1e-6, self._fps)
            end_sec = end_frame / max(1e-6, self._fps)
            with open(self._manifest_path, "a") as f:
                f.write(f"{clip_base}.npz,{os.path.abspath(out_path)},"
                        f"{os.path.abspath(str(self.input_file))},{tr['id']},"
                        f"{start_frame},{end_frame},{start_sec:.3f},{end_sec:.3f},"
                        f"{datetime.utcnow().isoformat()}Z\n")

        return out_path


    # PoseC3D inference (optional)
    def _infer_action(self, tr, vw, vh):
        if self.action_model is None:
            return None
        if len(tr['kp_buf']) < SEQ_LEN:
            return None
        kp_seq = list(tr['kp_buf'])[-SEQ_LEN:]
        ks_seq = list(tr['ks_buf'])[-SEQ_LEN:]
        T, K = len(kp_seq), kp_seq[0].shape[0]
        keypoint = np.stack(kp_seq, 0).reshape(T, 1, K, 2).astype(np.float32)
        keypoint_score = np.stack(ks_seq, 0).reshape(T, 1, K).astype(np.float32)

        inputs = dict(
            keypoint=torch.from_numpy(keypoint).unsqueeze(0),
            keypoint_score=torch.from_numpy(keypoint_score).unsqueeze(0),
            img_shape=torch.tensor([[vh, vw]]),
            total_frames=torch.tensor([T]),
        )
        with torch.no_grad():
            logits = self.action_model(inputs, return_loss=False)[0]  # numpy list or array
            logits = torch.tensor(logits)
            prob = torch.softmax(logits, dim=0).cpu().numpy()
        pred_id = int(prob.argmax())
        pred_conf = float(prob[pred_id])
        return (pred_id, pred_conf)

    # --- Drawing to video canvas!!! ---
    @staticmethod
    def draw(frame, results):
        for r in results:
            x1, y1, x2, y2 = r['bbox']
            is_held = r.get('held', False)
            is_held_pose = r.get('held_pose', False)
            color = (0, 165, 255) if is_held else ((255, 165, 0) if is_held_pose else (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{r['track_id']} {r['confidence']:.2f}"
            if is_held: label += " (HELD)"
            elif is_held_pose: label += " (POSE-HELD)"

            act = r.get('action_pred')
            if act:
                names = ["clean", "late", "near-fall", "fall", "drag"]
                pred_id, pred_conf = act
                if 0 <= pred_id < len(names):
                    label += f" | {names[pred_id]}:{pred_conf:.2f}"

            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            k = r['landmarks']
            edges = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12),
                     (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
            for (a, b) in edges:
                if a < len(k) and b < len(k) and k[a][2] > 0.2 and k[b][2] > 0.2:
                    cv2.line(frame, (k[a][0], k[a][1]), (k[b][0], k[b][1]), (255, 0, 0), 2)
            for (x, y, v) in k:
                if v > 0.2:
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        return frame

    # --- I/O ---
    def process_video(self, video_path, output_path=None):
        self.input_file = Path(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._vhw = (w, h) 
        self._fps = float(fps)  # keep fps for manifest timing

        # manifest CSV 
        os.makedirs(SKEL_EXPORT_DIR, exist_ok=True)
        self._manifest_path = os.path.join(SKEL_EXPORT_DIR, "manifest.csv")
        if not os.path.exists(self._manifest_path):
            with open(self._manifest_path, "w") as f:
                f.write("clip,npz_path,video,track_id,start_frame,end_frame,"
                        "start_sec,end_sec,exported_at_iso\n")

        if output_path is None:
            output_path = video_path.replace('.mp4', '_rtmpose_tracked.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        frame_idx = 0  # frame counter for overlay and debugging
        prev = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = self.step(frame)
            vis = self.draw(frame, res)

            # overlays
            now = time.time()
            cv2.putText(
                vis, f"FPS: {1.0 / max(1e-6, now - prev):.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            cv2.putText(  # show current frame index
                vis, f"frame: {frame_idx}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
            )
            prev = now
            out.write(vis)
            frame_idx += 1  # increment after writing

        cap.release()
        out.release()
        self.session_stats['end_time'] = time.time()
        print(f"Output saved: {output_path}")

# --- take runtime params ---
def main():
    parser = argparse.ArgumentParser(description='Occlusion-Robust Ski Pose Tracker (YOLOv8 + RTMPose + OKS-KF + PoseC3D export)')
    parser.add_argument('--input', required=True, help='Input video')
    parser.add_argument('--output', help='Output video')
    parser.add_argument('--model', default='yolov8m.pt', help='YOLOv8 model file in ../yolos/')
    parser.add_argument('--rtm-cfg', default='rtmpose-m_8xb256-420e_body8-256x192.py', help='RTMPose config relative to ../mmpose/')
    parser.add_argument('--rtm-ckpt', default='rtmpose-m_simcc-body7_pt-body7_420e-256x192.pth', help='RTMPose checkpoint relative to ../mmpose/')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--no-export', action='store_true', help='Disable skeleton clip export')
    parser.add_argument('--action-cfg', default=None, help='(Optional) PoseC3D config for online inference')
    parser.add_argument('--action-ckpt', default=None, help='(Optional) PoseC3D checkpoint for online inference')
    args = parser.parse_args()

    analyzer = SkiPoseAnalyzer(
        yolo_model=args.model,
        rtmpose_cfg=args.rtm_cfg,
        rtmpose_ckpt=args.rtm_ckpt,
        confidence=args.conf,
        imgsz=args.imgsz,
        enable_export=not args.no_export,
        action_cfg=args.action_cfg,
        action_ckpt=args.action_ckpt
    )
    analyzer.process_video(args.input, args.output)


if __name__ == '__main__':
    main()
