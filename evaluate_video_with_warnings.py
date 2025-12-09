#!/usr/bin/env python3
"""
Evaluate trained PoseC3D model on a video and overlay warning text
when dangerous situations are detected.
"""

import argparse
import os
import sys
import tempfile
import numpy as np
import cv2
import torch
import mmengine
from pathlib import Path
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
from mmengine.utils import track_iter_progress
from mmaction.apis import (inference_skeleton, init_recognizer, pose_inference)
from mmaction.utils import frame_extract

if 'weights_only' in torch.load.__code__.co_varnames:   # only on Torch>=2.6
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)  # RTMPose .pth requires pickle
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load


# YOLOv8 for detection 
try:
    from ultralytics import YOLO
    _HAVE_YOLO = True
except ImportError:
    _HAVE_YOLO = False
    print("Warning: YOLOv8 not available, will use MMDetection instead")

# class labels
CLASS_LABELS = ['clean', 'late', 'near-fall', 'fall', 'drag']
DANGEROUS_CLASSES = [1, 2, 3, 4]  # late, near-fall, fall, drag
CLIP_LEN = 48 


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate PoseC3D model on video with warning overlays')
    parser.add_argument(
        '--video',
        default='videos/fall_instance_3.mp4',
        help='Input video file path')
    parser.add_argument(
        '--config',
        default='src/configs/posec3d_ski.py',
        help='Model config file path')
    parser.add_argument(
        '--checkpoint',
        default='mmaction2/work_dirs/posec3d_ski/best_acc_top1_epoch_60.pth',
        help='Model checkpoint file path (using best checkpoint by default)')
    parser.add_argument(
        '--out-video',
        default='video_outputs/temp.mp4',
        help='Output video file path')
    parser.add_argument(
        '--yolo-model',
        default='yolos/yolov8m.pt',
        help='YOLOv8 model path')
    parser.add_argument(
        '--pose-config',
        default='mmpose/rtmpose-m_8xb256-420e_body8-256x192.py',
        help='Pose estimation config')
    parser.add_argument(
        '--pose-checkpoint',
        default='mmpose/rtmpose-m_simcc-body7_pt-body7_420e-256x192.pth',
        help='Pose estimation checkpoint')
    parser.add_argument(
        '--device',
        type=str,
        default='mps' if torch.backends.mps.is_available() else 'cpu',
        help='Device (mps/cuda/cpu)')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.4,
        help='Human detection score threshold')
    parser.add_argument(
        '--danger-threshold',
        type=float,
        default=0.2,
        help='Probability threshold to show warning (sum of dangerous classes)')
    parser.add_argument(
        '--track-buffer',
        type=int,
        default=30,
        help='Frames to keep track alive during occlusion')
    return parser.parse_args()


def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def track_people_across_frames(det_results, iou_threshold=0.3):
    """
    Track people across frames using IoU-based association.
    """
    tracks = {}
    next_track_id = 0
    track_states = {} 
    
    for frame_idx, bboxes in enumerate(det_results):
        if len(bboxes) == 0:
            # update miss counts for all tracks
            for tid in list(track_states.keys()):
                track_states[tid]['miss_count'] += 1
                if track_states[tid]['miss_count'] > 30:
                    del track_states[tid]
            continue
        
        # build cost matrix for Hungarian algorithm
        active_tracks = {tid: state for tid, state in track_states.items() 
                        if state['miss_count'] < 10}
        
        if len(active_tracks) == 0:
            # create new tracks for all detections
            for det_idx in range(len(bboxes)):
                tid = next_track_id
                next_track_id += 1
                tracks[tid] = [(frame_idx, bboxes[det_idx], det_idx)]
                track_states[tid] = {'last_bbox': bboxes[det_idx], 'miss_count': 0}
        else:
            # match detections to existing tracks
            track_ids = list(active_tracks.keys())
            cost_matrix = np.ones((len(track_ids), len(bboxes)))
            
            for i, tid in enumerate(track_ids):
                last_bbox = track_states[tid]['last_bbox']
                for j, bbox in enumerate(bboxes):
                    iou = calculate_iou(last_bbox, bbox)
                    cost_matrix[i, j] = 1.0 - iou  # cost = 1 - IoU
            
            # Hungarian algorithm
            if len(track_ids) > 0 and len(bboxes) > 0:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                matched_tracks = set()
                matched_dets = set()
                
                for i, j in zip(row_ind, col_ind):
                    if cost_matrix[i, j] < (1.0 - iou_threshold):  # if IoU > threshold
                        tid = track_ids[i]
                        tracks[tid].append((frame_idx, bboxes[j], j))
                        track_states[tid]['last_bbox'] = bboxes[j]
                        track_states[tid]['miss_count'] = 0
                        matched_tracks.add(tid)
                        matched_dets.add(j)
                
                # make new tracks for unmatched detections
                for j in range(len(bboxes)):
                    if j not in matched_dets:
                        tid = next_track_id
                        next_track_id += 1
                        tracks[tid] = [(frame_idx, bboxes[j], j)]
                        track_states[tid] = {'last_bbox': bboxes[j], 'miss_count': 0}
                
                # update miss counts for unmatched tracks
                for tid in track_ids:
                    if tid not in matched_tracks:
                        track_states[tid]['miss_count'] += 1
                        if track_states[tid]['miss_count'] > 30:
                            if tid in track_states:
                                del track_states[tid]
    
    return tracks

# extract skeleton sequence for a tracked person
def extract_skeleton_for_track(track_data, pose_results, img_shape):
    h, w = img_shape
    keypoints_list = []
    scores_list = []
    
    for frame_idx, bbox, det_idx in track_data:
        if frame_idx >= len(pose_results):
            continue
        
        frame_poses = pose_results[frame_idx]
        if 'keypoints' not in frame_poses:
            continue
        
        kpts_array = frame_poses['keypoints']
        kpt_scores_array = frame_poses['keypoint_scores']
        
        if det_idx >= len(kpts_array):
            continue
        
        # extract keypoints for this person
        kpts = kpts_array[det_idx] 
        kpt_scores = kpt_scores_array[det_idx]
        
        keypoints_list.append(kpts)
        scores_list.append(kpt_scores)
    
    if len(keypoints_list) == 0:
        return None, None
    
    # put into sequences
    keypoint_seq = np.stack(keypoints_list, axis=0)
    keypoint_score_seq = np.stack(scores_list, axis=0)
    
    return keypoint_seq, keypoint_score_seq

# run action recognition on a person's skeleton sequence
def process_track_with_model(model, keypoint_seq, keypoint_score_seq, img_shape):
    if keypoint_seq is None or len(keypoint_seq) < CLIP_LEN:
        return None, None, None, None
    
    h, w = img_shape
    T = len(keypoint_seq)
    
    # if sequence is longer than CLIP_LEN, use  most recent CLIP_LEN frames
    if T > CLIP_LEN:
        keypoint_seq = keypoint_seq[-CLIP_LEN:]
        keypoint_score_seq = keypoint_score_seq[-CLIP_LEN:]
        T = CLIP_LEN
    

    pose_results_list = []
    for t in range(T):
        # each frame has 1 person with K keypoints
        kpts = keypoint_seq[t:t+1]
        kpt_scores = keypoint_score_seq[t:t+1]
        
        pose_results_list.append({
            'keypoints': kpts.astype(np.float16),
            'keypoint_scores': kpt_scores.astype(np.float16)
        })
    
    try:
        result = inference_skeleton(model, pose_results_list, (h, w))
        pred_scores = result.pred_score.cpu().numpy()
        
        # apply softmax to get probabilities
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.numpy()
        
        # apply softmax to convert logits to probabilities
        scores_sum = pred_scores.sum()
        if scores_sum < 0.99 or scores_sum > 1.01 or np.any(pred_scores < 0):
            pred_scores = torch.softmax(torch.from_numpy(pred_scores), dim=0).numpy()
        
        pred_class = int(pred_scores.argmax())
        pred_confidence = float(pred_scores.max())
        dangerous_prob = float(pred_scores[DANGEROUS_CLASSES].sum())
        
        return pred_class, pred_confidence, dangerous_prob, pred_scores
    except Exception as e:
        print(f'Warning: Error processing track: {e}')
        import traceback
        traceback.print_exc()
        return None, None, None, None


def overlay_warning(frame, text="WARNING!", position=(50, 50)):
    """Overlay warning text on frame."""
    # create a semi-transparent red background
    overlay = frame.copy()
    cv2.rectangle(overlay, (position[0]-10, position[1]-30), 
                  (position[0]+300, position[1]+10), (0, 0, 255), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # write text
    # font = cv2.FONT_HERSHEY_BOLD   # this font throws errors
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 4
    color = (255, 255, 255)
    
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame

# use YOLOv8 with tracking for person detection and tracking
def yolo_detection_with_tracking(yolo_model_path, frames, det_score_thr=0.4, device='cpu'):
    if not _HAVE_YOLO:
        raise ImportError("YOLOv8 not available. Install with: pip install ultralytics")
    
    yolo_path = Path(yolo_model_path)
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
    
    print(f'Loading YOLOv8 model: {yolo_path}')
    detector = YOLO(str(yolo_path))
    
    det_results = []
    track_ids_per_frame = []
    
    # check if botsort.yaml exists
    botsort_path = os.path.join(os.path.dirname(yolo_model_path), '..', 'botsort.yaml')
    if not os.path.exists(botsort_path):
        botsort_path = 'botsort.yaml' if os.path.exists('botsort.yaml') else None
    
    print('Running YOLOv8 detection with tracking...')
    for frame_idx, frame in enumerate(track_iter_progress(frames)):
        
        try:
            if botsort_path and frame_idx == 0:
                # first frame: init tracking
                results = detector.track(
                    frame, 
                    conf=det_score_thr, 
                    classes=[0], 
                    verbose=False, 
                    device=device,
                    persist=True,
                    tracker=botsort_path
                )
            else:
                # subsequent frames: continue tracking
                results = detector.track(
                    frame, 
                    conf=det_score_thr, 
                    classes=[0], 
                    verbose=False, 
                    device=device,
                    persist=True
                )
        except Exception as e:
            print(f'Warning: Tracking failed, using detection only: {e}')
            results = detector(frame, conf=det_score_thr, classes=[0], verbose=False, device=device)
        
        bboxes = []
        track_ids = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy() if boxes.id is not None else None
            
            valid = scores >= det_score_thr
            if np.any(valid):
                bboxes = xyxy[valid].astype(np.float32)
                if ids is not None:
                    track_ids = ids[valid].astype(int).tolist()
                else:
                    track_ids = [-1] * len(bboxes)
            else:
                bboxes = np.zeros((0, 4), dtype=np.float32)
                track_ids = []
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            track_ids = []
        
        det_results.append(bboxes)
        track_ids_per_frame.append(track_ids)
    
    return det_results, track_ids_per_frame


def main():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, args.video) if not os.path.isabs(args.video) else args.video
    config_path = os.path.join(script_dir, args.config) if not os.path.isabs(args.config) else args.config
    checkpoint_path = os.path.join(script_dir, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    out_video_path = os.path.join(script_dir, args.out_video) if not os.path.isabs(args.out_video) else args.out_video
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video not found: {video_path}')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config not found: {config_path}')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    
    print(f'Loading video: {video_path}')
    print(f'Using config: {config_path}')
    print(f'Using checkpoint: {checkpoint_path}')
    print(f'Output will be saved to: {out_video_path}')
    
    # extract frames
    tmp_dir = tempfile.TemporaryDirectory()
    print('Extracting frames from video...')
    frame_paths, frames = frame_extract(video_path, short_side=480, out_dir=tmp_dir.name)
    num_frames = len(frames)
    h, w, _ = frames[0].shape
    print(f'Extracted {num_frames} frames (resolution: {w}x{h})')
    
    # human detection with tracking
    print('Running human detection with tracking...')
    yolo_model_path = os.path.join(script_dir, args.yolo_model) if not os.path.isabs(args.yolo_model) else args.yolo_model
    
    if not _HAVE_YOLO or not os.path.exists(yolo_model_path):
        raise ValueError("YOLOv8 is required for person tracking. Please install ultralytics.")
    
    det_results, track_ids_per_frame = yolo_detection_with_tracking(
        yolo_model_path, frames,
        det_score_thr=args.det_score_thr,
        device=args.device)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # pose estimation
    print('Running pose estimation...')
    pose_config_path = os.path.join(script_dir, args.pose_config) if not os.path.isabs(args.pose_config) else args.pose_config
    pose_checkpoint_path = os.path.join(script_dir, args.pose_checkpoint) if not os.path.isabs(args.pose_checkpoint) else args.pose_checkpoint
    
    if not os.path.exists(pose_config_path):
        raise FileNotFoundError(f'Pose config not found: {pose_config_path}')
    if not os.path.exists(pose_checkpoint_path):
        raise FileNotFoundError(f'Pose checkpoint not found: {pose_checkpoint_path}')
    
    pose_results, pose_data_samples = pose_inference(
        pose_config_path, pose_checkpoint_path,
        frame_paths, det_results, device=args.device)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # load action recognition model
    print('Loading action recognition model...')
    config = mmengine.Config.fromfile(config_path)
    model = init_recognizer(config, checkpoint_path, device=args.device)
    
    # build tracks from detections and track IDs
    print('Building person tracks...')
    tracks = defaultdict(list)
    
    for frame_idx, (bboxes, track_ids) in enumerate(zip(det_results, track_ids_per_frame)):
        for det_idx, (bbox, tid) in enumerate(zip(bboxes, track_ids)):
            if tid == -1:
                # assign new track ID if not tracked
                tid = len(tracks) + 1000  
            tracks[tid].append((frame_idx, bbox, det_idx))
    
    print(f'Found {len(tracks)} person tracks')
    
    # process each track with sliding windows
    print('Running action recognition on person tracks (sliding windows)...')
    frame_track_predictions = defaultdict(dict)
    
    track_items = list(tracks.items())
    stride = 12  # Overlap between windows
    
    for track_id, track_data in track_iter_progress(track_items):
        if len(track_data) < CLIP_LEN:
            continue
        
        # extract full skeleton sequence for this track
        keypoint_seq, keypoint_score_seq = extract_skeleton_for_track(
            track_data, pose_results, (h, w))
        
        if keypoint_seq is None or len(keypoint_seq) < CLIP_LEN:
            continue
        
        # get frame indices for this track
        track_frame_indices = [t[0] for t in track_data]
        num_frames_in_track = len(track_frame_indices)
        
        window_predictions = []
        window_starts = []
        
        for start_idx in range(0, num_frames_in_track - CLIP_LEN + 1, stride):
            # extract window
            end_idx = start_idx + CLIP_LEN
            window_keypoint_seq = keypoint_seq[start_idx:end_idx]
            window_keypoint_score_seq = keypoint_score_seq[start_idx:end_idx]
            
            # run inference on this window
            pred_class, pred_confidence, dangerous_prob, scores = process_track_with_model(
                model, window_keypoint_seq, window_keypoint_score_seq, (h, w))
            
            if pred_class is not None:
                window_predictions.append({
                    'class': pred_class,
                    'confidence': pred_confidence,
                    'dangerous_prob': dangerous_prob,
                    'scores': scores,
                    'start_frame_idx': start_idx,
                    'center_frame_idx': start_idx + CLIP_LEN // 2,
                })
                window_starts.append(start_idx)
        
        for frame_idx_in_track in range(num_frames_in_track):
            actual_frame_idx = track_frame_indices[frame_idx_in_track]
            
            # find best prediction for this frame 
            best_pred = None
            best_weighted_danger = -1.0
            best_weighted_confidence = -1.0
            
            for win_idx, win_start in enumerate(window_starts):
                win_center = win_start + CLIP_LEN // 2
                distance = abs(frame_idx_in_track - win_center)
                weight = max(0, 1.0 - (distance / (CLIP_LEN / 2)))
                
                if weight <= 0:
                    continue
                
                pred = window_predictions[win_idx]
                weighted_danger = pred['dangerous_prob'] * weight
                weighted_conf = pred['confidence'] * weight
                
                # use prediction with highest confidence
                if (weighted_danger > best_weighted_danger or 
                    (weighted_danger == best_weighted_danger and weighted_conf > best_weighted_confidence)):
                    best_weighted_danger = weighted_danger
                    best_weighted_confidence = weighted_conf
                    best_pred = {
                        'class': pred['class'],
                        'confidence': pred['confidence'] * weight,
                        'dangerous_prob': pred['dangerous_prob'] * weight,
                        'scores': pred['scores']
                    }
            
            if best_pred is not None:
                frame_track_predictions[actual_frame_idx][track_id] = best_pred
    
    print(f'Got predictions for frames with tracked people')
    
    # get FPS for warning persistence calculation
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    cap.release()
    
    min_persistence_frames = max(20, int(fps * 1.0)) 
    print(f'Warning persistence: {min_persistence_frames} frames ({min_persistence_frames/fps:.2f} seconds at {fps} FPS)')
    
    # create output video with warnings and danger scores
    print('Creating output video with warnings and danger scores...')
    output_frames = []
    dangerous_frame_count = 0
    
    active_warnings = {} 
    
    for frame_idx in track_iter_progress(range(num_frames)):
        output_frame = frames[frame_idx].copy()
        
        
        frame_predictions = frame_track_predictions.get(frame_idx, {})
        
        # draw danger scores in lower left corner
        y_start = h - 20
        x_start = 10
        line_height = 25
        
        track_danger_list = []
        for track_id, pred in sorted(frame_predictions.items()):
            dangerous_prob = pred['dangerous_prob']
            track_danger_list.append((track_id, dangerous_prob, pred))
        
        # sort by danger (highest first)
        track_danger_list.sort(key=lambda x: x[1], reverse=True)
        
        # display up to 5 most dangerous tracks
        for i, (track_id, dangerous_prob, pred) in enumerate(track_danger_list[:5]):
            y_pos = y_start - (i * line_height)
            if y_pos < 20:
                break
            
            class_label = CLASS_LABELS[pred['class']]
            danger_text = f"ID{track_id}: {class_label} (danger: {dangerous_prob:.2f})"
            
            # color based on danger level
            if dangerous_prob >= 0.5:
                color = (0, 0, 255) 
            elif dangerous_prob >= 0.3:
                color = (0, 165, 255)  
            else:
                color = (0, 255, 0) 
            
            cv2.putText(output_frame, danger_text, (x_start, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # cseeeck if any person is in danger and trigger warning
        max_danger = max([p['dangerous_prob'] for p in frame_predictions.values()], default=0.0)
        max_danger_track = None
        for track_id, pred in frame_predictions.items():
            if pred['dangerous_prob'] == max_danger and max_danger >= args.danger_threshold:
                max_danger_track = (track_id, pred)
                break
        
       
        if max_danger_track is not None:
            track_id, pred = max_danger_track
            
            for future_frame in range(frame_idx, min(frame_idx + min_persistence_frames, num_frames)):
                if future_frame not in active_warnings:
                    active_warnings[future_frame] = (track_id, pred, frame_idx)
            dangerous_frame_count += 1
        
       
        should_show_warning = False
        warning_track_id = None
        warning_pred = None
        
        if max_danger_track is not None:
            
            should_show_warning = True
            warning_track_id, warning_pred = max_danger_track
        elif frame_idx in active_warnings:
            warning_track_id, warning_pred, trigger_frame = active_warnings[frame_idx]
            if frame_idx - trigger_frame < min_persistence_frames:
                should_show_warning = True
        
        if should_show_warning and warning_pred is not None:
            output_frame = overlay_warning(output_frame, "WARNING!")
            
            # add class label
            class_label = CLASS_LABELS[warning_pred['class']]
            label_text = f"{class_label} (ID{warning_track_id}, danger: {warning_pred['dangerous_prob']:.2f})"
            cv2.putText(output_frame, label_text, (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        output_frames.append(output_frame)
    
    print(f'\nDetected dangerous situations in {dangerous_frame_count}/{num_frames} frames '
          f'({100*dangerous_frame_count/num_frames:.1f}%)')
    
    # save video
    print(f'Saving output video to {out_video_path}...')
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
    
    # write video     
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
    if not out_writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
    
    print(f'Writing {len(output_frames)} frames at {fps} FPS...')
    for frame in track_iter_progress(output_frames):
        out_writer.write(frame)
    
    out_writer.release()
    
    print(f'Done! Output saved to: {out_video_path}')
    tmp_dir.cleanup()


if __name__ == '__main__':
    main()
