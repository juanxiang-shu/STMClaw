import json
import logging
import os
import pickle
import queue
import random
import shutil
import sys
import threading
import time
from math import sqrt
from pathlib import Path
from multiprocessing import Process, Queue
from scipy.linalg import lstsq
from collections import deque

import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FuncAnimation
from core import NanonisController
import cv2
import keyboard
import matplotlib.pyplot as plt
import numpy as np
import torch


class SimpleReplayMemory:
    def save(self, path, name, obj):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, name), 'w', encoding='utf-8') as f:
            json.dump(obj, f)

# DQN subsystem removed — replaced by simple random fallback for tip actions.
from modules.evaluation import evaluate_image_quality
from modules.conditioning_agent import ConditioningAgentMixin, center_to_square, increase_radius, pix_to_nanocoordinate, linear_whole, linear_normalize_whole, images_equalization, Hex_to_BGR
from modules.planning_agent import NavigatorPathAgentMixin
from mol_segment.detect import Segmented_image
from tasks.LineScanchecker import *

def get_latest_checkpoint(parent_folder, checkpoint_name="checkpoint.json"):
    parent_folder = os.path.abspath(parent_folder)
    subfolders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    if not subfolders:
        return None
    latest_subfolder = max(subfolders, key=os.path.getctime)
    return os.path.join(latest_subfolder, checkpoint_name)


def time_trajectory_list(parent_path, file_extension='.json'):
    npy_files = []
    for root, dirs, files in os.walk(parent_path):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                creation_time = os.path.getctime(file_path)
                npy_files.append((file_path, creation_time))

    npy_files_sorted = sorted(npy_files, key=lambda x: x[1])
    return [file[0] for file in npy_files_sorted]


def tip_in_boundary(inter_closest, plane_size, real_scan_factor):
    return (
        inter_closest[0] <= plane_size / 2 * (1 + real_scan_factor)
        and inter_closest[0] >= plane_size / 2 * (1 - real_scan_factor)
        and inter_closest[1] <= plane_size / 2 * (1 + real_scan_factor)
        and inter_closest[1] >= plane_size / 2 * (1 - real_scan_factor)
    )


class ConsoleTee:
    """Write stream content to both console and a local file."""
    def __init__(self, stream, file_path):
        self.stream = stream
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.file = open(file_path, 'a', encoding='utf-8', buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass

class  Mustard_AI_Nanonis(ConditioningAgentMixin, NavigatorPathAgentMixin, NanonisController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))    # record the start time of the scan
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scan_start_time = None
        self.total_scan_time = 0.0
        self.per_scan_time = 0.0
        self.ScandataQueue_1 = Queue(5)
        self.tipdataQueue = Queue(5)
        self.Tipshaper_signal = 0
        self.iftipshaper = 0
        self.circle_list = []                       # the list to save the circle which importantly to the tip path
        self.circle_list_save = []                  # the list to save the circle as npy file
        self.nanocoodinate_list = []                # the list to save the coodinate which send to nanonis directly
        self.visual_circle_buffer_list = []

        self.good_image_count = 0
        self.count_choose = 100
        self.frame_move_signal = 0
        self.line_scan_signal = 0
        self.scale_image_for = []
        self.scale_image_back = []
        self.scale_signal = 0
        self.tippath_stop_event = threading.Event()
        self.batch_stop_event = threading.Event()
        self.savetip_stop_event = threading.Event()
        self.linescan_stop_event = threading.Event()
        self.mode = None

        self.trajectory_buffer_size = 5             # the size of the trajectory buffer
        self.trajectory_state_list = deque([], maxlen=self.trajectory_buffer_size)
        self.trajectory_action_list = deque([], maxlen=self.trajectory_buffer_size)
        self.trajectory_reward_list = deque([], maxlen=self.trajectory_buffer_size)
        self.trajectory_next_state_list = deque([], maxlen=self.trajectory_buffer_size)
        
        self.line_scan_change_times = 0             # initialize the line scan change times to 0
        self.episode_count = 0                      # initialize the episode count to 0
        self.AdjustTip_flag = 0                     # 0: the tip is not adjusted, 1: the tip is adjusted just now

        self.Scan_edge = '30n'                    # set the initial scan square edge length                           30pix  ==>>  30nm   80
        self.scan_square_Buffer_pix = 208            # scan pix
        self.plane_edge = "1u"                       # plane size repersent the area of the Scanable surface  2um*2um    2000pix  ==>>  2um

        self.Z_fixed_scan_time = 10                   # if the scan is after the Z fixed, how many seconds will the scan cost
        self.without_Z_fixed_scan_time = 10           # if the scan is without the Z fixed, how many seconds will the scan cost
        self.linescan_max_min_threshold = '10000p'      # the max and min threshold , if the line scan data is out of the threshold
        self.scan_max_min_threshold = '10n'      # the max and min threshold , if the line scan data is out of the threshold
        self.len_threshold_list = 10
        self.threshold_list = deque([], maxlen=self.len_threshold_list)     # the list to save the threshold of the line scan data, if threshold_list is full with 1, the scan will be skiped
        self.skip_list = deque([], maxlen=10)         # the list to save the skip flag, if the skip_list is full with 1, gave the tip a aggressive tip shaper
        self.skip_flag = 0                            # 0: the scan is not skiped, 1: the scan is skiped
        self.aggressive_tip_flag = 0                  # 0: the tip is not aggressive, 1: the tip is aggressive

        self.real_scan_factor = 0.8                  # the real scan area is 80% of the scan square edge length

        self.tip_move_mode = 0                       # 0: continuous_random_move_mode, 1: para_or_vert_move_mode

        # Planning Agent state. Fixed spiral/snake strategies are deprecated.
        self.navigator_mode = 'custom'              # supported: 'custom' (legacy spiral/snake mapping available)
        self.navigator_step_pix = None              # path spacing in pixel; None means auto use scan_square_edge
        self.navigator_snake_points = []            # cached snake-path waypoints
        self.navigator_snake_index = 0
        self.navigator_custom_points = []           # user-defined/custom waypoints in pixel
        self.navigator_custom_index = 0
        self.navigator_user_instruction = ''
        self.navigator_last_plan = {}
        self.navigator_start_points = []          # optional first two points from Starting Planning Agent

        self.line_scan_activate = 1                  # 0: line scan is not activate, 1: line scan is activate

        self.equalization_alpha = 0.3                # the alpha of the equalization

        self.scan_qulity_threshold = 0.6             # the threshold of the scan qulity, higher means more strict for the scan qulity

        self.scan_qulity = 1                         # 1: the scan qulity is good, 0: the scan qulity is bad        

        self.nanonis_mode = 'real'                   # the nanonis mode, 'demo' or 'real'

        # action selection uses a simple random policy now (DQN removed)
        self.action_space = 6

        # Conditioning Agent state.
        self.pending_conditioning_action = None     # action executed at previous bad image, waiting for effectiveness check
        self.consecutive_repair_failures = 0        # continuous ineffective repair count
        self.conditioning_tipshaper_streak = 0      # continuous TipShaper interventions from Conditioning Agent
        self.navigator_skip_budget = 0              # extra points to skip in next navigator step(s)
        self.total_step_count = 0
        self.scan_was_skipped = False                # True means current frame used synthetic skip data
        self.last_evaluation_info = {}
        self.conditioning_learned_ranges = {}
        self.conditioning_learning_time = None
        self.conditioning_history_file = None
        self.scan_step_records = []
        self.pending_action_step_index = None
        self.scan_step_records_json_path = None
        self.scan_step_records_npy_path = None
        self.evaluation_history_jsonl_path = None

        # Assignment Agent state.
        self.molecule_match_enabled = str(os.getenv('MOLECULE_MATCH_ENABLED', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
        self.molecule_match_structure_image = os.getenv('DEFAULT_STRUCTURE_IMAGE_PATH', '').strip()
        self.molecule_match_surface = os.getenv('DEFAULT_SURFACE', 'Au(111)').strip() or 'Au(111)'
        self.molecule_match_model = os.getenv('GEMINI_MODEL_MATCH', os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')).strip()
        molecule_match_interval_raw = str(os.getenv('MOLECULE_MATCH_INTERVAL', '50')).strip()
        try:
            self.molecule_match_interval = max(1, int(molecule_match_interval_raw))
        except Exception:
            self.molecule_match_interval = 50
        self.molecule_match_last_run_step = None
        self.molecule_match_result_dir = None
        self.last_molecule_match_info = {}

        if self.nanonis_mode == 'demo':
            self.signal_channel_list = [0, 30]               # the mode channel list, 0 is Current, 30 is Z_m
        elif self.nanonis_mode == 'real':
            self.signal_channel_list = [0, 8, 14]            # the mode channel list, 0 is Current, 8 is Bias, 14 is Z_m
        
        self.agent_upgrate = 1                              # 0:the DQN model will not be upgrated, 1: the DQN model will be upgrated    

        self.quality_model_path = os.getenv('GEMINI_API_KEY', '')

        # Unified Gemini request config for Planning/Starting/Path agents.
        # Keep source-of-truth in main class __init__, mixins should read from these fields.
        self.gemini_model_navigator = os.getenv('GEMINI_MODEL_NAVIGATOR', os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'))
        self.gemini_model_path = os.getenv('GEMINI_MODEL_PATH', os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'))
        self.gemini_model_codegen = os.getenv('GEMINI_MODEL_CODEGEN', os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'))
        self.gemini_timeout_seconds = int(os.getenv('GEMINI_TIMEOUT', '90'))
        self.gemini_request_retries = int(os.getenv('GEMINI_RETRIES', '2'))
        self.gemini_retry_backoff_seconds = float(os.getenv('GEMINI_RETRY_BACKOFF', '1.0'))

        self.segment_model_path = './mol_segment/unet_modelsingle.pth'  # the path of the segment model weights

        # DQN model paths removed

        self.main_data_save_path = './log'

        self.log_path =  self.main_data_save_path + '/' + self.start_time

        # Redirect console output to both terminal and local file.
        self.runtime_log_file = os.path.join(self.log_path, 'runtime_console.log')
        self._enable_console_file_logging()
        self._init_scan_step_recorder()

        self.memory_path = os.path.join(self.main_data_save_path, 'memory', self.start_time)
        self.memory = SimpleReplayMemory()

        self.Scan_edge_SI = self.convert(self.Scan_edge)                        # the scan edge in SI unit   Scan_edge_SI = 30 * 1e-9

        self.plane_size = int(self.convert(self.plane_edge)*10**9)            # the plane size in pix  plane_size = 2000
        self.scan_square_edge = int(self.convert(self.Scan_edge)*10**9)            # the scan square edge in pix  scan_square_edge = 30

        self.tip_path_img = np.ones((self.plane_size, self.plane_size, 3), np.uint8) * 255  # the tip path image

        self.inter_closest = (round(self.plane_size/2), round(self.plane_size/2))                          # initialize the inter_closest
        # R_init = round(scan_square_edge*(math.sqrt(2))*1.5)
        self.R_init = self.scan_square_edge -1                                                        # initialize the Radius of tip step
        self.R_max = self.R_init*3
        self.R_step = int(0.5*self.R_init)
        self.R = self.R_init

        # Initialize Planning Agent with custom path generation.
        self.navigator_step_pix = self.scan_square_edge
        self.navigator_reset()


        # initialize the other parameters that appear in the function
        self.Scan_data = {}                                                                         # the dictionary to save the scan data
        self.image_for = None   # 2D nparray the image of the scan data, have been nomalized and linear background
        self.image_back = None
        self.equalization_for = None    # the equalization image of the image_for and image_back
        self.equalization_back = None
        self.image_for_tensor = None    # the tensor of the image, 4 dimension, [1, 1, 256, 256]
        self.image_back_tensor = None   
        self.image_save_time = None         # when the image is saved in log
        self.npy_data_save_path     =  self.log_path + '/' + 'npy'                                 # self.log_path = './log/' + self.start_time
        self.image_data_save_path   =  self.log_path + '/' + 'image'
        self.segmented_image_path = None    # the path of the segmented image saving
        self.nemo_nanocoodinate = None      # the nanocoodinate of the nemo point, the format is SI unit
        self.coverage = None                # the moleculer coverage of the image
        self.line_start_time = None
        self.episode_start_time = None    # the start time of the episode
        
        # initialize the queue, the Queue is used to communicate between different threads
        self.lineScanQueue = Queue(5)    # lineScan_data_producer → lineScan_data_consumer
        self.lineScanQueue_back = Queue(5)    # lineScan_data_consumer → lineScan_data_producer
        self.ScandataQueue = Queue(5)    # batch_scan_producer → batch_scan_consumer

        self.tippathvisualQueue = Queue(5)   # main program  → tip_path_visualization

        self.scanqulityqueue = Queue() # TipShaperthreading → main program

        self.PyQtimformationQueue = Queue(5) # PyQt_GUI → main program
        self.PyQtPulseQueue = Queue(5)       # PyQt_GUI → HandPulsethreading
        self.PyQtTipShaperQueue = Queue(5)   # PyQt_GUI → TipShaperthreading

    def _init_scan_step_recorder(self):
        os.makedirs(self.log_path, exist_ok=True)
        self.scan_step_records = []
        self.pending_action_step_index = None
        self.scan_step_records_json_path = os.path.join(self.log_path, 'scan_step_records.json')
        self.scan_step_records_npy_path = os.path.join(self.log_path, 'scan_step_records.npy')
        self.evaluation_history_jsonl_path = os.path.join(self.log_path, 'evaluation_history.jsonl')
        with open(self.evaluation_history_jsonl_path, 'w', encoding='utf-8'):
            pass
        self._persist_scan_step_records()

    def _append_evaluation_history(self, payload):
        if not self.evaluation_history_jsonl_path:
            return
        os.makedirs(self.log_path, exist_ok=True)
        with open(self.evaluation_history_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def _persist_scan_step_records(self):
        if not self.scan_step_records_json_path or not self.scan_step_records_npy_path:
            return
        os.makedirs(self.log_path, exist_ok=True)
        with open(self.scan_step_records_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.scan_step_records, f, ensure_ascii=False, indent=2)
        np.save(self.scan_step_records_npy_path, np.array(self.scan_step_records, dtype=object), allow_pickle=True)

    def _resolve_molecule_structure_image_path(self):
        candidate = str(self.molecule_match_structure_image or '').strip()
        if candidate and os.path.isfile(candidate):
            return candidate

        molecules_dir = os.path.join('.', 'molecules')
        if not os.path.isdir(molecules_dir):
            return ''

        supported_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
        image_names = [
            name for name in sorted(os.listdir(molecules_dir))
            if os.path.isfile(os.path.join(molecules_dir, name)) and name.lower().endswith(supported_ext)
        ]
        if not image_names:
            return ''

        resolved = os.path.join(molecules_dir, image_names[0])
        self.molecule_match_structure_image = resolved
        return resolved

    def _save_current_stm_image_for_molecule_match(self):
        if self.image_for is None:
            raise ValueError('image_for is empty, cannot run Assignment Agent.')

        out_dir = os.path.join(self.log_path, 'molecule_match')
        os.makedirs(out_dir, exist_ok=True)
        self.molecule_match_result_dir = out_dir

        stm_path = os.path.join(out_dir, f'stm_step_{int(self.total_step_count):05d}.png')
        image = np.array(self.image_for)
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            max_val = float(np.max(image)) if image.size > 0 else 0.0
            if max_val <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

        ok = cv2.imwrite(stm_path, image)
        if not ok:
            raise RuntimeError(f'Failed to write STM image for molecule matching: {stm_path}')
        return stm_path

    def run_molecule_match_agent(self):
        result_stub = {
            'judgement': 'not_run',
            'overall_match': None,
            'confidence': 0.0,
            'reason': '',
            'source_agent': 'assignment',
            'run_interval': int(self.molecule_match_interval),
            'run_step': None,
        }

        if not self.molecule_match_enabled:
            result_stub['reason'] = 'disabled_by_config'
            self.last_molecule_match_info = result_stub
            return result_stub

        if self.scan_was_skipped:
            result_stub['reason'] = 'skip_frame'
            self.last_molecule_match_info = result_stub
            return result_stub

        current_step = int(self.total_step_count)
        if self.molecule_match_interval > 1 and (current_step % self.molecule_match_interval != 0):
            result_stub['reason'] = f'throttled_every_{self.molecule_match_interval}_frames'
            result_stub['next_run_step'] = int(
                current_step + (self.molecule_match_interval - (current_step % self.molecule_match_interval))
            )
            self.last_molecule_match_info = result_stub
            return result_stub

        structure_path = self._resolve_molecule_structure_image_path()
        if not structure_path:
            result_stub['reason'] = 'no_structure_image_found_in_molecules_or_env'
            self.last_molecule_match_info = result_stub
            print('[Assignment Agent] no structure image found, skip this step.')
            return result_stub

        try:
            stm_path = self._save_current_stm_image_for_molecule_match()
        except Exception as exc:
            result_stub['reason'] = f'prepare_stm_image_failed: {exc}'
            self.last_molecule_match_info = result_stub
            print(f'[Assignment Agent] STM image export failed: {exc}')
            return result_stub

        try:
            from modules import assignment_agent as assignment_module
        except Exception as exc:
            result_stub['reason'] = f'import_failed: {exc}'
            self.last_molecule_match_info = result_stub
            print(f'[Assignment Agent] import failed: {exc}')
            return result_stub

        try:
            raw_result = assignment_module.run_match_pipeline(
                structure_path=Path(structure_path),
                stm_path=Path(stm_path),
                surface=self.molecule_match_surface,
                model_name=self.molecule_match_model,
                pdf_cache_dir=getattr(molecule_match_module, 'DEFAULT_PDF_CACHE_DIR', './data/literature_pdf_cache'),
            )
        except Exception as exc:
            result_stub['reason'] = f'pipeline_failed: {exc}'
            self.last_molecule_match_info = result_stub
            print(f'[Assignment Agent] pipeline failed: {exc}')
            return result_stub

        judgement = raw_result.get('match_judgement', {}) if isinstance(raw_result, dict) else {}
        confidence = judgement.get('confidence', 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        overall_match = judgement.get('overall_match', None)
        if not isinstance(overall_match, bool):
            overall_match = None

        match_info = {
            'judgement': str(judgement.get('judgement', 'uncertain')),
            'overall_match': overall_match,
            'confidence': confidence,
            'reason': str(judgement.get('evidence_summary', '')),
            'source_agent': 'assignment',
            'run_interval': int(self.molecule_match_interval),
            'run_step': current_step,
        }
        self.molecule_match_last_run_step = current_step

        try:
            if self.molecule_match_result_dir:
                result_path = os.path.join(self.molecule_match_result_dir, f'match_step_{int(self.total_step_count):05d}.json')
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(raw_result, f, ensure_ascii=False, indent=2)
                match_info['result_path'] = result_path
        except Exception as exc:
            print(f'[Assignment Agent] failed to persist raw result: {exc}')

        self.last_molecule_match_info = match_info
        print('[Assignment Agent] structured judgement:')
        print(json.dumps(match_info, ensure_ascii=False, indent=2))
        return match_info

    def record_scan_step(self, scan_qulity):
        eval_info = self.last_evaluation_info if isinstance(self.last_evaluation_info, dict) else {}
        molecule_match_info = self.last_molecule_match_info if isinstance(self.last_molecule_match_info, dict) else {}
        label = eval_info.get('label', 'good' if scan_qulity == 1 else 'bad')
        timestamp = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
        defect_tags = list(eval_info.get('defect_tags', [])) if isinstance(eval_info.get('defect_tags', []), list) else []
        match_confidence = molecule_match_info.get('confidence', 0.0)
        try:
            match_confidence = float(match_confidence)
        except Exception:
            match_confidence = 0.0
        evaluation_info = {
            'label': str(label),
            'reason': str(eval_info.get('reason', '')),
            'severity': str(eval_info.get('severity', 'unknown')),
            'defect_tags': defect_tags,
            'good_probability': float(eval_info.get('good_probability', 0.0)),
            'confidence': float(eval_info.get('confidence', 0.0)),
        }
        molecule_match_record = {
            'judgement': str(molecule_match_info.get('judgement', 'not_run')),
            'overall_match': molecule_match_info.get('overall_match', None),
            'confidence': match_confidence,
            'reason': str(molecule_match_info.get('reason', '')),
        }
        record = {
            'step_index': int(self.total_step_count),
            'timestamp': timestamp,
            'scan_center_pix': [float(self.inter_closest[0]), float(self.inter_closest[1])] if self.inter_closest is not None else None,
            'scan_center_nano': [float(self.nanocoodinate[0]), float(self.nanocoodinate[1])] if self.nanocoodinate is not None else None,
            'scan_was_skipped': bool(self.scan_was_skipped),
            'scan_quality': int(scan_qulity),
            'molecule_match': molecule_match_record,
            'evaluation': evaluation_info,
            'operation': {
                'executed': False,
                'type': None,
                'value': None,
                'params': {},
            },
            'post_operation_evaluation': None,
        }
        self.scan_step_records.append(record)
        self._persist_scan_step_records()

        self._append_evaluation_history(
            {
                'step_index': int(self.total_step_count),
                'timestamp': timestamp,
                'scan_quality': int(scan_qulity),
                'scan_was_skipped': bool(self.scan_was_skipped),
                'scan_center_pix': record['scan_center_pix'],
                'scan_center_nano': record['scan_center_nano'],
                'molecule_match': molecule_match_record,
                'evaluation': evaluation_info,
            }
        )
        return len(self.scan_step_records) - 1

    def record_conditioning_action_for_step(self, step_record_index, action_plan):
        if step_record_index is None:
            return
        if step_record_index < 0 or step_record_index >= len(self.scan_step_records):
            return
        if not isinstance(action_plan, dict):
            return

        action_type = str(action_plan.get('action', ''))
        action_value = None
        if action_type == 'Pulse':
            action_value = float(action_plan.get('voltage', 0.0))
        elif action_type == 'TipShaper':
            action_value = str(action_plan.get('TipLift', ''))

        self.scan_step_records[step_record_index]['operation'] = {
            'executed': True,
            'type': action_type,
            'value': action_value,
            'params': dict(action_plan),
        }
        self.pending_action_step_index = step_record_index
        self._persist_scan_step_records()

    def record_post_action_evaluation(self, current_scan_qulity):
        if self.pending_action_step_index is None:
            return
        idx = self.pending_action_step_index
        if idx < 0 or idx >= len(self.scan_step_records):
            self.pending_action_step_index = None
            return

        eval_info = self.last_evaluation_info if isinstance(self.last_evaluation_info, dict) else {}
        label = eval_info.get('label', 'good' if current_scan_qulity == 1 else 'bad')
        self.scan_step_records[idx]['post_operation_evaluation'] = {
            'evaluated_at_step_index': int(self.total_step_count),
            'timestamp': time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())),
            'scan_quality': int(current_scan_qulity),
            'label': str(label),
            'reason': str(eval_info.get('reason', '')),
            'severity': str(eval_info.get('severity', 'unknown')),
            'good_probability': float(eval_info.get('good_probability', 0.0)),
            'confidence': float(eval_info.get('confidence', 0.0)),
        }
        self.pending_action_step_index = None
        self._persist_scan_step_records()

    def _enable_console_file_logging(self):
        os.makedirs(self.log_path, exist_ok=True)
        if not hasattr(self, '_stdout_original'):
            self._stdout_original = sys.stdout
        if not hasattr(self, '_stderr_original'):
            self._stderr_original = sys.stderr

        if not isinstance(sys.stdout, ConsoleTee):
            sys.stdout = ConsoleTee(self._stdout_original, self.runtime_log_file)
        if not isinstance(sys.stderr, ConsoleTee):
            sys.stderr = ConsoleTee(self._stderr_original, self.runtime_log_file)

        print(f"[Logger] Console output is being saved to: {self.runtime_log_file}")

    def is_serializable(self, value):
        """ Attempt to serialize the value, return True if serializable, False otherwise. """
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False
        
    def save_checkpoint(self):
        """ Save the current state of serializable instance attributes to a file. """
        serializable_attrs = {k: v for k, v in self.__dict__.items() if self.is_serializable(v)}
        filename = os.path.join(self.log_path,'checkpoint.json')
        with open(filename, 'w') as file:
            json.dump(serializable_attrs, file)
        
    def load_checkpoint(self):
        """Loads class attributes from a json file and updates the instance."""
        checkpoint_json = get_latest_checkpoint(self.main_data_save_path)
        with open(checkpoint_json, 'r') as file:
            data = json.load(file)
            self.__dict__.update(data)
        checkpoint_tip_img = get_latest_checkpoint(self.main_data_save_path,checkpoint_name="tip_path.jpg")
        self.tip_path_img = cv2.imread(checkpoint_tip_img, cv2.IMREAD_COLOR)

    # activate all the threads of monitor   
    def monitor_thread_activate(self):
        Safe_Tip_thread = threading.Thread(target=self.SafeTipthreading, args=('5n', 100),daemon=True)                                          # the SafeTipthrehold is 5n, the safe_withdraw_step is 100
        tip_visualization_thread = threading.Thread(target=self.tip_path_visualization, daemon=True)                                            # the tip path visualization thread
        batch_scan_consumer_thread = threading.Thread(target=self.batch_scan_consumer, daemon=True)                                             # the batch scan consumer thread
        Hand_Pulse_thread = threading.Thread(target=self.HandPulsethreading, args=(self.PyQtPulseQueue,), daemon=True)                          # the HandPulse thread
        TipShaper_thread = threading.Thread(target=self.Tipshaperthreading, args=(self.PyQtTipShaperQueue,self.scanqulityqueue), daemon=True)   # the TipShaper thread
        instruct_thread = threading.Thread(target=self.instructthreading, args=(self.PyQtimformationQueue,), daemon=True)                       # the instruct thread

        # if tip_visualization_thread.is_alive():
        #     Safe_Tip_thread.join()
        #     tip_visualization_thread.join()
        #     batch_scan_consumer_thread.join()
        #     self.tippath_stop_event = 0
        #     self.batch_stop_event = 0
        #     self.savetip_stop_event = 0
        #     Safe_Tip_thread.start()
        #     tip_visualization_thread.start()
        #     batch_scan_consumer_thread.start()
        # else:
        #     Safe_Tip_thread.start()
        #     tip_visualization_thread.start()
        #     batch_scan_consumer_thread.start()
        Safe_Tip_thread.start()
        tip_visualization_thread.start()
        batch_scan_consumer_thread.start()
        Hand_Pulse_thread.start()
        TipShaper_thread.start()
        instruct_thread.start()

    def SafeTipthreading(self, SafeTipthrehold = '5n', safe_withdraw_step = 100):
        print('The tipsafe monitoring activated')
        # if the type of SafeTipthrehold is string, convert it to float
        if type(SafeTipthrehold) == str:
            SafeTipthrehold = self.convert(SafeTipthrehold)

        current_list = []
        # while True:
        while not self.savetip_stop_event.is_set():
            time.sleep(0.5)
            try:
                current = self.CurrentGet()
            except:
                current = 0
            current_list.append(current)
            # print(current_list)
            if len(current_list) >= 8:
                current_list.pop(0)
                # if all the current absoulte value is bigger than SafeTipthrehold, stop the scan and withdraw the tip than move Motor Z- 50? steps
                if all(abs(current) > SafeTipthrehold for current in current_list):
                    #raise ValueError('The tunneling current is too large, the tip protection is activated, and the scan stops.')
                    print('The tunneling current is too large, the tip protection is activated.')
                    self.ScanStop()
                    self.Withdraw()
                    self.MotorMoveSet('Z-', safe_withdraw_step)
                    current_list = []
                    print('The tip is withdrawed')
                    break
    
    def HandPulsethreading(self, PyQtPulseQueue):
        print("Pulsethreading activated")
        number_of_HandPulse = 0
        while True:
            if not PyQtPulseQueue.empty():
                item = PyQtPulseQueue.get()  # Retrieve the next item from the queue.
                # Check for a special character and perform an action.
                if item[0] == "Pulse":
                    Pulse_start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))                                    # record the start time of the Pulse
                    self.BiasPulse(item[1], width=0.05)
                    # check the save path exist or not
                    if not os.path.exists(self.log_path):
                        os.makedirs(self.log_path)
                    np.save(self.log_path + '/' + Pulse_start_time  + '_Pulse_' + str(number_of_HandPulse).zfill(5) +'.npy', [item[0],item[1] ,number_of_HandPulse], allow_pickle = True)
                    number_of_HandPulse += 1
                    print(f"Pulse {item[1]}V")
    
    def Tipshaperthreading(self, PyQtTipShaperQueue,scanqulityqueue):
        print("TipShaperthreading activated")
        number_of_Tipshaper = 0
        while True:
            item = PyQtTipShaperQueue.get()  # Retrieve the next item from the queue.
            # Check for a special character and perform an action.
            # scanqulityqueue.put(0)
            if item[0] == "TipShaper":
                scanqulityqueue.put(0)
                try:
                    Tipshaper_start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))                                    # record the start time of the Pulse
    
                    self.TipShaper(TipLift = item[1])
    
                    self.ScanStop()  # stop the scan so that the batch scan can be interrupted
                    # check the save path exist or not
                    if not os.path.exists(self.log_path):
                        os.makedirs(self.log_path)
                    np.save(self.log_path + '/' + Tipshaper_start_time + '_TipShaper_' + str(number_of_Tipshaper).zfill(5) +'.npy', [item[0],item[1] ,number_of_Tipshaper], allow_pickle = True)
                    number_of_Tipshaper += 1
                    print(f"TipShaper {item[1]}")
                except:
                    print('The TipShaper is not successful')
                print("install the scan , move to the next point...")
                # nanonis.ScanStart()

    def instructthreading(self, PyQtimformationQueue):
        """
        Thread function to monitor a queue for special characters.
        :param q: The queue to monitor.
        :param identifier: A string to identify which queue (and hence which button) this thread is monitoring.
        """
        while True:
            item = PyQtimformationQueue.get()  # Retrieve the next item from the queue.
            # Check for a special character and perform an action.
            if item == "Home":
                self.ScanPause()
                self.ScanFrameSet(0, 0, "100n", "100n", angle=0)
                self.TipXYSet(0, 0)
                print(f"set Tip and Scan frame to center")
                PyQtimformationQueue.put('stop whole scan')
                break
            if item == "skip to next":
                self.ScanStop()
                print(f"skip to next")
                time.sleep(0.5)
                self.ScanStart()

    def batch_scan_producer(self,  Scan_posion = (0.0 ,0.0), Scan_edge = "30n", Scan_pix = 304 ,  angle = 0, ):
        if self.skip_flag == 1:
            print('creating skip data...')
            self.scan_was_skipped = True
            Scan_data_for = {'data': np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                'row': Scan_pix,
                'col': Scan_pix,
                'scan_direction': 0,
                'channel_name': 'Z (m)',
                }
            Scan_data_back = {'data': np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                'row': Scan_pix,
                'col': Scan_pix,
                'scan_direction': 0,
                'channel_name': 'Z (m)',
                }
            # Keep skip behavior one-shot to avoid endless synthetic scans.
            self.skip_flag = 0
        else:
            print('Scaning the image...')
            self.scan_was_skipped = False
            try:
                self.ScanBufferSet(Scan_pix, Scan_pix, self.signal_channel_list) # 14 is the index of the Z_m channel in real scan mode , in demo mode, the index of the Z_m channel is 30
                self.ScanPropsSet(Continuous_scan = 2 , Bouncy_scan = 2,  Autosave = 1, Series_name = ' ', Comment = 'inter_closest')# close continue_scan & bouncy_scan, but save all data
                self.ScanFrameSet(Scan_posion[0], Scan_posion[1], Scan_edge, Scan_edge, angle=angle)

                self.ScanStart()
                t = (self.ScanSpeedGet()['Forward time per line'] + self.ScanSpeedGet()['Backward time per line']) * self.ScanBufferGet()['Lines']
                self.calculate_total_scan_time(t)
                self.ScanStop()

                try:   # some times the scan data is not successful because of the TCP/IP communication problem
                    Scan_data_for = self.ScanFrameData(self.signal_channel_list[-1], data_dir=1)
                except: #if is not successful, set fake data
                    Scan_data_for = {'data': np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                                    'row': Scan_pix,
                                    'col': Scan_pix,
                                    'scan_direction': 0,
                                    'channel_name': 'Z (m)',
                                    }
                time.sleep(1)
                try:
                    Scan_data_back = self.ScanFrameData(self.signal_channel_list[-1], data_dir=0)
                except:
                    Scan_data_back = {'data': np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                                    'row': Scan_pix,
                                    'col': Scan_pix,
                                    'scan_direction': 0,
                                    'channel_name': 'Z (m)',
                                    }
                if np.isnan(Scan_data_for['data'][0][0]) or np.isnan(Scan_data_for['data'][-1][-1]) or np.isnan(Scan_data_back['data'][0][0]) or np.isnan(Scan_data_back['data'][-1][-1]):
                    Scan_data_for['data'][np.isnan(Scan_data_for['data'])] = 0
                    Scan_data_back['data'][np.isnan(Scan_data_back['data'])] = 0
            except Exception as e:
                print(f'Scaning failed, fallback to fake data: {e}')
                Scan_data_for = {'data': np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                                'row': Scan_pix,
                                'col': Scan_pix,
                                'scan_direction': 0,
                                'channel_name': 'Z (m)',
                                }
                Scan_data_back = {'data': np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                                'row': Scan_pix,
                                'col': Scan_pix,
                                'scan_direction': 0,
                                'channel_name': 'Z (m)',
                                }

        self.image_for = linear_normalize_whole(Scan_data_for['data'])                     # image_for and image_back are 2D nparray
        self.image_back = linear_normalize_whole(Scan_data_back['data'])

        self.image_for = images_equalization(self.image_for, alpha=self.equalization_alpha)
        self.image_back = images_equalization(self.image_back, alpha=self.equalization_alpha)

        self.image_for_tensor = torch.tensor(self.image_for, dtype=torch.float32, device=self.device).unsqueeze(0)  # for instance tensor.shape are [1, 1, 256, 256]  which is for DQN
        self.image_back_tensor = torch.tensor(self.image_back, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.Scan_data = {'Scan_data_for':Scan_data_for, 'Scan_data_back':Scan_data_back}

        if self.ScandataQueue.full():
            try:
                self.ScandataQueue.get_nowait()
                print('[Queue] ScandataQueue full, drop oldest item')
            except queue.Empty:
                pass
        self.ScandataQueue.put_nowait(self.Scan_data)
        time.sleep(0.5)
        # self.ScandataQueue_1.put(self.Scan_data)
        # put the batch scan data into the queue, blocking if Queue is full
        print('Scaning complete! \n ready to save...')
        return self.Scan_data

    def lineScan_data_producer(self,angle=0, scan_continue_time = 90):
            
        end_time = time.time() + scan_continue_time

        self.ScanPropsSet(Continuous_scan = 1, Bouncy_scan = 1,  Autosave = 3, Series_name = ' ', Comment = 'LineScan')
        self.lineScanmode(3, angle=angle)
        Z_m_index = self.signal_channel_list[-1]

        self.ScanStart()
        self.WaitEndOfScan()
        while True:
            lineScandata = self.lineScanGet(Z_m_index)
            if self.lineScanQueue.full():
                self.lineScanQueue.get()
            if time.time() > end_time or self.skip_flag or self.linescan_stop_event.is_set(): # if the time is over or the scan is skiped, break the loop
                self.lineScanQueue.put('end')
                
                if self.aggressive_tip_flag: # aggressive_tip_flag = 1 means the scan is skiped 5 times continuously, so that might be the super terable tip!
                    time.sleep(1)
                    self.TipShaper(TipLift= '-6n')
                    time.sleep(1)
                    self.BiasPulse(-6, width=0.05)
                    time.sleep(1)
                    self.aggressive_tip_flag = 0    # reset the aggressive_tip_flag
                break

            
            self.lineScanQueue.put(lineScandata)

    def batch_scan_consumer(self):
        
        self.npy_data_save_path     =  self.log_path + '/' + 'npy'                                 # self.log_path = './log/' + self.start_time
        self.image_data_save_path   =  self.log_path + '/' + 'image'
        self.equalization_save_path = self.log_path + '/' + 'equalize'        
        
        if not os.path.exists(self.equalization_save_path):                                 # check the save path exist or not
            os.makedirs(self.equalization_save_path)

        if not os.path.exists(self.npy_data_save_path):
            os.makedirs(self.npy_data_save_path)

        if not os.path.exists(self.image_data_save_path):
            os.makedirs(self.image_data_save_path)

        # while True:
        while not self.batch_stop_event.is_set():
            # time.sleep(1)
            if not self.ScandataQueue.empty():
                Scan_data = self.ScandataQueue.get()
                Scan_data_for = Scan_data['Scan_data_for']['data']
                Scan_data_back = Scan_data['Scan_data_back']['data']
                # preprocess the scan data, and save the scan data and image
                image_for = linear_normalize_whole(Scan_data_for)
                image_back = linear_normalize_whole(Scan_data_back)

                equalization_for = images_equalization(image_for, alpha=self.equalization_alpha)
                equalization_back = images_equalization(image_back, alpha=self.equalization_alpha)      # equalize the image

                self.image_save_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))

                npy_data_save_path_for      = self.npy_data_save_path + '/' + 'Scan_data_for_'+ self.image_save_time +'.npy'
                npy_data_save_path_back     = self.npy_data_save_path + '/' + 'Scan_data_back_'+ self.image_save_time +'.npy'
                image_data_save_path_for    = self.image_data_save_path + '/' + 'Scan_data_for'+ self.image_save_time +'.png'
                image_data_save_path_back   = self.image_data_save_path + '/' + 'Scan_data_back'+ self.image_save_time +'.png'
                equalization_save_path_for  = self.equalization_save_path + '/' + 'Scan_data_for'+ self.image_save_time +'.png'
                equalization_save_path_back = self.equalization_save_path + '/' + 'Scan_data_back'+ self.image_save_time +'.png'

                cv2.imwrite(equalization_save_path_for, equalization_for)
                cv2.imwrite(equalization_save_path_back, equalization_back)     # save the equalization image                
                np.save(npy_data_save_path_for, Scan_data_for, allow_pickle = True)
                np.save(npy_data_save_path_back, Scan_data_back, allow_pickle = True)
                cv2.imwrite(image_data_save_path_for, image_for)                # save the image
                cv2.imwrite(image_data_save_path_back, image_back)              # save the image
                # cv2.namedWindow('image_for', cv2.WINDOW_NORMAL)
                # cv2.namedWindow('image_back', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('image_for', 400, 400)
                # cv2.resizeWindow('image_back', 400, 400)
                # cv2.imshow('image_for', image_for)
                # cv2.imshow('image_back', image_back)
            cv2.waitKey(100)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    def lineScan_data_consumer(self):
        number_of_line_scan = 0
        self.line_start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
        lineScan_save_path = self.log_path + '/' + 'lineScan' +'/'
        if not os.path.exists(lineScan_save_path + self.line_start_time + '_line_scan_' + str(self.line_scan_change_times).zfill(5)):
            
            os.makedirs(lineScan_save_path + self.line_start_time + '_line_scan_' + str(self.line_scan_change_times).zfill(5))

        
        while True:
            lineScandata_1 = self.lineScanQueue.get()
            # print('lineScan_data_consumer_works')
            
            # time_per_line = nanonis.ScanSpeedGet()['Forward time per line']
            # time.sleep(time_per_line) # wait for the line scan data to be collected

            time.sleep(0.153) # wait for the line scan data to be collected
            if lineScandata_1 == 'end':                                                             # if the line scan data producer is over, stop the line scan data consumer than draw the line scan data
                self.line_scan_signal = 0
                break                                                                               # end the lineScan_data_consumer
            if len(self.nanocoodinate_list) > 1:                                                         # if the nanocoodinate list have more than one nanocoodinate, save the line scan data
                lineScandata_1_for = fit_line(lineScandata_1['line_Scan_data_for'])                         # t-1 line scan data for
                lineScandata_1_back = fit_line(lineScandata_1['line_Scan_data_back'])                       # t-1 line scan data back
                # np.save(lineScan_save_path + self.line_start_time + '_line_scan_' + str(self.line_scan_change_times).zfill(5) +'/lineScandata_'+str(number_of_line_scan).zfill(5) +'.npy', 
                #         [self.AdjustTip_flag ,self.nanocoodinate_list[-1], self.nanocoodinate_list[-2] ,lineScandata_1_for, lineScandata_1_back])
                # print('lineScandata_{}_for.npy'.format(number_of_line_scan) + ' is saved')
                
                # if the line scan max - min is bigger than 500, set the skip flag to 1
                if linescan_max_min_check(lineScandata_1_for) > self.convert(self.linescan_max_min_threshold):
                    self.threshold_list.append(1)
                else:
                    self.threshold_list.append(0)
                # if the threshold_list is full with 1, the scan will be skiped
                if len(self.threshold_list) == self.len_threshold_list and all(threshold == 1 for threshold in self.threshold_list):
                    self.skip_flag = 1      # switch the skip flag to 1, because the line scan data is out of the threshold several times continuously
                    self.threshold_list.clear() # clear the threshold list, recount the threshold
                
                self.skip_list.append(self.skip_flag) # add the skip flag to the skip list
                    
                # if len(self.skip_list) ==10 and all(skip == 1 for skip in self.skip_list):


                if len(self.skip_list) >= 5 and all(skip == 1 for skip in self.skip_list):
                    self.aggressive_tip_flag = 1
                    self.skip_list.clear()      # clear the skip list, recount the skip flag
                    print('The line scan is skiped')

                number_of_line_scan += 1

    # def a function that content the line scan producer and consumer
    def line_scan_thread_activate(self):
        if self.line_scan_activate == 1:

            if self.AdjustTip_flag == 1:
                scan_continue_time = self.Z_fixed_scan_time
            else:
                scan_continue_time = -(1/200)*self.R**2 + 1.1*self.R
            t1 = threading.Thread(target = self.lineScan_data_producer, args=(0, scan_continue_time), daemon=True)               # lunch the line scan data producer
            t2 = threading.Thread(target = self.lineScan_data_consumer, daemon=True)                                             # lunch the line scan data consumer
            t1.start()
            t2.start()
            self.line_scan_signal = 1
            t1.join()
            t2.join()

            linescan_messages = [
                f'Line scan will continue for {scan_continue_time} seconds',
                'line scanning...',
                'lineScan has been set',
                'lineScan complete!',
            ]
            if self.skip_flag == 1:
                linescan_messages.extend(['creating skip data...', 'Scaning complete!'])

            linescan_diag = {
                'label': 'bad' if self.skip_flag == 1 else 'good',
                'reason': 'linescan_precheck_skip_next_scan' if self.skip_flag == 1 else 'linescan_precheck_pass',
                'severity': 'severe' if self.skip_flag == 1 else 'mild',
                'defect_tags': ['linescan_out_of_threshold'] if self.skip_flag == 1 else [],
                'confidence': 1.0,
                'source_agent': 'linescan',
                'messages': linescan_messages,
            }
            print('[LineScan Agent] structured diagnosis:')
            print(json.dumps(linescan_diag, ensure_ascii=False, indent=2))

    def tip_path_visualization(self):
        if self.mode == 'new':
            self.start_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
            self.ScandataQueue_1 = Queue(5)
            self.tipdataQueue = Queue(5)
            self.circle_list = []  # the list to save the circle which importantly to the tip path
            self.circle_list_save = []  # the list to save the circle as npy file
            self.nanocoodinate_list = []  # the list to save the coodinate which send to nanonis directly
            self.visual_circle_buffer_list = []
            self.line_scan_change_times = 0  # initialize the line scan change times to 0
            self.episode_count = 0  # initialize the episode count to 0
            self.AdjustTip_flag = 0
            self.len_threshold_list = 10
            self.threshold_list = deque([],
                                        maxlen=self.len_threshold_list)  # the list to save the threshold of the line scan data, if threshold_list is full with 1, the scan will be skiped
            self.skip_list = deque([],
                                   maxlen=10)  # the list to save the skip flag, if the skip_list is full with 1, gave the tip a aggressive tip shaper
            self.skip_flag = 0  # 0: the scan is not skiped, 1: the scan is skiped
            self.aggressive_tip_flag = 0
            self.agent_upgrate = 1
            self.log_path = self.main_data_save_path + '/' + self.start_time
            os.makedirs(self.log_path, exist_ok=True)
            self.memory_path = os.path.join(self.main_data_save_path, 'memory', self.start_time)
            self.Scan_edge_SI = self.convert(self.Scan_edge)  # the scan edge in SI unit   Scan_edge_SI = 30 * 1e-9

            self.scan_square_edge = int(
                self.convert(self.Scan_edge) * 10 ** 9)  # the scan square edge in pix  scan_square_edge = 30

            self.tip_path_img = np.ones((self.plane_size, self.plane_size, 3), np.uint8) * 255  # the tip path image


            self.R_init = self.scan_square_edge - 1  # initialize the Radius of tip step
            self.R_max = self.R_init * 3
            self.R_step = int(0.5 * self.R_init)
            self.R = self.R_init

            # initialize the other parameters that appear in the function
            self.Scan_data = {}  # the dictionary to save the scan data
            self.image_for = None  # 2D nparray the image of the scan data, have been nomalized and linear background
            self.image_back = None
            self.equalization_for = None  # the equalization image of the image_for and image_back
            self.equalization_back = None
            self.image_for_tensor = None  # the tensor of the image, 4 dimension, [1, 1, 256, 256]
            self.image_back_tensor = None
            self.image_save_time = None  # when the image is saved in log
            self.npy_data_save_path = self.log_path + '/' + 'npy'  # self.log_path = './log/' + self.start_time
            self.image_data_save_path = self.log_path + '/' + 'image'
            self.equalization_save_path = self.log_path + '/' + 'equalize'
            self.segmented_image_path = None  # the path of the segmented image saving
            self.nemo_nanocoodinate = None  # the nanocoodinate of the nemo point, the format is SI unit
            self.coverage = None  # the moleculer coverage of the image
            self.line_start_time = None
            self.episode_start_time = None  # the start time of the episode

            # initialize the queue, the Queue is used to communicate between different threads
            self.lineScanQueue = Queue(5)  # lineScan_data_producer → lineScan_data_consumer
            self.lineScanQueue_back = Queue(5)  # lineScan_data_consumer → lineScan_data_producer
            self.ScandataQueue = Queue(5)  # batch_scan_producer → batch_scan_consumer

            self.tippathvisualQueue = Queue(5)
            self.scan_start_time = None
            self.per_scan_time = 0
            self.total_scan_time = 0
            self._init_scan_step_recorder()
        square_color_hex = "#BABABA"                        # good image color
        square_bad_color_hex = "#FE5E5E"                    # bad image color
        line_color_hex = "#8AAEFA"                          # tip path line color
        border_color_hex = "#64FF00"                        # the border color of the whole plane
        scan_border_color_hex = "#FFC8CB"                   # usually the area of the 70% of the whole plane

        sample_bad_color_hex = "#FF6E6E"       #FF6E6E       #000000      # the color of the bad LineScan data

        color_max = "#385723"#385723  #C55A11
        color_min = "#C5E0B4"#C5E0B4  #2E75B6

        border_color = Hex_to_BGR(border_color_hex)
        scan_border_color = Hex_to_BGR(scan_border_color_hex)
        square_good_color = Hex_to_BGR(square_color_hex)
        square_bad_color = Hex_to_BGR(square_bad_color_hex)
        sample_bad_color = Hex_to_BGR(sample_bad_color_hex)
        line_color = Hex_to_BGR(line_color_hex)        

        # cv2.namedWindow('Tip Path', cv2.WINDOW_KEEPRATIO)
        # cv2.resizeWindow('Tip Path', 800, 800)
        

        # creat a 400*400 pix white image by numpy array
        cv2.rectangle(self.tip_path_img, (0, 0), (self.plane_size, self.plane_size), border_color, 10) # draw the border of the plane, which is the same color as the nanonis scan border on the scan controloer
        
        scan_border_left_top =  round(self.plane_size/2 * (1 - self.real_scan_factor))
        scan_border_right_bottom = round(self.plane_size/2 * (1 + self.real_scan_factor))
        cv2.rectangle(self.tip_path_img, (scan_border_left_top, scan_border_left_top), (scan_border_right_bottom, scan_border_right_bottom), scan_border_color, 10)

        # while True:
        while not self.tippath_stop_event.is_set():
            time.sleep(1)
            if not self.tippathvisualQueue.empty():
                circle = self.tippathvisualQueue.get()

                if len(circle) <= 2:                                                                        # data is a point which is the point of ready to scan 
                    cv2.circle(self.tip_path_img, (round(circle[0]), round(circle[1])), round(self.scan_square_edge/5) , (255, 0, 0), -1)
                elif len(circle) >= 4:                                                                      # data is a circle which is the circle of already scanned
                    left_top, right_bottom = center_to_square((circle[0], circle[1]), self.scan_square_edge)
                    t = circle[4]   #the coverge fiactor of the image
                    cover_color = interpolate_colors(color_min, color_max, t)                    
                    
                    if self.skip_flag == 1:                                                     # if the scan is skiped, the color of the square is set on
                        cover_color = sample_bad_color                    
                    if circle[3] == 1:
                        square_color = square_good_color    # good image color
                    elif circle[3] == 0:
                        square_color = square_bad_color     # bad image color
                    
                    if len(self.circle_list) == 1:
                        
                        # cv2.rectangle(self.tip_path_img, left_top, right_bottom, square_color, -1)
                        cv2.rectangle(self.tip_path_img, left_top, right_bottom, cover_color, -1)           # the box will be colored 
                        cv2.rectangle(self.tip_path_img, left_top, right_bottom, square_color, 3)
                        self.visual_circle_buffer_list.append(circle)                                            # add the first circle to the self.visual_circle_buffer_list

                    elif len(self.circle_list) > 1:
                        (Xn_1,Yn_1) = (self.visual_circle_buffer_list[-1][0], self.visual_circle_buffer_list[-1][1])
                        # cv2.rectangle(self.tip_path_img, left_top, right_bottom, square_color, -1)                        # draw the square
                        

                        cv2.rectangle(self.tip_path_img, left_top, right_bottom, cover_color, -1)           # the box will be colored 
                        cv2.rectangle(self.tip_path_img, left_top, right_bottom, square_color, 3)           # the edge of the box
                        cv2.line(self.tip_path_img, (round(Xn_1), round(Yn_1)), (round(circle[0]), round(circle[1])), line_color, 4)                                # ues the last circle center to draw the line to show the tip path
                        self.visual_circle_buffer_list.append(circle)                                            # add the new circle to the self.visual_circle_buffer_list
                        # delete the first circle in the self.visual_circle_buffer_list
                        if len(self.visual_circle_buffer_list) > 2:
                            self.visual_circle_buffer_list.pop(0)

                    else:
                        raise ValueError('the length of the circle list is not right')

                    cv2.imwrite(self.log_path + '/tip_path' +'.jpg', self.tip_path_img)
                    # save circle_list as a npy file
                    # np.save(self.log_path + '/circle_list.npy', self.circle_list)
                    # save coodinate_list as a npy file
                    # np.save(self.log_path + '/nanocoodinate_list.npy', self.nanocoodinate_list)
                elif circle == 'end':
                    # save the tip path image
                    cv2.imwrite(self.log_path + '/tip_path' +'.jpg', self.tip_path_img)
                    # np.save(self.log_path + '/circle_list.npy', self.circle_list)
                    # np.save(self.log_path + '/nanocoodinate_list.npy', self.nanocoodinate_list)
                    print('The tip path image is saved as ' + self.log_path + '/tip_path' +'.jpg')
                    break

                else:
                    raise ValueError('the circle data is not a point or a circle')

                self.tipdataQueue.put(self.tip_path_img)
                # cv2.imshow('Tip Path', self.tip_path_img)

            cv2.waitKey(100)

    def move_to_next_point(self):
        if self.navigator_mode == 'custom':
            points = self.navigator_custom_points if isinstance(self.navigator_custom_points, list) else []
            if len(points) > 0 and int(self.navigator_custom_index) >= len(points):
                print('[Planning Agent] custom path exhausted, stop scanning loop.')
                return False

        if len(self.nanocoodinate_list) == 0:
            self.inter_closest = self.navigator_next_point()                                       # mode-aware first point
            self.nanocoodinate = pix_to_nanocoordinate(self.inter_closest,plane_size = self.plane_size)
            self.nanocoodinate_list.append(self.nanocoodinate)
        else:
            self.inter_closest = self.navigator_next_point()                                       # calculate next center via Planning Agent
            self.nanocoodinate = pix_to_nanocoordinate(self.inter_closest,plane_size = self.plane_size)
            self.nanocoodinate_list.append(self.nanocoodinate)

        if self.tippathvisualQueue.full():
            try:
                self.tippathvisualQueue.get_nowait()
                print('[Queue] tippathvisualQueue full, drop oldest item')
            except queue.Empty:
                pass
        self.tippathvisualQueue.put_nowait(self.inter_closest)                                             # put the inter_closest to the tippathvisualQueue
        print('The scan center is ' + str(self.inter_closest))

        print('move to the area...')
        self.ScanFrameSet(self.nanocoodinate[0],self.nanocoodinate[1] + 0.5*self.Scan_edge_SI, self.Scan_edge, 1e-15, angle = 0)
        os.makedirs(self.log_path, exist_ok=True)
        np.save(self.log_path + '/nanocoodinate_list.npy', self.nanocoodinate_list)
        return True
    
    
    # def a function to segment the image
    def image_segmention(self, Scan_image):
        self.segmented_image_path = self.image_data_save_path + '/segmented_image/'
        if not os.path.exists(self.segmented_image_path):                            # create the segmented_image folder if not exist
            os.makedirs(self.segmented_image_path)
        # self.nemo_pointnemo_point in here the dift persentage of the matrix image 
        seg_result = Segmented_image(Scan_image, self.segmented_image_path, model_path = self.segment_model_path)

        # Segmented_image normally returns (nemo_point, coverage).
        if isinstance(seg_result, (tuple, list)) and len(seg_result) >= 2:
            nemo_point = seg_result[0]
            try:
                self.coverage = float(seg_result[1])
            except Exception:
                self.coverage = 0.0
        else:
            nemo_point = seg_result
            if self.coverage is None:
                self.coverage = 0.0

        if not isinstance(nemo_point, (tuple, list, np.ndarray)) or len(nemo_point) < 2:
            raise TypeError(f'Invalid nemo_point from Segmented_image: {repr(seg_result)}')

        nemo_point_pix = (self.inter_closest[0]+nemo_point[0]*self.scan_square_edge, self.inter_closest[1]+nemo_point[1]*self.scan_square_edge)
        self.nemo_nanocoodinate = pix_to_nanocoordinate(nemo_point_pix, plane_size = self.plane_size)   

        print('The nemo point is ' + str(nemo_point_pix))
        print('The coverage is ' + str(round(self.coverage,2)))
    
    
    # def a function to predict the scan qulity
    def image_recognition(self):
        if self.scan_was_skipped:
            self.last_evaluation_info = {
                'label': 'bad',
                'reason': 'linescan_precheck_skip_next_scan',
                'severity': 'severe',
                'defect_tags': ['linescan_out_of_threshold'],
                'confidence': 1.0,
                'source_agent': 'linescan',
            }
            scan_qulity = 0
            self.image_segmention(self.image_for)
            self.circle_list.append([self.inter_closest[0], self.inter_closest[1], self.R, scan_qulity])
            self.circle_list_save.append([self.inter_closest[0], self.inter_closest[1], self.R, scan_qulity, self.coverage])
            np.save(self.log_path + '/circle_list.npy', self.circle_list_save)
            if self.tippathvisualQueue.full():
                try:
                    self.tippathvisualQueue.get_nowait()
                    print('[Queue] tippathvisualQueue full, drop oldest item')
                except queue.Empty:
                    pass
            self.tippathvisualQueue.put_nowait([self.inter_closest[0], self.inter_closest[1], self.R, scan_qulity, self.coverage])
            return scan_qulity

        # judge the gap between the max and min of the image
        # if the gap is bigger than the threshold, the scan is skiped
        data_linear = linear_whole(self.Scan_data['Scan_data_for']['data'])
        line_gap = float(linescan_max_min_check(data_linear))
        line_gap_threshold = float(self.convert(self.scan_max_min_threshold))
        print('The gap between the max and min of the image is ' + str(line_gap))

        if line_gap >= line_gap_threshold:
            self.skip_flag = 1
            print('The scan is skiped')

        # use Kimi LLM Evaluation Agent to predict quality and provide diagnosis
        diagnosis = evaluate_image_quality(self.image_for, model_path=self.quality_model_path, return_prompt_debug=False)
        probability = float(diagnosis.get('good_probability', 0.0))
        diagnosis_label = str(diagnosis.get('label', 'unknown')).strip().lower()
        diagnosis_severity = str(diagnosis.get('severity', 'unknown')).strip().lower()
        recommended_action_bias = str(diagnosis.get('recommended_action_bias', 'pulse_first')).strip().lower()

        if diagnosis_label == 'good':
            recommended_action_bias = ''
        elif diagnosis_severity == 'severe':
            recommended_action_bias = 'tipshaper_first'

        self.last_evaluation_info = {
            'label': str(diagnosis.get('label', 'unknown')),
            'reason': str(diagnosis.get('rationale', '')),
            'severity': str(diagnosis.get('severity', 'unknown')),
            'defect_tags': diagnosis.get('defect_tags', []),
            'recommended_action_bias': recommended_action_bias,
            'confidence': float(diagnosis.get('confidence', probability)),
            'good_probability': probability,
            'source_agent': 'evaluation',
        }

        eval_label = str(self.last_evaluation_info.get('label', 'unknown')).strip().lower()
        eval_severity = str(self.last_evaluation_info.get('severity', 'unknown')).strip().lower()
        eval_defect_tags = self.last_evaluation_info.get('defect_tags', [])
        if not isinstance(eval_defect_tags, list):
            eval_defect_tags = []
        eval_defect_tags = [str(tag) for tag in eval_defect_tags]

        print('[Evaluation Agent] structured diagnosis:')
        print(json.dumps(self.last_evaluation_info, ensure_ascii=False, indent=2))
        if eval_label == 'good' and eval_severity in ('mild', 'moderate') and len(eval_defect_tags) > 0:
            print(
                '[Evaluation Agent] note: overall GOOD with minor defects '
                f'(severity={eval_severity}, defect_tags={eval_defect_tags}).'
            )

        # Conditioning is triggered by Evaluation Agent bad result only.
        # skip_flag is kept as diagnostic/scan-control signal and should not force a bad label here.
        if probability > self.scan_qulity_threshold:  # 0.5 is the self.scan_qulity_threshold of the probability
            scan_qulity = 1 # good image
            self.good_image_count +=1
        else:
            scan_qulity = 0 # bad image

        # calculate the R depend on the scan_qulity
        if len(self.circle_list) == 0:                                                      # if the circle_list is empty, initialize the R 
            self.R = self.R_init
        else:
            self.R = increase_radius(scan_qulity, self.circle_list[-1][2], self.R_init, self.R_max, self.R_step)     # increase the R

        self.image_segmention(self.image_for)
        self.circle_list.append([self.inter_closest[0], self.inter_closest[1], self.R, scan_qulity])
        # save the circle_list as a npy file
        self.circle_list_save.append([self.inter_closest[0], self.inter_closest[1], self.R, scan_qulity, self.coverage])
        np.save(self.log_path + '/circle_list.npy', self.circle_list_save)
        if self.tippathvisualQueue.full():
            try:
                self.tippathvisualQueue.get_nowait()
                print('[Queue] tippathvisualQueue full, drop oldest item')
            except queue.Empty:
                pass
        self.tippathvisualQueue.put_nowait([self.inter_closest[0], self.inter_closest[1], self.R, scan_qulity, self.coverage])        
        #save the scan image via the scan_qulity

        # create the good_scan and bad_scan folder in self.image_data_save_path if not exist
        if not os.path.exists(self.image_data_save_path + '/good_scan'):
            os.makedirs(self.image_data_save_path + '/good_scan')
        if not os.path.exists(self.image_data_save_path + '/bad_scan'):
            os.makedirs(self.image_data_save_path + '/bad_scan')
        self.image_save_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
        if scan_qulity == 1:
            cv2.imwrite(self.image_data_save_path + '/good_scan/' + 'Scan_data_for'+ self.image_save_time +'.png', self.image_for)  #save the image in the good_scan folder
            cv2.imwrite(self.image_data_save_path + '/good_scan/' + 'Scan_data_back'+ self.image_save_time +'.png', self.image_back)  #save the image in the good_scan folder

        else:
            cv2.imwrite(self.image_data_save_path + '/bad_scan/' + 'Scan_data_for'+ self.image_save_time +'.png', self.image_for)   #save the image in the bad_scan folder
            cv2.imwrite(self.image_data_save_path + '/bad_scan/' + 'Scan_data_back'+ self.image_save_time +'.png', self.image_back)   #save the image in the bad_scan folder

        return scan_qulity

    #def a function to select the action for nanonis
    def select_nanonis_action(self, state, select_mode = 'agent'):
        # DQN removed: fallback to random or deterministic rule
        if select_mode == 'random':
            nanonis_action_number = random.randint(0, 5)
        else:
            nanonis_action_number = random.randint(0, 5)

        if nanonis_action_number == 0:
            self.BiasPulse(-2, width=0.05)
        elif nanonis_action_number == 1:
            self.BiasPulse(-4, width=0.05)
        elif nanonis_action_number == 2:
            self.BiasPulse(-6, width=0.05)
        elif nanonis_action_number >= 3:
            self.TipXYSet(self.nemo_nanocoodinate[0], self.nemo_nanocoodinate[1])
            time.sleep(1)
            if nanonis_action_number == 3:
                self.TipShaper(TipLift='-1.5n')
            elif nanonis_action_number == 4:
                self.TipShaper(TipLift='-2.5n')
            elif nanonis_action_number == 5:
                self.TipShaper(TipLift='-6n')
        print('ACTION:', nanonis_action_number)
        return nanonis_action_number

    # def a function to crate the reward
    def reward_function(self, action_evaluation):
        return 10 if action_evaluation else -1
    
    # def a function to creata the trajectory
    def create_trajectory(self,scan_qulity):
        time.sleep(0.5)
        if self.skip_flag == 1:# if the scan is skiped, return
            self.skip_flag = 0  # reset the skip flag
            return
        if scan_qulity == 0: # meet the bad image
            # if the memory_path + '/' + episode_start_time is not exist, create the folder
            # episode_start_time = time.strftime('%H-%M-%S',time.localtime(time.time()))                                  # record the start time of the trajectory
            
            self.nanonis_action_number = self.select_nanonis_action(self.image_for, select_mode = 'agent')                              # implement the action which have the hightest Q value accordding to the image and policy_net
            
            if len(self.trajectory_state_list) == 0:
                self.episode_start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))                                  # record the start time of the trajectory
                self.trajectory_state_list.append((self.image_for,self.image_back))
                self.trajectory_action_list.append(self.nanonis_action_number)

            elif len(self.trajectory_state_list) > 0:
                print('the tip fix is not successful...')
                action_evaluation = 0
                
                self.trajectory_state_list.append((self.image_for,self.image_back))                       # second bad image, state_list aways have one more element than next_state_list.
                self.trajectory_next_state_list.append((self.image_for,self.image_back))                  # second bad image
                self.trajectory_action_list.append(self.nanonis_action_number)                                          # the action which have the hightest Q value
                self.trajectory_reward_list.append(self.reward_function(action_evaluation))                             # the reward of the action

                print('reward:', self.trajectory_reward_list[-1])
                # set trajectory_for to a dictionary
                trajectory_for ={ 'state': self.trajectory_state_list[-2][0].tolist(), 
                                 'action': self.trajectory_action_list[-1], 
                                'reward': self.trajectory_reward_list[-1], 
                                'next_state': self.trajectory_next_state_list[-1][0].tolist()}
                
                trajectory_back = {'state': self.trajectory_state_list[-2][1].tolist(),
                                   'action': self.trajectory_action_list[-1], 
                                'reward': self.trajectory_reward_list[-1], 
                                'next_state': self.trajectory_next_state_list[-1][0].tolist()}
                
                trajectory_start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
                trajectory_for_name = trajectory_start_time +'_trajectory_for'+ '.json'
                trajectory_back_name = trajectory_start_time +'_trajectory_back'+ '.json'
                # save the trajectory to the replay memory
                
                trajectory_path = self.memory_path + '/' + self.episode_start_time + 'episode'
                self.memory.save(trajectory_path, trajectory_for_name, trajectory_for)
                self.memory.save(trajectory_path, trajectory_back_name, trajectory_back)


        elif scan_qulity == 1:  # meet the good image
            if len(self.trajectory_state_list) > 0:   # first image after the bad image
                print('the tip fix is successful !!!')
                action_evaluation = 1

                self.trajectory_next_state_list.append((self.image_for,self.image_back))                 # add the last good image to the next_state_list but not the state_list to complete the episode
                self.trajectory_reward_list.append(self.reward_function(action_evaluation))

                print('reward:', self.trajectory_reward_list[-1])
                trajectory_for = {'state': self.trajectory_state_list[-1][0].tolist(),
                                  'action': self.trajectory_action_list[-1], 
                                'reward': self.trajectory_reward_list[-1], 
                                'next_state': self.trajectory_next_state_list[-1][0].tolist()}
                trajectory_back = {'state': self.trajectory_state_list[-1][1].tolist(),
                                    'action': self.trajectory_action_list[-1], 
                                    'reward': self.trajectory_reward_list[-1], 
                                    'next_state': self.trajectory_next_state_list[-1][1].tolist()}
                
                trajectory_start_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
                trajectory_for_name = trajectory_start_time +'_trajectory_for'+ '.json'
                trajectory_back_name = trajectory_start_time +'_trajectory_back'+ '.json'
                
                trajectory_path = self.memory_path + '/' + self.episode_start_time + 'episode'
                self.memory.save(trajectory_path, trajectory_for_name, trajectory_for)
                self.memory.save(trajectory_path, trajectory_back_name, trajectory_back)

                # initialize the trajectory list
                self.trajectory_state_list.clear()                # the list to save the one step in memo
                self.trajectory_action_list.clear()
                self.trajectory_reward_list.clear()
                self.trajectory_next_state_list.clear()
    # def a function to pulse
    def bias_pulse_set(self, width, bias, wait=True):
        while True:
            try:
                # 检查是否两个队列都为空
                if width is None and bias is None:
                    break
                # 如果队列有值，发送脉冲命令
                if width is not None or bias is not None:
                    self.send('Bias.Pulse', 'uint32', int(wait), 'float32', float(
                        width), 'float32', float(bias), 'uint16', 0, 'uint16', 0)
                    print(f"Pulse with voltage: {bias}V\n"
                          f"Pulse with width:{width}m")
                    bias = None
                    width = None

            except Exception as e:
                # 处理异常，例如队列操作失败或send方法失败
                print(f"An error occurred: {e}")
                # 可以选择在这里退出循环，或者设置标志来通知其他部分的代码
                break
    # def a function to record the statues of scan
    def controls_set(self, controls):
        while True:
            try:
                # 检查是否两个队列都为空
                if controls is None:
                    break
                # 如果队列有值，发送脉冲命令
                if controls == 1:
                    self.send('Scan.Action', 'uint16', controls, 'uint32', 0)
                    print(f"programming stop")
                    controls = None
                elif controls == 2:
                    self.send('Scan.Action', 'uint16', controls, 'uint32', 0)
                    print(f"programming pause")
                    controls = None
                elif controls == 3:
                    self.send('Scan.Action', 'uint16', controls, 'uint32', 0)
                    print(f"programming resume")
                    controls = None

            except Exception as e:
                # 处理异常，例如队列操作失败或send方法失败
                print(f"An error occurred: {e}")
                # 可以选择在这里退出循环，或者设置标志来通知其他部分的代码
                break
    # def a function to set the scan speed
    def speed_set(self, time_per_frame):
        Keep_parameter_constant = 2
        Speed_ratio = 1
        height = self.ScanBufferGet()['Lines']
        time_per_line = time_per_frame/height/2
        Forward_linear_speed = 1
        Backward_linear_speed = 1
        while True:
            try:
                # 检查是否两个队列都为空
                if time_per_frame is None:
                    break
                # 如果队列有值，发送脉冲命令
                if time_per_frame is not None:
                    self.send('Scan.SpeedSet', 'float32', Forward_linear_speed, 'float32', Backward_linear_speed,
                              'float32', time_per_line, 'float32', time_per_line, 'uint16',
                              Keep_parameter_constant, 'float32', Speed_ratio)
                    time_per_frame = None
            except Exception as e:
                # 处理异常，例如队列操作失败或send方法失败
                print(f"An error occurred: {e}")
                break
    def z_position_adjust(self):
        z_position = self.ZPosGet()
        if z_position < -2.3E-7:
            self.MotorMoveSet(4, 1)
            z_position_change = self.ZPosGet()
            q = 2.3E-7/(z_position_change - z_position)-1
            self.MotorMoveSet(4, q)
        elif z_position > 2.3E-7:
            self.MotorMoveSet(5, 1)
            z_position_change = self.ZPosGet()
            q = 2.3E-7/(z_position - z_position_change)-1
            self.MotorMoveSet(5, q)

    def TipShaper_set(self, TipLift, Switch_Off_Delay=0.05, Lift_Time_1=0.1, Bias_Setting_Time=0.06,
                  Lift_Time_2=0.06,  Lifting_Bias=-1.0, Lifting_Height='2n',
                  End_Wait_Time=0.01, stepstotarget=10):

        TipLift = self.try_convert(TipLift)
        Lifting_Height = self.try_convert(Lifting_Height)

        Origin_ScanStatus = self.ScanStatusGet()  # check the scan status
        if Origin_ScanStatus == 1:
            self.ScanPause()  # Pause the scan (not stop), if the scan is running

        ZCtrl_status = self.ZCtrlOnOffGet()  # get current Z control status

        Z_current = self.TipZGet()  # get current Z position

        Bias_current = self.BiasGet()  # get current Bias

        self.ZCtrlOnOffSet("off")  # close Z control

        time.sleep(Switch_Off_Delay)  # wait for Switch_Off_Delay

        # use for loop to simulate the uniform motion of the tip
        for i in range(stepstotarget):
            self.TipZSet(Z_current + (i + 1) * (TipLift / stepstotarget))  # set the new Z position

            time.sleep(Lift_Time_1 / stepstotarget)  # wait for Lift_Time_1/stepstotarget time

        time.sleep(1)

        self.BiasSet(Lifting_Bias)  # set the new Bias

        time.sleep(Bias_Setting_Time)  # wait for Bias_Setting_Time

        # use for loop to simulate the uniform motion of the tip
        for i in range(stepstotarget):
            self.TipZSet(Z_current + TipLift + (i + 1) * (
                        Lifting_Height - TipLift) / stepstotarget)  # set the new Z position → Lifting_Height

            time.sleep(Lift_Time_2 / stepstotarget)  # wait for Lift_Time_2/stepstotarget time

        time.sleep(1)
        # self.Tipshaper_signal = 1 #set a signal while tipshaper
        self.BiasSet(Bias_current)  # set the new Bias → Bias_current

        time.sleep(End_Wait_Time)  # wait for End_Wait_Time

        # reopen Z control or not?
        if self.iftipshaper == 1:
            x = self.ScanFrameGet()['center_x'] + 5E-8
            y = self.ScanFrameGet()['center_y'] + 5E-8
            self.ScanFrameSet(x, y, 5E-8, 5E-8, 0)
            self.iftipshaper = 0
        if ZCtrl_status == 1:
            self.ZCtrlOnOffSet("on")
        else:
            self.ZCtrlOnOffSet("off")

        if Origin_ScanStatus == 1:
            self.ScanResume()  # Resume the scan (not restart), if the scan was running

    # def a function to record pause_time while scan running
    def calculate_total_scan_time(self, expected_total_time):
        self.scan_start_time = None
        self.per_scan_time = 0
        self.total_scan_time = 0
        while True:
            status = self.ScanStatusGet()
            if self.Tipshaper_signal == 1:
                self.Tipshaper_signal = 0
                break
            elif self.frame_move_signal == 1:
                self.frame_move_signal = 0
                break
            elif expected_total_time != self.current_time_per_frame():
                break
            elif status == 1:
                if self.scan_start_time is None:
                    self.scan_start_time = time.time()  # 记录扫描开始时间
            elif status == 0:
                if self.scan_start_time is not None:
                    # 计算当前扫描周期的时间
                    self.per_scan_time = time.time() - self.scan_start_time
                    # 累加到总扫描时间
                    self.total_scan_time += self.per_scan_time
                    # 重置扫描开始时间和当前扫描时间
                    self.scan_start_time = None
                    self.per_scan_time = 0

                    # 检查是否达到预期的总扫描时间
                    if self.total_scan_time+3 >= expected_total_time:
                        break
                if self.scan_start_time is None:
                    pass
            # 每秒检查一次状态
            time.sleep(2)
    def current_time_per_frame(self):
        true_t = (self.ScanSpeedGet()['Forward time per line'] + self.ScanSpeedGet()[
            'Backward time per line']) * \
            self.ScanBufferGet()['Lines']
        return true_t

    def Gainset(self, proportional, integral):
        while True:
            try:
                if proportional is None and integral is None:
                    break
                if proportional is not None or integral is not None:
                    self.send('ZCtrl.GainSet', 'float32', float(proportional), 'float32', float(
                        proportional/integral), 'float32', float(integral))
                    print(f"P-gain set: {proportional}pm\n"
                          f"I-gain set:{integral}pm/s")
                    proportional = None
                    integral = None

            except Exception as e:
                print(f"An error occurred: {e}")
                break
    def get_current_value(self):
        q = self.SignalsValsGet(2, (0, 30))
        return q['value']
    # function to get current and z
    def plot_realtime_current(self, fig_max_len=500, update_interval=10):
        xdata, ydata = deque(maxlen=fig_max_len), deque(maxlen=fig_max_len)
        fig, ax = plt.subplots(figsize=(10,6))
        ln, = ax.plot(xdata, ydata, animated=True)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        ax.set_ylabel('Current (pA)')
        ax.set_title('Real-time Current Measurement')

        # 更新图表
        def update(frame):
            current_value = abs(self.get_current_value()[0])*1E+12
            xdata.append(frame)
            ydata.append(current_value)
            ax.set_xlim(max(0, frame - fig_max_len), frame+1)
            ax.set_ylim(min(ydata), max(ydata))
            ln.set_data(xdata, ydata)
            return ln,

        ani = FuncAnimation(fig, update, interval=update_interval, blit=True)

        plt.show()

    def plot_realtime_z(self, fig_max_len=500):
        xdata, ydata = deque(maxlen=fig_max_len), deque(maxlen=fig_max_len)
        fig, ax = plt.subplots(figsize=(10,6))
        ln, = ax.plot(xdata, ydata, animated=True)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        ax.set_ylabel('Z(m)')
        ax.set_title('Real-time Z Measurement')

        def update(frame):
            current_value = abs(self.get_current_value()[1]) * 1E+12
            xdata.append(frame)
            ydata.append(current_value)
            ax.set_xlim(max(0, frame - fig_max_len), frame + 1)
            ax.set_ylim(min(ydata), max(ydata))
            ln.set_data(xdata, ydata)
            return ln,

        ani = FuncAnimation(fig, update, interval=10, blit=True)
        plt.show()

    def if_fix_tip(self):
        self.iftipshaper = 1
    def bias_pulse(self):
        self.send('Bias.Pulse', 'uint32', int(True), 'float32', float(5E-2), 'float32', float(-2), 'uint16', 0,
                          'uint16', 0)
    def move_to_next_area(self):
        self.ScanStop()
        self.ScanFrameSet(0, 0, 5E-8, 5E-8)
        self.ZCtrlWithdraw(1)
        time.sleep(0.5)
        self.MotorMoveSet('Z-', 100)
        time.sleep(0.5)
        self.MotorMoveSet('X-', 10)
        time.sleep(0.5)
        self.MotorMoveSet('Y-', 10)
        # self.nanonis.ZCtrlOnOffSet(1)
        self.AutoApproachOpen()
        time.sleep(0.5)
        self.AutoApproachSet()
        print("area move succeed, wait for auto approach")


def main(shutdown_event=None, navigation_instruction=None):
    nanonis = Mustard_AI_Nanonis()

    if navigation_instruction:
        nanonis.plan_navigator_from_text(user_instruction=navigation_instruction)
    else:
        # Planning Agent default behavior is custom path generation.
        nanonis.plan_navigator_from_text(user_instruction='The scan shall start from the top-left and proceed in a serpentine pattern until the entire area is covered.')

    # nanonis.tip_init(mode = 'new') # deflaut mode is 'new' mode = 'new' : the tip is initialized to the center and create a new log folder, mode = 'latest' : load the latest checkpoint
    nanonis.monitor_thread_activate()                                                # activate the monitor thread

    while tip_in_boundary(nanonis.inter_closest, nanonis.plane_size, nanonis.real_scan_factor):
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info('Shutdown requested before next scan step.')
            break

        nanonis.total_step_count += 1

        if not nanonis.move_to_next_point():                                             # move the scan area to the next point
            custom_total = len(nanonis.navigator_custom_points) if isinstance(nanonis.navigator_custom_points, list) else 0
            logging.info(
                f"[Planning Agent] scanning finished: total_steps={nanonis.total_step_count}, "
                f"custom_waypoints={custom_total}, consumed_index={nanonis.navigator_custom_index}"
            )
            break

        nanonis.AdjustTip_flag = nanonis.AdjustTipToPiezoCenter()                       # check & adjust the tip to the center of the piezo
        nanonis.line_scan_thread_activate()                                             # activate the line scan, producer-consumer architecture, pre-check the tip and sample

        nanonis.batch_scan_producer(nanonis.nanocoodinate, nanonis.Scan_edge, nanonis.scan_square_Buffer_pix, 0)    # Scan the area

        # Molecular target check before image quality evaluation.
        molecule_match_info = nanonis.run_molecule_match_agent()

        scan_qulity = nanonis.image_recognition()                                       # assessment the scan quality
        if isinstance(nanonis.last_evaluation_info, dict):
            nanonis.last_evaluation_info['molecule_match'] = molecule_match_info
        step_record_index = nanonis.record_scan_step(scan_qulity)

        if nanonis.scan_was_skipped:
            # Skip frames are synthetic placeholders; do not evaluate repair effect or trigger conditioning.
            logging.info('[Conditioning Agent] skip frame detected, bypass effectiveness check and repair action this round.')
            nanonis.save_checkpoint()
            time.sleep(3)
            continue

        if shutdown_event is not None and shutdown_event.is_set():
            logging.info('Shutdown requested after scan step.')
            break

        nanonis.record_post_action_evaluation(scan_qulity)

        # Evaluate previous conditioning action effectiveness on the first real scan result after repair.
        nanonis.update_conditioning_effectiveness(scan_qulity, eval_info=nanonis.last_evaluation_info)

        if scan_qulity == 0:
            eval_info = {
                'label': 'Bad',
                'reason': nanonis.last_evaluation_info.get('reason', 'evaluation_agent_bad'),
                'severity': nanonis.last_evaluation_info.get('severity', 'unknown'),
                'defect_tags': nanonis.last_evaluation_info.get('defect_tags', []),
                'recommended_action_bias': nanonis.last_evaluation_info.get('recommended_action_bias', 'pulse_first'),
                'good_probability': nanonis.last_evaluation_info.get('good_probability', 0.0),
                'confidence': nanonis.last_evaluation_info.get('confidence', 0.0),
                'gap_check': linescan_max_min_check(linear_whole(nanonis.Scan_data['Scan_data_for']['data']))
            }
            action_plan = nanonis.apply_conditioning_agent(eval_info, nanonis.image_for)
            nanonis.record_conditioning_action_for_step(step_record_index, action_plan)
            logging.info(f"[Conditioning Agent] action_plan={action_plan}")

        nanonis.save_checkpoint()                                                       # save the checkpoint
        time.sleep(3)

    if shutdown_event is not None and shutdown_event.is_set():
        try:
            logging.info('Shutdown requested; attempting to stop scan and withdraw tip.')
            if hasattr(nanonis, 'StopScanAndWithdraw'):
                nanonis.StopScanAndWithdraw()
        except Exception as exc:
            logging.exception('Error while requesting StopScanAndWithdraw: %s', exc)
        try:
            nanonis.save_checkpoint()
        except Exception:
            pass

    return nanonis


if __name__ == '__main__':
    main()
