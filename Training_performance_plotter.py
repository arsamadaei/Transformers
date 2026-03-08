# Training_performance_plotter.py
import sys
import pandas as pd
import numpy as np

from pathlib import Path
import psutil
from PyQt6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

from translate import translate
from train import get_model, get_ds, greedy_decode
from config import get_config, get_epoch_from_file
from tokenizers import Tokenizer

import json
import warnings

import torch
from torchmetrics.text.bleu import BLEUScore
import torch.nn as nn
import threading
import time
import math

# Altair kept for compatibility if train.py expects it
import altair as alt

# Import Mindmap for process visualization
import sys
sys.path.insert(0, str(Path(__file__).parent))
from processes import Mindmap, Process

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
ATTENTION_MODEL_LOADED = False
model = None
vocab_src = None
vocab_tgt = None
val_dataloader = None
config = None
SOURCE_TEXT = "No data loaded."
TARGET_TEXT = "No data loaded."
ATTN_LAYERS = [0, 1, 2, 3, 4, 5]


class GanttChartWidget(QtWidgets.QWidget):
    """
    Gantt chart widget for visualizing process timelines.
    Each process gets its own row. Time flows horizontally.
    Fixed labels on left, scrollable chart on right.
    """
    # Signal emitted when mouse hovers (for crosshair sync)
    sigHoverTime = QtCore.pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main horizontal splitter: scrollable labels | scrollable chart
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Left side: Scrollable labels panel
        self.labels_scroll = QtWidgets.QScrollArea()
        self.labels_scroll.setWidgetResizable(True)
        self.labels_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.labels_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.labels_scroll.setFixedWidth(250)
        
        self.labels_panel = QtWidgets.QWidget()
        labels_layout = QtWidgets.QVBoxLayout(self.labels_panel)
        labels_layout.setContentsMargins(5, 5, 5, 5)
        labels_layout.setSpacing(0)
        
        # Header for labels
        self.labels_header = QtWidgets.QLabel("Epoch | Process")
        self.labels_header.setFixedHeight(40)
        self.labels_header.setStyleSheet("font-weight: bold; font-size: 11px; border-bottom: 2px solid #333;")
        labels_layout.addWidget(self.labels_header)
        
        # Container for process labels
        self.labels_container = QtWidgets.QWidget()
        self.labels_layout = QtWidgets.QVBoxLayout(self.labels_container)
        self.labels_layout.setSpacing(2)
        self.labels_layout.setContentsMargins(0, 0, 0, 0)
        labels_layout.addWidget(self.labels_container)
        labels_layout.addStretch(1)
        
        self.labels_scroll.setWidget(self.labels_panel)
        self.main_layout.addWidget(self.labels_scroll)
        
        # Right side: Scrollable chart area
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Connect vertical scroll to sync with labels
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._sync_labels_scroll)
        
        # Chart container
        self.chart_container = QtWidgets.QWidget()
        self.chart_layout = QtWidgets.QVBoxLayout(self.chart_container)
        self.chart_layout.setSpacing(2)
        self.chart_layout.setContentsMargins(5, 5, 5, 5)
        self.scroll_area.setWidget(self.chart_container)
        
        self.main_layout.addWidget(self.scroll_area, stretch=1)
        
        # Store process data
        self.processes = {}
        self.process_labels = []  # List of label widgets
        self.min_time = 0
        self.max_time = 1
        self.time_scale = 50  # pixels per second
        self._epoch_range = (None, None)  # Filter range
        
        # Crosshair overlay
        self.crosshair_overlay = None
        
    def _sync_labels_scroll(self, value):
        """Sync label panel scroll with chart scroll."""
        # Labels are fixed, no need to scroll them
        pass
        
    def clear_gantt(self):
        """Clear all process rows from the chart."""
        # Clear labels
        while self.labels_layout.count():
            item = self.labels_layout.takeAt(0)
            if item.widget() and item.widget() != self.labels_header:
                item.widget().deleteLater()
        self.process_labels = []
        
        # Clear chart
        while self.chart_layout.count():
            item = self.chart_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.processes = {}
        self.crosshair_overlay = None
        self.current_epoch = None
        
    def set_epoch_range(self, start_epoch, end_epoch):
        """Set epoch range filter."""
        self._epoch_range = (start_epoch, end_epoch)
        
    def load_processes(self, processes_dict):
        """
        Load processes and create Gantt chart with one row per process.
        
        Args:
            processes_dict: Dictionary of {uid: process_data}
        """
        self.clear_gantt()
        
        if not processes_dict:
            return
            
        self.processes = processes_dict
        
        # Build process list with timeline data
        process_list = []
        for uid, proc in processes_dict.items():
            timeline = proc.get('timeline', {})
            init_time = timeline.get('initialized', 0)
            term_time = timeline.get('terminated', -1)
            if term_time == -1 or term_time < init_time:
                term_time = time.time()
            
            layer = proc.get('layer', 0)
            name = proc.get('name', 'Unknown')
            epoch = proc.get('epoch', 0)
            
            duration = term_time - init_time
            # Filter out instant processes (construction) but keep short forward passes
            # Forward passes can be very short (microseconds), so use a smaller threshold
            if duration < 0.000001:  # 1 microsecond threshold
                continue
            
            # Apply epoch filter if set
            start_ep, end_ep = self._epoch_range
            if start_ep is not None and epoch < start_ep:
                continue
            if end_ep is not None and epoch > end_ep:
                continue
                
            process_list.append({
                'uid': uid,
                'name': name,
                'init': init_time,
                'term': term_time,
                'layer': layer,
                'epoch': epoch,
                'duration': duration
            })
        
        if not process_list:
            return
            
        # Sort by epoch, then by start time
        process_list.sort(key=lambda x: (x['epoch'], x['init']))
        
        # Find time range (relative to start)
        self.min_time = min(p['init'] for p in process_list)
        self.max_time = max(p['term'] for p in process_list)
        total_duration = self.max_time - self.min_time
        
        # Normalize times
        for p in process_list:
            p['init'] -= self.min_time
            p['term'] -= self.min_time
        
        # Calculate width needed
        chart_width = max(800, int(total_duration * self.time_scale) + 100)
        
        # Colors for epochs
        epoch_colors = [
            (70, 130, 180),   # Steel blue
            (60, 179, 113),   # Medium sea green
            (255, 165, 0),    # Orange
            (147, 112, 219),  # Medium purple
            (255, 99, 71),    # Tomato
            (32, 178, 170),   # Light sea green
        ]
        
        # Create header row with time scale
        header = self._create_time_header(chart_width, total_duration)
        self.chart_layout.addWidget(header)
        
        # Add legend for colors
        legend = self._create_epoch_legend(process_list, epoch_colors)
        self.chart_layout.addWidget(legend)
        
        # Group processes by base task name, but split into multiple rows if they overlap
        # Extract base name: "train_epoch_05" -> "train_epoch", "EncoderBlock_L0" -> "EncoderBlock"
        def get_base_task_name(name):
            """Extract base task name by removing epoch/layer numbers."""
            import re
            base = re.sub(r'_\d+$', '', name)  # Remove _NN at end
            base = re.sub(r'_L\d+$', '', base)  # Remove _LNN at end
            return base
        
        # Group by base name first
        task_groups = {}  # {base_task_name: [process1, process2, ...]}
        for p in process_list:
            base_name = get_base_task_name(p['name'])
            if base_name not in task_groups:
                task_groups[base_name] = []
            task_groups[base_name].append(p)
        
        # For each base task group, assign processes to rows based on time overlap
        # If two processes overlap in time, they need separate rows
        def assign_to_rows(processes):
            """Assign processes to rows, creating new rows only when there's a time conflict."""
            if not processes:
                return []
            
            # Sort by start time
            sorted_procs = sorted(processes, key=lambda p: p['init'])
            
            # rows: list of (row_index, [processes_in_row])
            rows = []
            
            for p in sorted_procs:
                # Find a row where this process doesn't overlap with any existing process
                placed = False
                for row_procs in rows:
                    # Check if p overlaps with any process in this row
                    overlaps = False
                    for existing in row_procs:
                        # Overlap if: p starts before existing ends AND p ends after existing starts
                        if p['init'] < existing['term'] and p['term'] > existing['init']:
                            overlaps = True
                            break
                    
                    if not overlaps:
                        # Place in this row
                        row_procs.append(p)
                        placed = True
                        break
                
                if not placed:
                    # Create new row
                    rows.append([p])
            
            return rows
        
        # Build list of (base_name, row_index, [processes]) tuples
        rows_to_create = []  # [(base_name, row_index, [processes]), ...]
        for base_name in sorted(task_groups.keys()):
            processes = task_groups[base_name]
            rows = assign_to_rows(processes)
            for row_idx, row_procs in enumerate(rows):
                rows_to_create.append((base_name, row_idx, row_procs))
        
        # Sort rows: first by base_name, then by earliest start time in each row
        def sort_key(item):
            base_name, row_idx, procs = item
            earliest_start = min(p['init'] for p in procs) if procs else 0
            return (base_name, earliest_start)
        
        rows_to_create.sort(key=sort_key)
        
        # Create rows
        for base_name, row_idx, row_procs in rows_to_create:
            # Create label with row number if there are multiple rows for this task
            if row_idx > 0:
                label_text = f"{base_name[:24]} ({row_idx+1})"
            else:
                label_text = base_name[:28]
            
            label = QtWidgets.QLabel(label_text)
            label.setFixedHeight(28)
            label.setStyleSheet("font-size: 10px; padding: 2px 5px; border-bottom: 1px solid #eee; font-weight: bold;")
            label.setToolTip(f"Task: {base_name}\n{len(row_procs)} non-overlapping instances")
            self.labels_layout.addWidget(label)
            self.process_labels.append(label)
            
            # Create chart row
            row_widget = self._create_multi_epoch_row(row_procs, chart_width, epoch_colors)
            self.chart_layout.addWidget(row_widget)
        
        # Add stretch at bottom
        self.chart_layout.addStretch(1)
        self.labels_layout.addStretch(1)
        
        # Set container width
        self.chart_container.setMinimumWidth(chart_width + 50)
        
        # Create crosshair overlay
        self._create_crosshair()
        
    def _create_time_header(self, chart_width, total_duration):
        """Create time axis header."""
        header = QtWidgets.QWidget()
        header.setFixedHeight(40)
        header.setStyleSheet("background-color: #f5f5f5; border-bottom: 2px solid #333;")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 10, 5)
        header_layout.setSpacing(0)
        
        # Time markers
        num_markers = min(20, max(5, int(total_duration / 10)))
        interval = total_duration / num_markers if num_markers > 0 else 1
        
        for i in range(num_markers + 1):
            time_val = interval * i
            x_pos = int(time_val * self.time_scale)
            
            marker_widget = QtWidgets.QWidget()
            marker_layout = QtWidgets.QVBoxLayout(marker_widget)
            marker_layout.setContentsMargins(0, 0, 0, 0)
            marker_layout.setSpacing(0)
            
            # Tick mark
            tick = QtWidgets.QFrame()
            tick.setFixedWidth(1)
            tick.setFixedHeight(10)
            tick.setStyleSheet("background-color: #666;")
            marker_layout.addWidget(tick)
            
            # Label
            label = QtWidgets.QLabel(f"{time_val:.0f}s")
            label.setStyleSheet("font-size: 9px; color: #666;")
            label.setFixedWidth(40)
            marker_layout.addWidget(label)
            
            header_layout.addWidget(marker_widget)
            header_layout.addSpacing(int(interval * self.time_scale) - 40)
        
        header_layout.addStretch(1)
        return header
        
    def _create_epoch_legend(self, process_list, epoch_colors):
        """Create a legend showing epoch colors."""
        legend_widget = QtWidgets.QWidget()
        legend_widget.setFixedHeight(30)
        legend_layout = QtWidgets.QHBoxLayout(legend_widget)
        legend_layout.setContentsMargins(10, 5, 10, 5)
        legend_layout.setSpacing(10)
        
        legend_title = QtWidgets.QLabel("Epochs:")
        legend_title.setStyleSheet("font-weight: bold; font-size: 10px;")
        legend_layout.addWidget(legend_title)
        
        # Get unique epochs
        epochs = sorted(set(p['epoch'] for p in process_list))
        
        for epoch in epochs[:8]:  # Show up to 8 epochs
            color = epoch_colors[epoch % len(epoch_colors)]
            color_str = f"rgb({color[0]}, {color[1]}, {color[2]})"
            
            # Color box
            color_box = QtWidgets.QFrame()
            color_box.setFixedSize(16, 16)
            color_box.setStyleSheet(f"background-color: {color_str}; border: 1px solid #333;")
            legend_layout.addWidget(color_box)
            
            # Epoch label
            epoch_label = QtWidgets.QLabel(f"E{epoch}")
            epoch_label.setStyleSheet("font-size: 9px;")
            legend_layout.addWidget(epoch_label)
        
        if len(epochs) > 8:
            more_label = QtWidgets.QLabel(f"... +{len(epochs) - 8} more")
            more_label.setStyleSheet("font-size: 9px; color: #666;")
            legend_layout.addWidget(more_label)
        
        legend_layout.addStretch(1)
        return legend_widget
        
    def _create_multi_epoch_row(self, processes, chart_width, epoch_colors):
        """Create a row with multiple epoch bars for the same task."""
        row = QtWidgets.QWidget()
        row.setFixedHeight(28)
        row.setStyleSheet("background-color: white;")
        
        # Sort processes by start time
        processes = sorted(processes, key=lambda p: p['init'])
        
        # Create a bar for each epoch
        for p in processes:
            x_start = int(p['init'] * self.time_scale) + 10
            width = max(3, int(p['duration'] * self.time_scale))
            
            color = epoch_colors[p['epoch'] % len(epoch_colors)]
            color_str = f"rgb({color[0]}, {color[1]}, {color[2]})"
            
            bar = QtWidgets.QFrame(row)
            bar.setGeometry(x_start, 4, width, 20)
            bar.setStyleSheet(f"background-color: {color_str}; border-radius: 3px; border: 1px solid rgba(0,0,0,0.3);")
            
            # Epoch label inside bar
            if width > 25:
                epoch_label = QtWidgets.QLabel(f"E{p['epoch']}", bar)
                epoch_label.setGeometry(2, 2, width - 4, 16)
                epoch_label.setStyleSheet("color: white; font-size: 8px; background: transparent;")
                epoch_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            
            # Tooltip
            bar.setToolTip(f"{p['name']}\nEpoch: {p['epoch']}, Layer: {p['layer']}\n"
                          f"Start: {p['init']:.2f}s\nDuration: {p['duration']:.2f}s")
        
        # Mouse tracking for crosshair
        row.setMouseTracking(True)
        row.mouseMoveEvent = lambda e, r=row: self._on_row_mouse_move(e, r)
        row.leaveEvent = lambda e: self._on_row_mouse_leave()
        
        return row
        
    def _create_crosshair(self):
        """Create crosshair overlay widget."""
        self.crosshair_overlay = QtWidgets.QFrame(self.chart_container)
        self.crosshair_overlay.setGeometry(0, 0, 1, self.chart_container.height())
        self.crosshair_overlay.setStyleSheet("background-color: red;")
        self.crosshair_overlay.hide()
        
    def _on_row_mouse_move(self, event, row):
        """Handle mouse movement over a row."""
        # Calculate time from mouse position within the row
        x = event.pos().x()
        time_x = (x - 10) / self.time_scale  # Account for margin
        
        # Position crosshair at exact cursor X position
        if hasattr(self, 'crosshair_overlay') and self.crosshair_overlay:
            self.crosshair_overlay.setGeometry(x, 0, 1, self.chart_container.height())
            self.crosshair_overlay.show()
            self.crosshair_overlay.raise_()
        
        # Emit signal for other charts (relative time, same as resource plots)
        # Resource plots use relative time (starting from 0)
        self.sigHoverTime.emit(time_x)
        
    def _on_row_mouse_leave(self):
        """Handle mouse leaving a row."""
        if hasattr(self, 'crosshair_overlay') and self.crosshair_overlay:
            self.crosshair_overlay.hide()
        
def get_bleuscore(predicted: str, expected: str):
    """
    Calculate BLEU score between predicted and expected (reference) text.
    
    Args:
        predicted: The predicted/generated text string
        expected: The expected/reference text string
    
    Returns:
        BLEU score as a float (0.0 to 1.0)
    """
    bleu = BLEUScore()
    print(f"Predicted: {predicted}\nExpected: {expected}")
    
    # BLEUScore expects:
    # - preds: List[str] - list of predictions
    # - target: List[List[str]] - list of lists of references (multiple refs per prediction allowed)
    result = bleu([predicted], [[expected]])
    return float(result)


def minmax_downsample(x, y, target_points=2000):
    """
    Min-Max downsampling: For each bin, keep both min and max values.
    Preserves peaks and valleys which is critical for resource monitoring.
    
    Args:
        x: array-like, time values (pandas Series or numpy array)
        y: array-like, amplitude values (pandas Series or numpy array)
        target_points: desired number of output points (default 2000 for typical screen width)
    
    Returns:
        x_down, y_down: downsampled arrays (numpy arrays)
    """
    n = len(x)
    if n <= target_points:
        return x.values if hasattr(x, 'values') else x, y.values if hasattr(y, 'values') else y
    
    # Convert to numpy arrays for faster processing
    x_arr = x.values if hasattr(x, 'values') else np.array(x)
    y_arr = y.values if hasattr(y, 'values') else np.array(y)
    
    # Each bin produces 2 points (min and max)
    num_bins = max(1, target_points // 2)
    bin_size = max(2, n // num_bins)
    
    x_down = []
    y_down = []
    
    for i in range(num_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, n)
        
        if start >= n:
            break
        
        bin_x = x_arr[start:end]
        bin_y = y_arr[start:end]
        
        # Find indices of min and max
        min_idx = np.argmin(bin_y)
        max_idx = np.argmax(bin_y)
        
        # Sort by x to maintain temporal order
        if bin_x[min_idx] <= bin_x[max_idx]:
            x_down.append(bin_x[min_idx])
            y_down.append(bin_y[min_idx])
            x_down.append(bin_x[max_idx])
            y_down.append(bin_y[max_idx])
        else:
            x_down.append(bin_x[max_idx])
            y_down.append(bin_y[max_idx])
            x_down.append(bin_x[min_idx])
            y_down.append(bin_y[min_idx])
    
    return np.array(x_down), np.array(y_down)


class AttentionHeatmapWidget(pg.PlotWidget):
    def __init__(self, df: pd.DataFrame, row_tokens: list, col_tokens: list, title: str, parent=None):
        super().__init__(parent, title=title)

        self.setMinimumSize(400, 450)
        self.setMaximumHeight(450)
        self.setBackground('w')
        self.setAspectLocked(False) 

        # --- Reshape the DataFrame back into a matrix ---
        if df.empty:
            self.addItem(pg.TextItem("No Attention Data.", anchor=(0.5, 0.5), color=(255, 0, 0)))
            return

        max_row = df['row'].max() + 1
        max_col = df['column'].max() + 1
        matrix = np.zeros((max_row, max_col))

        for _, row in df.iterrows():
            matrix[int(row['row']), int(row['column'])] = row['value']

        # --- Create the ImageItem (Heatmap) ---
        img = pg.ImageItem(matrix)
        img.setRect(0, 0, max_col, max_row)
        self.addItem(img)
        
        # --- set the color map (Two shades of blue) ---
        pos = np.linspace(0.0, 1.0, 2)
        colors = np.array([
            (10, 24, 74, 255),    #  hot
            (60, 204, 153, 255)    # cold
        ])

        cmap = pg.ColorMap(pos, colors)
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        img.setLookupTable(lut)
    
        row_tokens_clean = [t.strip("<>") for t in row_tokens]
        col_tokens_clean = [t.strip("<>") for t in col_tokens]

        row_ticks = [(i + 0.5, row_tokens_clean[i]) for i in range(max_row) if i < len(row_tokens_clean)]
        col_ticks = [(i + 0.5, col_tokens_clean[i]) for i in range(max_col) if i < len(col_tokens_clean)]

        # --- AXIS CONFIGURATION ---

        self.getAxis('left').setLabel('Key (attended token)', units=None)
        self.getAxis('left').setTextPen('black')
        self.getAxis('left').setTickFont(QtGui.QFont("Arial", 8))
        self.getAxis('left').setTicks([row_ticks])

        self.getAxis('bottom').setLabel('Query (attending token)', units=None)
        self.getAxis('bottom').setTextPen('black')
      
        font = QtGui.QFont("Arial", 8)
        self.getAxis('bottom').setTickFont(font)
        self.getAxis('bottom').setTicks([col_ticks])
        

            
        vb = self.getViewBox()
        vb.setRange(xRange=(0, max_col), yRange=(0, max_row), padding=0.02)

        vb.invertY(True)
        
        vb.setLimits(xMin=0, xMax=max_col, yMin=0, yMax=max_row, minXRange=0.5, minYRange=0.5)

        # --- ColorBar ---
        bar = pg.ColorBarItem(values=(0, matrix.max()), interactive=False)
        bar.setColorMap(cmap)
        self.addItem(bar)

        bar.setImageItem(img, insert_in=self.getPlotItem())


def load_next_batch(model, val_dataloader, vocab_src, vocab_tgt, config, device):
    batch = next(iter(val_dataloader))
    encoder_input = batch["encoder_input"].to(device)
    encoder_mask = batch["encoder_mask"].to(device)
    decoder_input = batch["decoder_input"].to(device)

    encoder_input_tokens = [vocab_src.id_to_token(idx) for idx in encoder_input[0].cpu().numpy()]
    decoder_input_tokens = [vocab_tgt.id_to_token(idx) for idx in decoder_input[0].cpu().numpy()]

    assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

    # Note: greedy_decode is called later from load_model_and_generate_data (we do not call it here)
    return batch, encoder_input_tokens, decoder_input_tokens


def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def get_final_attn_map(attn_type: str, layer: int):
    """Return a single attention map by averaging all heads in a layer."""
    global model
    if attn_type == "encoder":
        attn = model.encoder.layers[layer].self_attention_block.attention_scores
    elif attn_type == "decoder":
        attn = model.decoder.layers[layer].self_attention_block.attention_scores
    elif attn_type == "encoder-decoder":
        attn = model.decoder.layers[layer].cross_attention_block.attention_scores

    return attn[0].mean(dim=0).data


def generate_attention_data(attn_type, layer, row_tokens, col_tokens, max_sentence_len):
    df = mtx2df(
        get_final_attn_map(attn_type, layer),
        max_sentence_len,
        max_sentence_len,
        row_tokens,
        col_tokens,
    )
    title = f"Layer {layer} Final {attn_type.capitalize()} Attention"
    # Return DataFrame and the original tokens for axis labeling
    return df, title, row_tokens, col_tokens

def load_model_and_generate_data(weights_path: str, gpu_sample_interval: float = 0.1):

    global ATTENTION_MODEL_LOADED, model, vocab_src, vocab_tgt, config, val_dataloader
    global SOURCE_TEXT, TARGET_TEXT, ATTN_LAYERS

    chart_data_list = []
    inference_stats = {
        "gpu_samples": [],
        "ram_samples": [],
        "predicted_text": "<prediction unavailable>",
        "bleu": 0.0,
    }

    # --- Cache directories to monitor ---
    CACHE_DIRS = [
        Path.home() / ".cache" / "torch",
        Path.home() / ".cache" / "huggingface",
        Path("/tmp"),
    ]

    # --- Load config, dataset, and tokenizers ---
    config = get_config()
    _, val_dataloader, vocab_src, vocab_tgt = get_ds(config)

    # --- Load model and weights ---
    model = get_model(config, vocab_src.get_vocab_size(), vocab_tgt.get_vocab_size()).to(device)
    epoch = get_epoch_from_file(config, weights_path)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    ATTENTION_MODEL_LOADED = True

    # --- Load next batch ---
    batch, encoder_input_tokens, decoder_output_tokens = load_next_batch(
        model, val_dataloader, vocab_src, vocab_tgt, config, device
    )

    SOURCE_TEXT = batch.get("src_text", [""])[0]
    TARGET_TEXT = batch.get("tgt_text", [""])[0]

    # Determine max length
    try:
        sentence_len_enc = encoder_input_tokens.index("[PAD]")
    except ValueError:
        sentence_len_enc = len(encoder_input_tokens)
    try:
        sentence_len_dec = decoder_output_tokens.index("[PAD]")
    except ValueError:
        sentence_len_dec = len(decoder_output_tokens)
    max_len = min(config['seq_len'], sentence_len_enc, sentence_len_dec, 20)

    # Threaded decode
    result_container = {"result": None, "exception": None}

    def run_decode():
        try:
            res = greedy_decode(
                model,
                batch["encoder_input"].to(device),
                batch.get("encoder_mask", None).to(device) if batch.get("encoder_mask", None) is not None else None,
                vocab_src, vocab_tgt, config['seq_len'], device)
            result_container["result"] = res
        except Exception as e:
            result_container["exception"] = e

    decode_thread = threading.Thread(target=run_decode, daemon=True)

    # --- Sampling loop ---
    t0 = time.time()
    decode_thread.start()
    gpu_samples = []
    ram_samples = []
    t_rel = None

    while decode_thread.is_alive():
        t_rel = time.time() - t0

        # --- GPU usage ---
        if torch.cuda.is_available():
            try:
                gpu_gb = torch.cuda.memory_allocated(device) / 1e9
            except Exception:
                gpu_gb = 0.0
        else:
            gpu_gb = 0.0

        # --- ram  usage ---
        ram_samples.append((t_rel, psutil.virtual_memory().used / 1024**3))
        gpu_samples.append((t_rel, gpu_gb))
        time.sleep(gpu_sample_interval)

  
    inference_stats["gpu_samples"] = gpu_samples
    inference_stats["ram_samples"] = ram_samples
    inference_stats["decode_time"] = t_rel

    if result_container.get("exception"):
        raise result_container["exception"]

    t0 = time.time()
    predicted_text = translate(SOURCE_TEXT, epoch)
    t_rel = time.time() - t0

    inference_stats["predicted_text"] = predicted_text
    
    tokenizer = Tokenizer.from_file("tokenizer_fr.json")
    enc = tokenizer.encode(predicted_text)
    inference_stats['TPS'] = len(enc.ids) / t_rel

    bleu_score = get_bleuscore(predicted_text, TARGET_TEXT)
    print(f"BLEU score computed: {bleu_score} (type: {type(bleu_score)})")
    inference_stats["bleu"] = bleu_score

    # --- Generate attention maps ---
    for layer in ATTN_LAYERS:
        data_enc = generate_attention_data("encoder", layer, encoder_input_tokens, encoder_input_tokens, max_len)
        chart_data_list.append(data_enc)
        data_dec = generate_attention_data("decoder", layer, decoder_output_tokens, decoder_output_tokens, max_len)
        chart_data_list.append(data_dec)
        data_cross = generate_attention_data("encoder-decoder", layer, decoder_output_tokens, encoder_input_tokens, max_len)
        chart_data_list.append(data_cross)

    return True, "Generation successful.", chart_data_list, inference_stats


class ResourceMonitorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # [ ... Initialization remains unchanged ... ]
        self.log_dir = Path("eval_results/resource_usage")
        self.file_sec = self.log_dir / "usage_seconds.csv"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.setWindowTitle("Resource Usage Monitor")
        self.resize(1400, 1000)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self.tab_pReport = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_pReport, "Performance Reports")

        self.tab_LPR = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_LPR, "Linguistic Performance Report")

        self.tab_attn = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_attn, "Performance During Inference")

        # Setup Process Mindmap tab
        self.tab_mindmap = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_mindmap, "Process Mindmap")
        self._setup_process_mindmap_tab()

        # Connect tab change signal to update placeholders when Performance Reports tab is selected
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Setup Performance Reports tab with controls and scroll area
        self._setup_performance_reports_tab()
        
        # Setup Linguistic Performance Report tab with controls
        self._setup_linguistic_report_tab()
        
        self.scroll_attn, self.scroll_layout_attn = self._make_scroll_area(self.tab_attn)

        try:
            self.dash_line = QtCore.Qt.PenStyle.DashLine
        except AttributeError:
            self.dash_line = QtCore.Qt.DashLine

        # --- Per-Second Plots (filtered by epoch range) with sampling controls ---
        self.p_cpu_sec, self.cpu_line_sec, self.cpu_sampling_combo = self._make_sampling_plot(
            "CPU Usage (%)", "b", self.per_sec_layout, "cpu")
        self.p_gpu_sec, self.gpu_line_sec, self.gpu_sampling_combo = self._make_sampling_plot(
            "GPU Usage (GB)", "g", self.per_sec_layout, "gpu")
        self.p_ram_sec, self.ram_line_sec, self.ram_sampling_combo = self._make_sampling_plot(
            "RAM Usage (GB)", "orange", self.per_sec_layout, "ram")

        # --- Average Per-Epoch Plots (always show all epochs) ---
        self.p_cpu_epoch, self.cpu_line_epoch = self._make_plot("CPU Usage (%) per Epoch", "b", self.avg_layout, symbol="o")
        self.p_gpu_epoch, self.gpu_line_epoch = self._make_plot("GPU Usage (GB) per Epoch", "g", self.avg_layout, symbol="o")
        self.p_ram_epoch, self.ram_line_epoch = self._make_plot("RAM Usage (GB) per Epoch", "orange", self.avg_layout, symbol="o")

        self.epoch_lines = []
        self._cached_df = None  # Cache for loaded data

        df = pd.DataFrame()
        epochs = pd.Series()
        try:
            with open("eval_results/eval_metrics.json", "r") as f:
                data = json.load(f)
            df = pd.DataFrame([
                {
                    "epoch": entry["epoch"], 
                    "cer": entry["cer"], 
                    "wer": entry["wer"], 
                    "bleu": entry["bleu"],
                    "loss": entry.get("loss", 0.0)  # Loss may not exist in older data
                }
                for entry in data
            ])
            df.rename(columns={"cer": "CER", "wer": "WER", "bleu": "BLEU", "loss": "Loss"}, inplace=True)
            epochs = df["epoch"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        self.plot_linguistic, _ = self._make_plot("Translation Metrics Over Epochs", "k", self.scroll_layout_LPR)

        self.line_cer = self.plot_linguistic.plot(pen=pg.mkPen('r', width=2), symbol='o', symbolBrush='r')
        self.line_wer = self.plot_linguistic.plot(pen=pg.mkPen('b', width=2), symbol='o', symbolBrush='b')
        self.line_bleu = self.plot_linguistic.plot(pen=pg.mkPen('g', width=2), symbol='o', symbolBrush='g')
        self.line_loss = self.plot_linguistic.plot(pen=pg.mkPen('purple', width=2), symbol='o', symbolBrush='purple')

        legend = self.plot_linguistic.addLegend()

        legend.addItem(self.line_cer, "CER")
        legend.addItem(self.line_wer, "WER")
        legend.addItem(self.line_bleu, "BLEU")
        legend.addItem(self.line_loss, "Loss")

        if not df.empty:
            self.line_cer.setData(epochs, df["CER"])
            self.line_wer.setData(epochs, df["WER"])
            self.line_bleu.setData(epochs, df["BLEU"])
            if "Loss" in df.columns:
                self.line_loss.setData(epochs, df["Loss"])

        self.weights_path = ""
        self.attn_content_widgets = []
        self._setup_attention_tab_controls()

        self.infer_gpu_samples = []
        self.infer_storage_gb = 0.0

        # Timer for live mode updates - don't start yet
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_pReport_plots)
        # Timer only runs in LIVE MODE

        # Initialize plots with empty data (user must select range first)
        self._clear_per_second_plots()
        
        # Update epoch input placeholders on startup (if data exists)
        self._update_epoch_input_placeholders()
        
        # Set up a timer to periodically check for data file updates
        self.placeholder_timer = QtCore.QTimer()
        self.placeholder_timer.timeout.connect(self._update_epoch_input_placeholders)
        self.placeholder_timer.start(5000)  # Check every 5 seconds
        
    def _toggle_gantt_chart(self, checked):
        """Toggle the visibility of the Gantt chart."""
        self._gantt_visible = checked
        if checked:
            self.gantt_container.show()
            # Setup crosshair lines for plots
            self._setup_crosshair_lines()
            # Load processes data
            self._load_gantt_data()
            # Adjust splitter to give Gantt chart reasonable space (bottom)
            total_height = self.main_splitter.height()
            plots_height = int(total_height * 0.6)
            gantt_height = int(total_height * 0.4)
            self.main_splitter.setSizes([plots_height, gantt_height])
        else:
            self.gantt_container.hide()
            self.main_splitter.setSizes([self.main_splitter.height(), 0])
            # Hide crosshair lines
            for plot, line in self._crosshair_lines:
                line.hide()
            
    def _get_process_min_time(self):
        """Get the earliest process init time from processes.json for time synchronization."""
        json_path = Path("eval_results/processes.json")
        if not json_path.exists():
            return None
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            processes = data.get("processes", {})
            if not processes:
                return None
            
            # Find earliest process init time
            min_time = None
            for proc in processes.values():
                timeline = proc.get('timeline', {})
                init_time = timeline.get('initialized', 0)
                if init_time > 0:
                    if min_time is None or init_time < min_time:
                        min_time = init_time
            
            return min_time
        except Exception:
            return None
    
    def _load_gantt_data(self):
        """Load process data from JSON file and populate Gantt chart."""
        json_path = Path("eval_results/processes.json")
        if not json_path.exists():
            self.gantt_coord_label.setText("No process data available. Run training first.")
            return
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            processes = data.get("processes", {})
            if not processes:
                self.gantt_coord_label.setText("No process data available.")
                return
            
            # Apply current epoch range to Gantt chart
            start_epoch, end_epoch = self._epoch_range
            self.gantt_chart.set_epoch_range(start_epoch, end_epoch)
            
            # Load processes into Gantt chart
            self.gantt_chart.load_processes(processes)
            
            # Count displayed processes
            displayed = len(self.gantt_chart.process_labels)
            total = len(processes)
            
            if start_epoch is not None or end_epoch is not None:
                self.gantt_coord_label.setText(f"Showing {displayed}/{total} processes (filtered) | Epochs {start_epoch or 'start'}-{end_epoch or 'end'}")
            else:
                self.gantt_coord_label.setText(f"Loaded {displayed} processes | Scroll to view all")
            
        except Exception as e:
            self.gantt_coord_label.setText(f"Error loading process data: {str(e)}")
    
    def _refresh_gantt_chart(self):
        """Refresh the Gantt chart by reloading process data."""
        if not self._gantt_visible:
            return
        
        # Reload the data
        self._load_gantt_data()
        
        # Update status to show refresh happened
        current_text = self.gantt_coord_label.text()
        if "Refreshed" not in current_text:
            self.gantt_coord_label.setText(f"{current_text} | Refreshed: {QtCore.QTime.currentTime().toString('hh:mm:ss')}")
    
    def _on_gantt_auto_refresh_changed(self, state):
        """Handle auto-refresh checkbox toggle."""
        if state == QtCore.Qt.CheckState.Checked.value:
            # Start auto-refresh timer (5 seconds)
            self.gantt_refresh_timer.start(5000)
        else:
            # Stop timer
            self.gantt_refresh_timer.stop()
            
    def _on_gantt_hover(self, time_x):
        """Handle hover over Gantt chart - update coordinate label and sync crosshair."""
        if not self._gantt_visible:
            return
            
        # time_x is relative time (starting from 0), same as resource plots
        rel_time = time_x
        
        # Update coordinate label
        self.gantt_coord_label.setText(f"Time: {rel_time:.2f}s")
        
        # Update crosshair lines on per-second plots (they use relative time)
        for plot, line in self._crosshair_lines:
            line.setPos(rel_time)
            line.show()
            
    def _setup_crosshair_lines(self):
        """Setup crosshair lines for all per-second plots."""
        # Clear existing
        for plot, line in self._crosshair_lines:
            plot.removeItem(line)
        self._crosshair_lines = []
        
        # Create new crosshair lines
        for plot in [self.p_cpu_sec, self.p_gpu_sec, self.p_ram_sec]:
            line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('red', width=1, style=QtCore.Qt.PenStyle.DashLine))
            line.hide()
            plot.addItem(line)
            self._crosshair_lines.append((plot, line))
            
    def _on_plots_scroll_changed(self, value):
        """Handle scroll of plots area - could sync back to Gantt if needed."""
        # This is called when the scroll area scrolls vertically
        # Horizontal scrolling of plots is handled by the plot's view box
        pass

    def _setup_attention_tab_controls(self):

        control_frame = QtWidgets.QFrame()
        control_layout = QtWidgets.QVBoxLayout(control_frame)
        control_frame.setFrameShape(QtWidgets.QFrame.Shape.Box)
        control_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.scroll_layout_attn.addWidget(control_frame)

        # --- File selection ---
        file_layout = QtWidgets.QHBoxLayout()
        self.path_label = QtWidgets.QLabel("Weights Path: No file selected")
        self.select_button = QtWidgets.QPushButton("Select Weights File (.pt)")
        self.select_button.clicked.connect(self._select_weights_file)
        file_layout.addWidget(self.path_label)
        file_layout.addWidget(self.select_button)
        control_layout.addLayout(file_layout)

        # --- Submit button for generation ---
        self.submit_button = QtWidgets.QPushButton("Generate Attention Maps")
        self.submit_button.clicked.connect(self._generate_attention)
        self.submit_button.setEnabled(False)  # Disabled until file is selected
        control_layout.addWidget(self.submit_button)

        self.status_label = QtWidgets.QLabel("Ready. Select weights file.")
        control_layout.addWidget(self.status_label)

        # --- Inference summary (source, target, prediction, BLEU) ---
        self.infer_summary_widget = QtWidgets.QWidget()
        infer_sum_layout = QtWidgets.QVBoxLayout(self.infer_summary_widget)
        self.src_label = QtWidgets.QLabel("<b>Source:</b> No data loaded.")
        self.tgt_label = QtWidgets.QLabel("<b>Target:</b> No data loaded.")
        self.decoding_time = QtWidgets.QLabel("<b>Decoding Time of Source to Target:</b> No data loaded.")
        self.pred_label = QtWidgets.QLabel("<b>Predicted:</b> <prediction unavailable>")
        self.pred_time = QtWidgets.QLabel("<b>Generation of Tokens Per Second:</b> No data loaded.")
        self.bleu_label = QtWidgets.QLabel("<b>BLEU:</b> N/A")

        # --- Add summart widgets one by one ---
        for lbl in (self.src_label, self.tgt_label, self.decoding_time, self.pred_label, self.pred_time, self.bleu_label):
            lbl.setWordWrap(True)
            infer_sum_layout.addWidget(lbl)
        control_layout.addWidget(self.infer_summary_widget)

        # --- Small inference plots ---
        plots_frame = QtWidgets.QFrame()
        plots_layout = QtWidgets.QHBoxLayout(plots_frame)
        plots_frame.setMinimumHeight(180)

            # GPU inference plot
        self.p_gpu_infer = pg.PlotWidget(title="GPU Memory During Inference (GB)")
        self.p_gpu_infer.setBackground("w")
        self.p_gpu_infer.showGrid(x=True, y=True, alpha=0.3)
        self.p_gpu_infer.getAxis("left").setLabel("Memory (GB)")
        self.p_gpu_infer.getAxis("bottom").setLabel("Sample")
        self.gpu_infer_line = self.p_gpu_infer.plot(pen=pg.mkPen('g', width=2), symbol='o', symbolBrush='g')
        plots_layout.addWidget(self.p_gpu_infer)

            # Storage plot
        self.p_storage_infer = pg.PlotWidget(title="RAM Memory During Inference( GB)")
        self.p_storage_infer.setBackground("w")
        self.p_storage_infer.showGrid(x=True, y=True, alpha=0.3)
        self.p_storage_infer.getAxis("left").setLabel("Memory (GB)")
        self.p_storage_infer.getAxis("bottom").setLabel("Sample")
        self.storage_line = self.p_storage_infer.plot(pen=pg.mkPen('b', width=2), symbol='o', symbolBrush='b')
        plots_layout.addWidget(self.p_storage_infer)

        control_layout.addWidget(plots_frame)

        # --- Print to PDF Button ---
        print_layout = QtWidgets.QHBoxLayout()
        print_layout.addStretch(1)
        self.print_attn_pdf_btn = QtWidgets.QPushButton("🖨️ Print to PDF")
        self.print_attn_pdf_btn.setStyleSheet("QPushButton { padding: 5px 15px; background-color: #FF9800; color: white; border-radius: 3px; } QPushButton:hover { background-color: #F57C00; }")
        self.print_attn_pdf_btn.clicked.connect(self._print_attention_report_to_pdf)
        print_layout.addWidget(self.print_attn_pdf_btn)
        control_layout.addLayout(print_layout)

        self.scroll_layout_attn.addStretch(1)

    def _select_weights_file(self):
        try:
            print("Opening file dialog...")
            # Use DontUseNativeDialog option if native dialog has issues
            options = QtWidgets.QFileDialog.Option.DontUseNativeDialog
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 
                "Select Model Weights File", 
                str(Path.cwd()), 
                "PyTorch Weights (*.pt);;All Files (*)",
                options=options
            )
            print(f"File dialog returned: {file_name}")
            if file_name:
                self.weights_path = file_name
                self.path_label.setText(f"Weights Path: {self.weights_path}")
                self.status_label.setText(f"File selected: {Path(file_name).name}. Click 'Generate Attention Maps'.")
                self.submit_button.setEnabled(True)
                self._clear_attention_widgets()
        except Exception as e:
            print(f"Error in file selection: {e}")
            import traceback
            traceback.print_exc()

    def _clear_attention_widgets(self):
        for widget in self.attn_content_widgets:
            widget.setParent(None)
            if isinstance(widget, AttentionHeatmapWidget):
                widget.close()
                widget.deleteLater()
        self.attn_content_widgets = []

        global SOURCE_TEXT, TARGET_TEXT
        SOURCE_TEXT = "No data loaded."
        TARGET_TEXT = "No data loaded."

        # reset infer summary and plots
        self.src_label.setText("<b>Source:</b> No data loaded.")
        self.tgt_label.setText("<b>Target:</b> No data loaded.")
        self.decoding_time.setText("<b>Decoding Time of Source to Target:</b> No data loaded.")
        self.pred_label.setText("<b>Predicted:</b> <prediction unavailable>")
        self.pred_time.setText("<b>Generation of Tokens Per Second:</b> No data loaded")
        self.bleu_label.setText("<b>BLEU:</b> N/A")
        self.gpu_infer_line.setData([], [])
        self.storage_line.setData([], [])

    def _generate_attention(self):
        if not self.weights_path:
            self.status_label.setText("Error: No model weights file selected.")
            return

        self._clear_attention_widgets()
        self.status_label.setText("Loading model and generating maps from a random validation sample... This may take a moment.")
        QtWidgets.QApplication.processEvents()

        # Load model and get data: returns (success, message, chart_data_list, inference_stats)
        success, message, all_chart_data, inference_stats = load_model_and_generate_data(self.weights_path, gpu_sample_interval=0.1)

        if success:
            self.status_label.setText(f"Success: {message}")

            self._display_attention_maps(all_chart_data, inference_stats)
        else:
            self.status_label.setText(f"Failure: {message}")

            # still attempt to display any partial charts if present
            self._display_attention_maps(all_chart_data, inference_stats)

    def _display_attention_maps(self, chart_data_list: list, inference_stats: dict = None):
        global ATTENTION_MODEL_LOADED, ATTN_LAYERS
        if not ATTENTION_MODEL_LOADED:
            return

        if inference_stats is None:
            inference_stats = {"gpu_samples": [], "ram_samples": [], "storage_gb": 0.0, "predicted_text": "<prediction unavailable>", "bleu": 0.0, "decode_time": 0.0, "TPS": 0.0}
        
        # --- Display stats ---
        text_widget = QtWidgets.QWidget()
        text_layout = QtWidgets.QHBoxLayout(text_widget)
        src_label = QtWidgets.QLabel(f"<b>Source:</b> {SOURCE_TEXT}")
        tgt_label = QtWidgets.QLabel(f"<b>Target:</b> {TARGET_TEXT}")
        src_label.setWordWrap(True)
        tgt_label.setWordWrap(True)
        text_layout.addWidget(src_label, 1)
        text_layout.addWidget(tgt_label, 1)

        insert_index = 1
        self.scroll_layout_attn.insertWidget(insert_index, text_widget)
        self.attn_content_widgets.append(text_widget)
        insert_index += 1

        self.src_label.setText(f"<b>Source:</b> {SOURCE_TEXT}")
        self.tgt_label.setText(f"<b>Target:</b> {TARGET_TEXT}")
        predicted_text = inference_stats.get("predicted_text", "<prediction unavailable>")
        bleu_val = inference_stats.get("bleu", 0.0)
        print(f"Displaying BLEU: {bleu_val} (type: {type(bleu_val)})")
        self.pred_label.setText(f"<b>Predicted:</b> {predicted_text}")
        self.bleu_label.setText(f"<b>BLEU:</b> {bleu_val:.4f}")
        self.decoding_time.setText(f"<b>Decoding Time:</b> {inference_stats.get('decode_time', 0.0):.3f}")
        self.pred_time.setText(f"<b>Generation of Tokens Per Second:</b> {inference_stats.get('TPS', 0.0):.3f}")

        # --- Plot GPU samples (time vs GB) ---
        gpu_samples = inference_stats.get("gpu_samples", [])
        if gpu_samples:
            times = [t for (t, g) in gpu_samples]
            values = [g for (t, g) in gpu_samples]
            self.gpu_infer_line.setData(times, values)
        else:
            self.gpu_infer_line.setData([], [])

        # --- Plot RAM samples (time vs GB) ---
        ram_samples = inference_stats.get("ram_samples", [])
        if ram_samples:
            st_times = [t for (t, s) in ram_samples]
            st_vals = [s for (t, s) in ram_samples]
            self.storage_line.setData(st_times, st_vals)
        else:
            self.storage_line.setData([], [])



        chart_titles = [
            "Attention Heads",
            "",
            ""
            ]

        # --- Display Charts (6 charts per section) ---
        chart_index = 0

        for attn_type_index, title in enumerate(chart_titles):

            title_widget = QtWidgets.QLabel(f"<h3>{title}</h3>")
            self.scroll_layout_attn.insertWidget(insert_index, title_widget)
            self.attn_content_widgets.append(title_widget)
            insert_index += 1

            grid = QtWidgets.QGridLayout()
            grid_widget = QtWidgets.QWidget()
            grid_widget.setLayout(grid)

            for idx, layer in enumerate(ATTN_LAYERS):
                row = idx // 3
                col = idx % 3

                if chart_index >= len(chart_data_list):
                    # in case of partial data
                    df = pd.DataFrame(columns=["row", "column", "value", "row_token", "col_token"])
                    row_tokens = []
                    col_tokens = []
                    chart_title = f"Layer {layer} {title}"
                else:
                    data_tuple = chart_data_list[chart_index]
                    df, chart_title, row_tokens, col_tokens = data_tuple

                chart_widget = AttentionHeatmapWidget(df, row_tokens, col_tokens, chart_title)
                grid.addWidget(chart_widget, row, col)
                self.attn_content_widgets.append(chart_widget) 
                chart_index += 1

            self.scroll_layout_attn.insertWidget(insert_index, grid_widget)
            self.attn_content_widgets.append(grid_widget)
            insert_index += 1

        self.scroll_layout_attn.addStretch(1)


    def _make_scroll_area(self, parent):
        scroll = QtWidgets.QScrollArea(parent)
        scroll.setWidgetResizable(True)
        layout_parent = QtWidgets.QVBoxLayout(parent)
        layout_parent.addWidget(scroll)

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setSpacing(25)
        layout.setContentsMargins(20, 20, 20, 20)
        scroll.setWidget(content)

        return scroll, layout

    def _setup_performance_reports_tab(self):
        """Setup the Performance Reports tab with epoch range controls, Gantt chart, and separate sections."""
        # Main layout for the tab
        tab_layout = QtWidgets.QVBoxLayout(self.tab_pReport)
        tab_layout.setSpacing(10)
        tab_layout.setContentsMargins(10, 10, 10, 10)

        # --- Control Panel: Epoch Range Selection ---
        control_frame = QtWidgets.QFrame()
        control_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        control_frame.setStyleSheet("QFrame { background-color: #f0f0f0; border: 1px solid #cccccc; border-radius: 5px; }")
        control_layout = QtWidgets.QHBoxLayout(control_frame)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(15, 10, 15, 10)

        # Title label
        title_label = QtWidgets.QLabel("<b>Per-Second Resource Usage View</b>")
        title_label.setStyleSheet("font-size: 12px; color: #333333;")
        control_layout.addWidget(title_label)

        control_layout.addSpacing(20)

        # --- Live Mode Toggle ---
        self.live_mode_checkbox = QtWidgets.QCheckBox("🔴 LIVE MODE")
        self.live_mode_checkbox.setStyleSheet("QCheckBox { font-weight: bold; color: #d32f2f; font-size: 11px; } QCheckBox::indicator { width: 18px; height: 18px; }")
        self.live_mode_checkbox.setToolTip("Show only the most recent data and auto-update during training")
        self.live_mode_checkbox.stateChanged.connect(self._on_live_mode_changed)
        control_layout.addWidget(self.live_mode_checkbox)

        control_layout.addSpacing(15)

        # Live window size selector (seconds or epochs)
        live_window_label = QtWidgets.QLabel("Window:")
        control_layout.addWidget(live_window_label)
        
        self.live_window_combo = QtWidgets.QComboBox()
        self.live_window_combo.addItem("Last 60 sec", 60)
        self.live_window_combo.addItem("Last 5 min", 300)
        self.live_window_combo.addItem("Last 15 min", 900)
        self.live_window_combo.addItem("Last 1 epoch", -1)  # Special: last complete epoch
        self.live_window_combo.addItem("Last 3 epochs", -3)
        self.live_window_combo.setMaximumWidth(120)
        self.live_window_combo.setToolTip("How much recent data to display in live mode")
        self.live_window_combo.setEnabled(False)  # Disabled until live mode is on
        self.live_window_combo.currentIndexChanged.connect(self._on_live_window_changed)
        control_layout.addWidget(self.live_window_combo)

        control_layout.addSpacing(20)

        # Start Epoch
        start_label = QtWidgets.QLabel("Start Epoch:")
        control_layout.addWidget(start_label)
        self.epoch_start_input = QtWidgets.QLineEdit()
        self.epoch_start_input.setPlaceholderText("1")
        self.epoch_start_input.setMaximumWidth(80)
        self.epoch_start_input.setToolTip("Enter the starting epoch number to display")
        control_layout.addWidget(self.epoch_start_input)

        control_layout.addSpacing(10)

        # End Epoch
        end_label = QtWidgets.QLabel("End Epoch:")
        control_layout.addWidget(end_label)
        self.epoch_end_input = QtWidgets.QLineEdit()
        self.epoch_end_input.setPlaceholderText("max")
        self.epoch_end_input.setMaximumWidth(80)
        self.epoch_end_input.setToolTip("Enter the ending epoch number to display (leave empty for all)")
        control_layout.addWidget(self.epoch_end_input)

        control_layout.addSpacing(10)

        # Apply Button
        self.apply_range_btn = QtWidgets.QPushButton("Apply Range")
        self.apply_range_btn.setStyleSheet("QPushButton { padding: 5px 15px; background-color: #4CAF50; color: white; border-radius: 3px; } QPushButton:hover { background-color: #45a049; }")
        self.apply_range_btn.clicked.connect(self._apply_epoch_range)
        control_layout.addWidget(self.apply_range_btn)

        control_layout.addSpacing(10)

        # Reset Button
        self.reset_range_btn = QtWidgets.QPushButton("Show All")
        self.reset_range_btn.setStyleSheet("QPushButton { padding: 5px 15px; background-color: #2196F3; color: white; border-radius: 3px; } QPushButton:hover { background-color: #0b7dda; }")
        self.reset_range_btn.clicked.connect(self._reset_epoch_range)
        control_layout.addWidget(self.reset_range_btn)

        control_layout.addSpacing(10)

        # Gantt Chart Toggle Button
        self.gantt_toggle_btn = QtWidgets.QPushButton("📊 Toggle Gantt Chart")
        self.gantt_toggle_btn.setStyleSheet("QPushButton { padding: 5px 15px; background-color: #9C27B0; color: white; border-radius: 3px; } QPushButton:hover { background-color: #7B1FA2; }")
        self.gantt_toggle_btn.setCheckable(True)
        self.gantt_toggle_btn.clicked.connect(self._toggle_gantt_chart)
        control_layout.addWidget(self.gantt_toggle_btn)

        control_layout.addSpacing(10)

        # Print to PDF Button
        self.print_pdf_btn = QtWidgets.QPushButton("🖨️ Print to PDF")
        self.print_pdf_btn.setStyleSheet("QPushButton { padding: 5px 15px; background-color: #FF9800; color: white; border-radius: 3px; } QPushButton:hover { background-color: #F57C00; }")
        self.print_pdf_btn.clicked.connect(self._print_performance_report_to_pdf)
        control_layout.addWidget(self.print_pdf_btn)

        control_layout.addStretch(1)

        # Status label showing current range
        self.range_status_label = QtWidgets.QLabel("⚠️ Select a data range or enable LIVE MODE")
        self.range_status_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
        control_layout.addWidget(self.range_status_label)

        tab_layout.addWidget(control_frame)

        # --- Horizontal Splitter: Resource Plots (top) + Gantt Chart (bottom) ---
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        tab_layout.addWidget(self.main_splitter, stretch=1)

        # --- TOP: Scroll Area for Resource Plots ---
        self.scroll_pReport = QtWidgets.QScrollArea()
        self.scroll_pReport.setWidgetResizable(True)
        self.scroll_pReport.horizontalScrollBar().valueChanged.connect(self._on_plots_scroll_changed)
        
        # Content widget for scroll area
        content = QtWidgets.QWidget()
        self.scroll_layout_pReport = QtWidgets.QVBoxLayout(content)
        self.scroll_layout_pReport.setSpacing(30)
        self.scroll_layout_pReport.setContentsMargins(20, 20, 20, 20)
        self.scroll_pReport.setWidget(content)
        self.main_splitter.addWidget(self.scroll_pReport)

        # --- BOTTOM: Gantt Chart Area ---
        self.gantt_container = QtWidgets.QWidget()
        self.gantt_container.setMinimumHeight(150)  # Minimum height for resizing
        gantt_layout = QtWidgets.QVBoxLayout(self.gantt_container)
        gantt_layout.setContentsMargins(0, 0, 0, 0)
        
        # Gantt chart widget
        self.gantt_chart = GanttChartWidget()
        self.gantt_chart.sigHoverTime.connect(self._on_gantt_hover)
        gantt_layout.addWidget(self.gantt_chart)
        
        # Crosshair lines for per-second plots (sync with Gantt hover)
        self._crosshair_lines = []
        
        # Gantt chart control bar (refresh button + coord label)
        gantt_control_layout = QtWidgets.QHBoxLayout()
        
        # Refresh button
        self.gantt_refresh_btn = QtWidgets.QPushButton("🔄 Refresh")
        self.gantt_refresh_btn.setStyleSheet("QPushButton { padding: 3px 10px; font-size: 10px; background-color: #2196F3; color: white; border-radius: 3px; } QPushButton:hover { background-color: #1976D2; }")
        self.gantt_refresh_btn.setToolTip("Reload process data from file")
        self.gantt_refresh_btn.clicked.connect(self._refresh_gantt_chart)
        gantt_control_layout.addWidget(self.gantt_refresh_btn)
        
        # Auto-refresh checkbox
        self.gantt_auto_refresh_checkbox = QtWidgets.QCheckBox("Auto")
        self.gantt_auto_refresh_checkbox.setStyleSheet("font-size: 10px;")
        self.gantt_auto_refresh_checkbox.setToolTip("Auto-refresh every 5 seconds when visible")
        self.gantt_auto_refresh_checkbox.stateChanged.connect(self._on_gantt_auto_refresh_changed)
        gantt_control_layout.addWidget(self.gantt_auto_refresh_checkbox)
        
        gantt_control_layout.addStretch(1)
        
        # Coordinate label at bottom left of Gantt
        self.gantt_coord_label = QtWidgets.QLabel("Time: -- | Process: --")
        self.gantt_coord_label.setStyleSheet("font-size: 10px; color: #666; padding: 2px 5px;")
        gantt_control_layout.addWidget(self.gantt_coord_label)
        
        gantt_layout.addLayout(gantt_control_layout)
        
        # Initially hide Gantt chart
        self.gantt_container.hide()
        self.main_splitter.addWidget(self.gantt_container)
        
        # Timer for Gantt chart auto-refresh
        self.gantt_refresh_timer = QtCore.QTimer()
        self.gantt_refresh_timer.timeout.connect(self._refresh_gantt_chart)

        # Set initial splitter sizes (plots on top, Gantt hidden)
        self.main_splitter.setSizes([800, 0])
        
        # Set stretch factors so both widgets can be resized
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 1)

        # --- Section 1: Per-Second Resource Usage ---
        per_sec_frame = QtWidgets.QGroupBox("Per-Second Resource Usage (High Resolution)")
        per_sec_frame.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; color: #333333; border: 2px solid #4CAF50; border-radius: 5px; margin-top: 10px; padding-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        self.per_sec_layout = QtWidgets.QVBoxLayout(per_sec_frame)
        self.per_sec_layout.setSpacing(20)
        self.per_sec_layout.setContentsMargins(15, 15, 15, 15)
        self.scroll_layout_pReport.addWidget(per_sec_frame)

        # --- Section 2: Average Per-Epoch Resource Usage ---
        avg_frame = QtWidgets.QGroupBox("Average Resource Usage Per Epoch")
        avg_frame.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; color: #333333; border: 2px solid #2196F3; border-radius: 5px; margin-top: 10px; padding-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        self.avg_layout = QtWidgets.QVBoxLayout(avg_frame)
        self.avg_layout.setSpacing(20)
        self.avg_layout.setContentsMargins(15, 15, 15, 15)
        self.scroll_layout_pReport.addWidget(avg_frame)

        # Store current epoch range (None means show all)
        self._epoch_range = (None, None)  # (start, end)
        self._live_mode = False  # Live mode flag
        self._data_loaded = False  # Flag to track if user has made a selection
        self._gantt_visible = False

    def _setup_linguistic_report_tab(self):
        """Setup the Linguistic Performance Report tab with print button."""
        # Main layout for the tab
        tab_layout = QtWidgets.QVBoxLayout(self.tab_LPR)
        tab_layout.setSpacing(10)
        tab_layout.setContentsMargins(10, 10, 10, 10)

        # --- Control Panel ---
        control_frame = QtWidgets.QFrame()
        control_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        control_frame.setStyleSheet("QFrame { background-color: #f0f0f0; border: 1px solid #cccccc; border-radius: 5px; }")
        control_layout = QtWidgets.QHBoxLayout(control_frame)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(15, 10, 15, 10)

        # Title label
        title_label = QtWidgets.QLabel("<b>Linguistic Performance Report</b>")
        title_label.setStyleSheet("font-size: 12px; color: #333333;")
        control_layout.addWidget(title_label)

        control_layout.addStretch(1)

        # Print to PDF Button
        self.print_lpr_pdf_btn = QtWidgets.QPushButton("🖨️ Print to PDF")
        self.print_lpr_pdf_btn.setStyleSheet("QPushButton { padding: 5px 15px; background-color: #FF9800; color: white; border-radius: 3px; } QPushButton:hover { background-color: #F57C00; }")
        self.print_lpr_pdf_btn.clicked.connect(self._print_linguistic_report_to_pdf)
        control_layout.addWidget(self.print_lpr_pdf_btn)

        tab_layout.addWidget(control_frame)

        # --- Scroll Area for Plot ---
        self.scroll_LPR = QtWidgets.QScrollArea()
        self.scroll_LPR.setWidgetResizable(True)
        tab_layout.addWidget(self.scroll_LPR)

        # Content widget for scroll area
        content = QtWidgets.QWidget()
        self.scroll_layout_LPR = QtWidgets.QVBoxLayout(content)
        self.scroll_layout_LPR.setSpacing(30)
        self.scroll_layout_LPR.setContentsMargins(20, 20, 20, 20)
        self.scroll_LPR.setWidget(content)

    def _on_tab_changed(self, index):
        """Handle tab selection change."""
        # If Performance Reports tab is selected (index 0), update placeholders
        if index == 0:
            self._update_epoch_input_placeholders()
        # If Process Mindmap tab is selected (index 3), auto-refresh if data not loaded
        elif index == 3:
            if self.mindmap_data is None:
                self._load_and_draw_mindmap()

    def _update_epoch_input_placeholders(self):
        """Update input placeholders to show available epoch range."""
        try:
            if not self.file_sec.exists():
                self.epoch_start_input.setPlaceholderText("1")
                self.epoch_end_input.setPlaceholderText("max")
                return
            
            # Read just the epoch_marker column to find range
            df = pd.read_csv(self.file_sec, usecols=["epoch_marker"])
            
            if "epoch_marker" in df.columns:
                epochs = df["epoch_marker"].dropna().apply(
                    lambda x: int(x.split("_")[1]) if isinstance(x, str) and x.startswith("epoch_") else None
                ).dropna()
                
                
                if not epochs.empty:
                    min_epoch = int(epochs.min())
                    max_epoch = int(epochs.max())
                    self.epoch_start_input.setPlaceholderText(str(min_epoch))
                    self.epoch_end_input.setPlaceholderText(str(max_epoch))
                else:
                    self.epoch_start_input.setPlaceholderText("1")
                    self.epoch_end_input.setPlaceholderText("max")
            else:
                self.epoch_start_input.setPlaceholderText("1")
                self.epoch_end_input.setPlaceholderText("max")
        except Exception:
            # If anything fails, use defaults
            self.epoch_start_input.setPlaceholderText("1")
            self.epoch_end_input.setPlaceholderText("max")

    def _apply_epoch_range(self):
        """Apply the epoch range filter from user input."""
        try:
            start_text = self.epoch_start_input.text().strip()
            end_text = self.epoch_end_input.text().strip()

            start_epoch = int(start_text) if start_text else None
            end_epoch = int(end_text) if end_text else None

            self._epoch_range = (start_epoch, end_epoch)
            
            # Disable live mode (but don't trigger the signal handler)
            self._live_mode = False
            self.live_mode_checkbox.blockSignals(True)
            self.live_mode_checkbox.setChecked(False)
            self.live_mode_checkbox.blockSignals(False)
            self.live_window_combo.setEnabled(False)
            
            # Stop timer if running
            if self.timer.isActive():
                self.timer.stop()
            
            # Set data loaded flag
            self._data_loaded = True

            # Update status label
            if not self.file_sec.exists():
                self.range_status_label.setText("⚠️ No data file found. Start training to generate data.")
                QtWidgets.QMessageBox.information(self, "No Data", "No training data file found.\nStart training to generate data, or select a different file.")
                return
            elif start_epoch is None and end_epoch is None:
                self.range_status_label.setText("Showing: All epochs (Static)")
            elif start_epoch is not None and end_epoch is not None:
                self.range_status_label.setText(f"Showing: Epochs {start_epoch} - {end_epoch} (Static)")
            elif start_epoch is not None:
                self.range_status_label.setText(f"Showing: Epochs {start_epoch} - end (Static)")
            else:
                self.range_status_label.setText(f"Showing: Epochs 1 - {end_epoch} (Static)")

            # Clear cache to force reload with new filter
            self._cached_df = None
            self._update_plots_once()
            
            # Reload Gantt chart with new range if visible
            if self._gantt_visible:
                self._load_gantt_data()

        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid integer epoch numbers.")

    def _reset_epoch_range(self):
        """Reset to show all epochs."""
        self.epoch_start_input.clear()
        self.epoch_end_input.clear()
        self._epoch_range = (None, None)

        # Disable live mode (but don't trigger the signal handler)
        self._live_mode = False
        self.live_mode_checkbox.blockSignals(True)
        self.live_mode_checkbox.setChecked(False)
        self.live_mode_checkbox.blockSignals(False)
        self.live_window_combo.setEnabled(False)
        
        # Stop timer if running
        if self.timer.isActive():
            self.timer.stop()
        
        # Set data loaded flag
        self._data_loaded = True

        if not self.file_sec.exists():
            self.range_status_label.setText("⚠️ No data file found. Start training to generate data.")
        else:
            self.range_status_label.setText("Showing: All epochs (Static)")
        
        self._cached_df = None
        self._update_plots_once()
        
        # Reload Gantt chart with all epochs if visible
        if self._gantt_visible:
            self._load_gantt_data()

    def _on_live_mode_changed(self, state):
        """Handle live mode toggle."""
        self._live_mode = (state == QtCore.Qt.CheckState.Checked.value)
        self.live_window_combo.setEnabled(self._live_mode)

        if self._live_mode:
            self._data_loaded = True
            # Clear manual epoch inputs when entering live mode
            self.epoch_start_input.clear()
            self.epoch_end_input.clear()
            self._epoch_range = (None, None)
            
            # Start the timer for live updates
            if not self.timer.isActive():
                self.timer.start(1000)
            
            self._cached_df = None
            self.update_pReport_plots()
        else:
            # Stop auto-updates when leaving live mode
            if self.timer.isActive():
                self.timer.stop()
            
            # When manually unchecked (not via Apply Range), clear data
            # The Apply Range and Show All buttons handle their own state
            self.range_status_label.setText("⚠️ Select a data range to view")
            self._data_loaded = False
            self._clear_per_second_plots()

    def _update_plots_once(self):
        """Update plots a single time (for static/manual range viewing)."""
        # Stop timer if running (manual range = static view)
        if self.timer.isActive():
            self.timer.stop()
        
        if not self.file_sec.exists():
            self._clear_per_second_plots()
            return
        
        self.update_pReport_plots()

    def _on_live_window_changed(self):
        """Handle live window size change."""
        if self._live_mode:
            self._cached_df = None
            self.update_pReport_plots()

    def _get_live_window_range(self, df):
        """Calculate the epoch/time range for live mode based on window selection."""
        window_value = self.live_window_combo.currentData()
        
        if window_value is None or df.empty:
            return df
        
        max_time = df["time"].max()
        
        if window_value > 0:
            # Time-based window (in seconds)
            min_time = max(0, max_time - window_value)
            return df[df["time"] >= min_time]
        else:
            # Epoch-based window
            epochs = df["epoch"].dropna().unique()
            if len(epochs) == 0:
                return df
            
            num_epochs = abs(window_value)
            sorted_epochs = sorted(epochs)
            start_epoch = sorted_epochs[-num_epochs] if len(sorted_epochs) >= num_epochs else sorted_epochs[0]
            return df[df["epoch"] >= start_epoch]

    def _clear_per_second_plots(self):
        """Clear all per-second plot data."""
        self.cpu_line_sec.setData([], [])
        self.gpu_line_sec.setData([], [])
        self.ram_line_sec.setData([], [])
        self._update_point_count_label("cpu", 0)
        self._update_point_count_label("gpu", 0)
        self._update_point_count_label("ram", 0)
        self._clear_epoch_lines()

    def _make_plot(self, title, color, layout, symbol=None):
        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        frame.setMinimumHeight(350)
        vbox = QtWidgets.QVBoxLayout(frame)

        plot = pg.PlotWidget(title=title)
        plot.setMouseEnabled(x=True, y=True)
        plot.setMenuEnabled(False)
        plot.setInteractive(True)
        plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setBackground("w")
        plot.getAxis("left").setTextPen("black")
        plot.getAxis("bottom").setTextPen("black")
        
        # Set axis labels based on plot type
        plot.getAxis("bottom").setLabel("Epoch")
        if "CPU" in title:
            plot.getAxis("left").setLabel("Usage (%)")
        elif "GPU" in title:
            plot.getAxis("left").setLabel("Memory (GB)")
        elif "RAM" in title:
            plot.getAxis("left").setLabel("Memory (GB)")
        else:
            plot.getAxis("left").setLabel("Value")
        
        line = plot.plot(pen=pg.mkPen(color, width=2), symbol=symbol)

        vbox.addWidget(plot)
        layout.addWidget(frame)
        return plot, line

    def _make_sampling_plot(self, title, color, layout, plot_id):
        """
        Create a plot with a sampling rate dropdown control.
        Returns (plot_widget, line, combo_box).
        """
        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        frame.setMinimumHeight(380)
        vbox = QtWidgets.QVBoxLayout(frame)
        vbox.setSpacing(5)
        vbox.setContentsMargins(10, 10, 10, 10)

        # --- Sampling Control Bar ---
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(10)
        
        # Sampling label and dropdown
        sampling_label = QtWidgets.QLabel("<b>Sampling:</b>")
        sampling_label.setStyleSheet("font-size: 10px;")
        control_layout.addWidget(sampling_label)
        
        sampling_combo = QtWidgets.QComboBox()
        sampling_combo.setToolTip("Select data sampling rate. Lower = more points, higher detail. Higher = fewer points, faster rendering.")
        sampling_combo.addItem("1:1 (All Points)", 1)
        sampling_combo.addItem("1:5", 5)
        sampling_combo.addItem("1:10", 10)
        sampling_combo.addItem("1:50", 50)
        sampling_combo.addItem("1:100", 100)
        sampling_combo.addItem("1:500", 500)
        sampling_combo.addItem("1:1000", 1000)
        sampling_combo.setCurrentIndex(0)  # Default to 1:1
        sampling_combo.setMaximumWidth(120)
        sampling_combo.currentIndexChanged.connect(self._on_sampling_changed)
        control_layout.addWidget(sampling_combo)
        
        # Point count label
        point_count_label = QtWidgets.QLabel("<span style='color: gray; font-size: 9px;' id='point_count_%s'>Points: --</span>" % plot_id)
        point_count_label.setObjectName(f"point_count_{plot_id}")
        control_layout.addWidget(point_count_label)
        
        control_layout.addStretch(1)
        vbox.addLayout(control_layout)

        # --- Plot Widget ---
        plot = pg.PlotWidget(title=title)
        plot.setMouseEnabled(x=True, y=True)
        plot.setMenuEnabled(False)
        plot.setInteractive(True)
        plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setBackground("w")
        plot.getAxis("left").setTextPen("black")
        plot.getAxis("bottom").setTextPen("black")
        
        # Set axis labels based on plot type
        plot.getAxis("bottom").setLabel("Time (seconds)")
        if "CPU" in title:
            plot.getAxis("left").setLabel("Usage (%)")
        elif "GPU" in title:
            plot.getAxis("left").setLabel("Memory (GB)")
        elif "RAM" in title:
            plot.getAxis("left").setLabel("Memory (GB)")
        else:
            plot.getAxis("left").setLabel("Value")
        
        line = plot.plot(pen=pg.mkPen(color, width=2))
        
        # Add coordinate label at bottom left
        coord_label = QtWidgets.QLabel("Time: -- | Value: --")
        coord_label.setStyleSheet("font-size: 9px; color: #666; background-color: rgba(255,255,255,0.8); padding: 2px 5px;")
        
        # Enable hover events for coordinate tracking
        plot.scene().sigMouseMoved.connect(lambda pos, p=plot, l=coord_label: self._on_plot_mouse_moved(pos, p, l))
        plot.scene().sigMouseHover.connect(lambda items, l=coord_label: l.setText("Time: -- | Value: --") if not items else None)

        vbox.addWidget(plot)
        vbox.addWidget(coord_label)
        layout.addWidget(frame)
        
        # Store reference to point count label for updates
        setattr(self, f"_point_count_label_{plot_id}", point_count_label)
        setattr(self, f"_coord_label_{plot_id}", coord_label)
        
        return plot, line, sampling_combo
        
    def _on_plot_mouse_moved(self, pos, plot, label):
        """Handle mouse movement over a plot to update coordinate label."""
        if plot.sceneBoundingRect().contains(pos):
            mouse_point = plot.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()
            y = mouse_point.y()
            label.setText(f"Time: {x:.2f}s | Value: {y:.2f}")

    def _on_sampling_changed(self):
        """Handler called when sampling rate changes. Updates the plots."""
        self._cached_df = None  # Clear cache to force re-render with new sampling
        if self._data_loaded:
            if self._live_mode:
                self.update_pReport_plots()
            else:
                # In static mode, do a one-time update
                self._update_plots_once()

    def _apply_sampling(self, x, y, sampling_rate):
        """
        Apply sampling to data based on selected rate.
        sampling_rate: 1 = all points, 5 = every 5th point, etc.
        """
        # Convert to numpy arrays first (PyQtGraph needs arrays, not pandas Series)
        x_arr = x.values if hasattr(x, 'values') else np.array(x)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)
        
        if sampling_rate <= 1:
            return x_arr, y_arr
        
        n = len(x_arr)
        if n <= sampling_rate:
            return x_arr, y_arr
        
        # Simple uniform sampling: take every Nth point
        indices = np.arange(0, n, sampling_rate)
        
        # Ensure we include the last point for completeness
        if indices[-1] != n - 1:
            indices = np.append(indices, n - 1)
        
        return x_arr[indices], y_arr[indices]

    def _update_point_count_label(self, plot_id, count):
        """Update the point count display for a plot."""
        label = getattr(self, f"_point_count_label_{plot_id}", None)
        if label:
            label.setText(f"<span style='color: gray; font-size: 9px;'>Points: {count:,}</span>")

    def update_pReport_plots(self):
        """Main update function for Performance Reports tab."""
        try:
            # --- ALWAYS update average plots (they show all data regardless of selection) ---
            self._update_average_plots()

            # --- Per-second plots only update when user has made a selection ---
            if not self._data_loaded:
                # Don't load per-second data until user makes a selection
                self._clear_per_second_plots()
                self.range_status_label.setText("⚠️ Select a data range or enable LIVE MODE to view per-second data")
                return

            if not self.file_sec.exists():
                # File doesn't exist yet - show message but don't error
                self._clear_per_second_plots()
                if self._live_mode:
                    self.range_status_label.setText("🔴 LIVE: Waiting for training data...")
                else:
                    self.range_status_label.setText("⚠️ No data file found. Start training to generate data.")
                return

            # Check file modification time and size for caching
            file_stat = self.file_sec.stat()
            cache_key = (file_stat.st_mtime, file_stat.st_size)

            # In live mode, always reload to get latest data during training
            # In manual mode, use cache if file hasn't changed
            if self._live_mode or self._cached_df is None or getattr(self, '_cache_key', None) != cache_key:
                df = pd.read_csv(self.file_sec)
                if df.empty:
                    self._clear_per_second_plots()
                    return

                if "epoch_marker" in df.columns:
                    df["epoch"] = df["epoch_marker"].fillna("").apply(
                        lambda x: int(x.split("_")[1]) if isinstance(x, str) and x.startswith("epoch_") else np.nan
                    )
                    # Forward-fill epoch values so each data point knows its epoch
                    # (epochs are marked at the start of each epoch, so fill backward and forward)
                    df["epoch"] = df["epoch"].fillna(method="ffill")
                    df["epoch"] = df["epoch"].fillna(method="bfill")  # For rows before first epoch marker
                else:
                    df["epoch"] = np.nan

                df = df.dropna(subset=["time"])
                if df.empty:
                    self._clear_per_second_plots()
                    return

                # Use earliest process init time as reference (to sync with Gantt chart)
                # Falls back to first CSV entry if process data not available
                process_min_time = self._get_process_min_time()
                if process_min_time is not None:
                    df["time"] -= process_min_time
                else:
                    df["time"] -= df["time"].iloc[0]

                # Cache the processed dataframe
                self._cached_df = df
                self._cache_key = cache_key
            else:
                df = self._cached_df

            # --- Apply filtering based on mode ---
            if self._live_mode:
                # Live mode: show most recent window of data
                df_filtered = self._get_live_window_range(df)
                if not df_filtered.empty:
                    max_epoch = df_filtered["epoch"].max()
                    min_epoch = df_filtered["epoch"].min()
                    self.range_status_label.setText(f"🔴 LIVE: Epochs {min_epoch:.0f}-{max_epoch:.0f} | Window: {self.live_window_combo.currentText()}")
            else:
                # Manual range mode
                start_epoch, end_epoch = self._epoch_range
                df_filtered = df.copy()

                if start_epoch is not None:
                    df_filtered = df_filtered[df_filtered["epoch"] >= start_epoch]
                if end_epoch is not None:
                    df_filtered = df_filtered[df_filtered["epoch"] <= end_epoch]
                
                # If filtered data is empty, show warning but still use unfiltered for averages
                if df_filtered.empty:
                    df_filtered = df  # Fall back to all data
                    if start_epoch is not None or end_epoch is not None:
                        self.range_status_label.setText("Warning: No data in selected range, showing all")

            # Get sampling rates from user selection (default to 1 = all points)
            cpu_sampling = self.cpu_sampling_combo.currentData() or 1
            gpu_sampling = self.gpu_sampling_combo.currentData() or 1
            ram_sampling = self.ram_sampling_combo.currentData() or 1

            # Apply user-selected sampling to each plot
            t_cpu, cpu = self._apply_sampling(df_filtered["time"], df_filtered["cpu_percent"], cpu_sampling)
            t_gpu, gpu = self._apply_sampling(df_filtered["time"], df_filtered["gpu_gb"], gpu_sampling)
            t_ram, ram = self._apply_sampling(df_filtered["time"], df_filtered["ram_gb"], ram_sampling)

            self._update_visibility(self.p_cpu_sec, self.cpu_line_sec, t_cpu, cpu)
            self._update_visibility(self.p_gpu_sec, self.gpu_line_sec, t_gpu, gpu)
            self._update_visibility(self.p_ram_sec, self.ram_line_sec, t_ram, ram)

            # Update point count labels
            self._update_point_count_label("cpu", len(t_cpu))
            self._update_point_count_label("gpu", len(t_gpu))
            self._update_point_count_label("ram", len(t_ram))

            self._clear_epoch_lines()
            # Draw epoch lines only for the filtered range
            self._draw_epoch_lines(df_filtered)

        except Exception as e:
            print("Update error:", e)
            
    def _update_average_plots(self):
        """Update the average per-epoch plots. Always shown regardless of per-second selection."""
        try:
            if not self.file_sec.exists():
                # Clear average plots if no data
                self.cpu_line_epoch.setData([], [])
                self.gpu_line_epoch.setData([], [])
                self.ram_line_epoch.setData([], [])
                return

            df = pd.read_csv(self.file_sec)
            if df.empty:
                return

            if "epoch_marker" in df.columns:
                df["epoch"] = df["epoch_marker"].fillna("").apply(
                    lambda x: int(x.split("_")[1]) if isinstance(x, str) and x.startswith("epoch_") else np.nan
                )
                # Forward-fill epoch values so each data point knows its epoch
                df["epoch"] = df["epoch"].fillna(method="ffill")
                df["epoch"] = df["epoch"].fillna(method="bfill")
            else:
                df["epoch"] = np.nan

            df = df.dropna(subset=["time"])
            if df.empty:
                return

            # --- Average per-epoch plots (respect range selection) ---
            if df["epoch"].notna().any():
                df_epoch = df.groupby("epoch").agg({
                    "cpu_percent": "mean",
                    "gpu_gb": "mean",
                    "ram_gb": "mean",
                }).reset_index()
                
                # Apply the same epoch range filter as per-second plots
                start_epoch, end_epoch = self._epoch_range
                if start_epoch is not None:
                    df_epoch = df_epoch[df_epoch["epoch"] >= start_epoch]
                if end_epoch is not None:
                    df_epoch = df_epoch[df_epoch["epoch"] <= end_epoch]
                
                self._update_visibility(self.p_cpu_epoch, self.cpu_line_epoch, df_epoch["epoch"], df_epoch["cpu_percent"])
                self._update_visibility(self.p_gpu_epoch, self.gpu_line_epoch, df_epoch["epoch"], df_epoch["gpu_gb"])
                self._update_visibility(self.p_ram_epoch, self.ram_line_epoch, df_epoch["epoch"], df_epoch["ram_gb"])
        except Exception as e:
            print("Update average plots error:", e)

    def update_linguistic_plot(self):
        try:
            with open("eval_results/eval_metrics.json", "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            
            # Rename columns for consistency
            column_mapping = {"cer": "CER", "wer": "WER", "bleu": "BLEU", "loss": "Loss"}
            df.rename(columns=column_mapping, inplace=True)

            for col in ["epoch", "CER", "WER", "BLEU"]:
                if col not in df.columns:
                    return

            self.line_cer.setData(df["epoch"], df["CER"])
            self.line_wer.setData(df["epoch"], df["WER"])
            self.line_bleu.setData(df["epoch"], df["BLEU"])
            
            # Update loss line if available
            if "Loss" in df.columns:
                self.line_loss.setData(df["epoch"], df["Loss"])

            self.plot_linguistic.setLabel("left", "Score")
            self.plot_linguistic.setLabel("bottom", "Epoch")
            self.plot_linguistic.setTitle("Translation Metrics Over Epochs")

        except Exception as e:
            print("Update error:", e)

    def _update_visibility(self, plot, line, x, y):
        line.setData(x, y)
        plot.parentWidget().show()

    def _draw_epoch_lines(self, df):
        """Draw vertical lines marking epoch boundaries. Only shows first occurrence of each epoch."""
        epochs = sorted(df["epoch"].dropna().unique())
        if not epochs:
            return
        
        # Limit number of epoch lines to avoid clutter (show every Nth epoch if too many)
        total_epochs = len(epochs)
        if total_epochs > 50:
            # Show every 5th epoch if many epochs
            step = max(1, total_epochs // 20)
            epochs_to_show = epochs[::step]
        else:
            epochs_to_show = epochs

        for epoch in epochs_to_show:
            idx = df[df["epoch"] == epoch].index
            if len(idx) == 0:
                continue
            # Get the first time point for this epoch
            t = df.loc[idx[0], "time"]
            for p in [self.p_cpu_sec, self.p_gpu_sec, self.p_ram_sec]:
                line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen("gray", style=self.dash_line))
                label = pg.TextItem(f"E{int(epoch)}", anchor=(0, 1), color=(100, 100, 100))
                # Position label at top of current view
                view_range = p.viewRange()
                y_max = view_range[1][1] if view_range[1][1] > 0 else 100
                label.setPos(t, y_max * 0.95)
                p.addItem(line)
                p.addItem(label)
                self.epoch_lines.append((p, line, label))

    def _clear_epoch_lines(self):
        for p, line, label in self.epoch_lines:
            p.removeItem(line)
            p.removeItem(label)
        self.epoch_lines = []

    def _print_performance_report_to_pdf(self):
        """Print the Performance Reports tab to PDF."""
        try:
            # Get file path from user
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Performance Report as PDF",
                str(Path.cwd() / "performance_report.pdf"),
                "PDF Files (*.pdf)"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Ensure .pdf extension
            if not file_path.endswith('.pdf'):
                file_path += '.pdf'
            
            # Create PDF writer
            from PyQt6.QtGui import QPainter, QPageLayout, QPageSize
            
            pdf_writer = QtGui.QPdfWriter(file_path)
            pdf_writer.setPageLayout(QPageLayout(QPageSize(QPageSize.PageSizeId.A4), QPageLayout.Orientation.Portrait, QtCore.QMarginsF(20, 20, 20, 20)))
            
            # Create painter
            painter = QPainter(pdf_writer)
            
            # Get the scroll area widget
            widget = self.scroll_pReport.widget()
            
            # Calculate scaling to fit page width
            page_rect = pdf_writer.pageLayout().paintRectPixels(pdf_writer.resolution())
            scale = page_rect.width() / widget.width()
            
            # Scale painter
            painter.scale(scale, scale)
            
            # Render widget to PDF
            widget.render(painter)
            
            painter.end()
            
            QtWidgets.QMessageBox.information(self, "Success", f"Performance report saved to:\n{file_path}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save PDF:\n{str(e)}")
            print(f"PDF export error: {e}")

    def _print_linguistic_report_to_pdf(self):
        """Print the Linguistic Performance Report tab to PDF."""
        try:
            # Get file path from user
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Linguistic Report as PDF",
                str(Path.cwd() / "linguistic_report.pdf"),
                "PDF Files (*.pdf)"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Ensure .pdf extension
            if not file_path.endswith('.pdf'):
                file_path += '.pdf'
            
            # Create PDF writer
            from PyQt6.QtGui import QPainter, QPageLayout, QPageSize
            
            pdf_writer = QtGui.QPdfWriter(file_path)
            pdf_writer.setPageLayout(QPageLayout(QPageSize(QPageSize.PageSizeId.A4), QPageLayout.Orientation.Portrait, QtCore.QMarginsF(20, 20, 20, 20)))
            
            # Create painter
            painter = QPainter(pdf_writer)
            
            # Get the scroll area widget
            widget = self.scroll_LPR.widget()
            
            # Calculate scaling to fit page width
            page_rect = pdf_writer.pageLayout().paintRectPixels(pdf_writer.resolution())
            scale = page_rect.width() / widget.width()
            
            # Scale painter
            painter.scale(scale, scale)
            
            # Render widget to PDF
            widget.render(painter)
            
            painter.end()
            
            QtWidgets.QMessageBox.information(self, "Success", f"Linguistic report saved to:\n{file_path}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save PDF:\n{str(e)}")
            print(f"PDF export error: {e}")

    def _print_attention_report_to_pdf(self):
        """Print the Performance During Inference tab to PDF with multi-page support."""
        try:
            # Get file path from user
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Inference Report as PDF",
                str(Path.cwd() / "inference_report.pdf"),
                "PDF Files (*.pdf)"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Ensure .pdf extension
            if not file_path.endswith('.pdf'):
                file_path += '.pdf'
            
            # Create PDF writer
            from PyQt6.QtGui import QPainter, QPageLayout, QPageSize
            
            pdf_writer = QtGui.QPdfWriter(file_path)
            pdf_writer.setPageLayout(QPageLayout(QPageSize(QPageSize.PageSizeId.A4), QPageLayout.Orientation.Portrait, QtCore.QMarginsF(20, 20, 20, 20)))
            
            # Create painter
            painter = QPainter(pdf_writer)
            
            # Get the scroll area content widget
            widget = self.scroll_attn.widget()
            
            # Get page dimensions
            page_rect = pdf_writer.pageLayout().paintRectPixels(pdf_writer.resolution())
            page_width = page_rect.width()
            page_height = page_rect.height()
            
            # Calculate scale to fit width
            content_width = widget.width()
            scale = page_width / content_width
            
            # Get the full content height (not just visible portion)
            content_height = int(widget.height() * scale)
            
            # Calculate how many pages we need
            num_pages = max(1, (content_height + page_height - 1) // page_height)
            
            # Render each page
            for page in range(num_pages):
                if page > 0:
                    pdf_writer.newPage()
                
                # Save painter state
                painter.save()
                
                # Scale to fit page width
                painter.scale(scale, scale)
                
                # Translate to show the correct portion of the content
                # We need to shift up by the amount we've already rendered
                y_offset = page * (page_height / scale)
                painter.translate(0, -y_offset)
                
                # Render the widget
                widget.render(painter)
                
                # Restore painter state
                painter.restore()
            
            painter.end()
            
            QtWidgets.QMessageBox.information(self, "Success", f"Inference report saved to:\n{file_path}\n({num_pages} page(s))")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save PDF:\n{str(e)}")
            print(f"PDF export error: {e}")


    def _setup_process_mindmap_tab(self):
        """Setup the Process Mindmap tab for visualizing process relationships."""
        # Main layout for the tab
        tab_layout = QtWidgets.QVBoxLayout(self.tab_mindmap)
        tab_layout.setSpacing(10)
        tab_layout.setContentsMargins(10, 10, 10, 10)

        # --- Control Panel ---
        control_frame = QtWidgets.QFrame()
        control_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        control_frame.setStyleSheet("QFrame { background-color: #f0f0f0; border: 1px solid #cccccc; border-radius: 5px; }")
        control_layout = QtWidgets.QHBoxLayout(control_frame)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(15, 10, 15, 10)

        # Title label
        title_label = QtWidgets.QLabel("<b>Process Hierarchy (Tree View)</b>")
        title_label.setStyleSheet("font-size: 12px; color: #333333;")
        control_layout.addWidget(title_label)

        control_layout.addStretch(1)

        # Refresh Button
        self.refresh_mindmap_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_mindmap_btn.setStyleSheet("QPushButton { padding: 5px 15px; background-color: #4CAF50; color: white; border-radius: 3px; }")
        self.refresh_mindmap_btn.clicked.connect(self._load_and_draw_mindmap)
        control_layout.addWidget(self.refresh_mindmap_btn)

        # Print to PDF Button
        self.print_mindmap_pdf_btn = QtWidgets.QPushButton("Print to PDF")
        self.print_mindmap_pdf_btn.setStyleSheet("QPushButton { padding: 5px 15px; background-color: #FF9800; color: white; border-radius: 3px; }")
        self.print_mindmap_pdf_btn.clicked.connect(self._print_mindmap_to_pdf)
        control_layout.addWidget(self.print_mindmap_pdf_btn)

        tab_layout.addWidget(control_frame)

        # --- Status Label ---
        self.mindmap_status_label = QtWidgets.QLabel("Click Refresh to load process data")
        self.mindmap_status_label.setStyleSheet("font-size: 11px; color: #666666; padding: 5px;")
        tab_layout.addWidget(self.mindmap_status_label)

        # --- Scroll Area for Mindmap ---
        self.scroll_mindmap = QtWidgets.QScrollArea()
        self.scroll_mindmap.setWidgetResizable(True)
        self.scroll_mindmap.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_mindmap.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tab_layout.addWidget(self.scroll_mindmap, stretch=1)

        # Content widget for scroll area - fills entire space
        content = QtWidgets.QWidget()
        self.scroll_layout_mindmap = QtWidgets.QVBoxLayout(content)
        self.scroll_layout_mindmap.setSpacing(0)
        self.scroll_layout_mindmap.setContentsMargins(5, 5, 5, 5)
        self.scroll_mindmap.setWidget(content)

        # Store mindmap widget reference
        self.mindmap_widget = None
        self.mindmap_data = None

    def _load_and_draw_mindmap(self):
        """Load processes from JSON and draw the tree view."""
        json_path = Path("eval_results/processes.json")
        
        if not json_path.exists():
            self.mindmap_status_label.setText("File not found: " + str(json_path))
            return
        
        if json_path.stat().st_size == 0:
            self.mindmap_status_label.setText("Process file is empty")
            return
        
        try:
            with open(json_path, 'r') as f:
                self.mindmap_data = json.load(f)
        except json.JSONDecodeError as e:
            self.mindmap_status_label.setText("Invalid JSON: " + str(e))
            return
        except Exception as e:
            self.mindmap_status_label.setText("Error loading file: " + str(e))
            return
        
        # Clear previous mindmap
        if self.mindmap_widget:
            self.mindmap_widget.setParent(None)
            self.mindmap_widget.deleteLater()
            self.mindmap_widget = None
        
        # Create tree visualization
        processes = self.mindmap_data.get("processes", {})
        self.mindmap_widget = self._create_tree_visualization(processes)
        self.scroll_layout_mindmap.addWidget(self.mindmap_widget, stretch=1)
        
        process_count = len(processes)
        self.mindmap_status_label.setText(f"Loaded {process_count} processes")
    
    def _create_tree_visualization(self, processes):
        """Create a tree widget visualization showing hierarchy AND parallelism."""
        viz_widget = QtWidgets.QWidget()
        viz_layout = QtWidgets.QVBoxLayout(viz_widget)
        viz_layout.setSpacing(5)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        
        if not processes:
            no_data_label = QtWidgets.QLabel("No process data available")
            no_data_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            viz_layout.addWidget(no_data_label)
        else:
            # Create a tree widget for visualization with columns
            tree_widget = QtWidgets.QTreeWidget()
            tree_widget.setHeaderLabels(["Process", "Epoch", "Layer", "Duration", "Execution"])
            tree_widget.setColumnWidth(0, 400)  # Process name column
            tree_widget.setColumnWidth(1, 50)   # Epoch column
            tree_widget.setColumnWidth(2, 50)   # Layer column
            tree_widget.setColumnWidth(3, 100)  # Duration column
            tree_widget.setColumnWidth(4, 120)  # Execution type column
            
            tree_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            tree_widget.setMinimumHeight(600)
            tree_widget.setMinimumWidth(900)
            tree_widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            
            header = tree_widget.header()
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
            header.setStretchLastSection(False)
            
            # Colors for parallel groups
            parallel_bg_colors = [
                QtGui.QColor(200, 255, 200),  # Light green
                QtGui.QColor(255, 230, 200),  # Light orange
                QtGui.QColor(200, 220, 255),  # Light blue
                QtGui.QColor(255, 200, 220),  # Light pink
            ]
            
            tree_widget.setStyleSheet("""
                QTreeWidget {
                    font-size: 12px;
                    border: 1px solid #cccccc;
                    border-radius: 5px;
                    padding: 5px;
                }
                QTreeWidget::item {
                    padding: 6px;
                    margin: 2px;
                    border-radius: 3px;
                }
                QTreeWidget::item:selected {
                    background-color: #b3d9ff;
                }
            """)
            
            visited = set()
            parallel_group_counter = {}  # Track color assignment per parent
            
            def get_epoch_from_uid(uid):
                """Extract epoch from UID format: epochxlayer-id (e.g., '5x2-abc123')."""
                if "x" in uid and "-" in uid:
                    try:
                        return int(uid.split("x")[0])
                    except ValueError:
                        pass
                return 0
            
            def get_layer_from_uid(uid):
                """Extract layer from UID format: epochxlayer-id (e.g., '5x2-abc123')."""
                if "x" in uid and "-" in uid:
                    try:
                        layer_part = uid.split("x")[1]
                        return int(layer_part.split("-")[0])
                    except (ValueError, IndexError):
                        pass
                return 0
            
            def add_process_to_tree(parent_item, proc_uid, depth=0, is_parallel=False, parallel_color=None, parent_id=None):
                """Add a process to the tree with proper parallelism handling."""
                if proc_uid in visited:
                    return None
                visited.add(proc_uid)
                
                proc_data = processes.get(proc_uid, {})
                name = proc_data.get("name", "Unknown")
                # Extract epoch and layer from UID
                epoch = get_epoch_from_uid(proc_uid)
                layer = get_layer_from_uid(proc_uid)
                timeline = proc_data.get("timeline", {})
                init_time = timeline.get("initialized", 0)
                term_time = timeline.get("terminated", 0)
                duration = term_time - init_time if term_time and init_time else 0
                
                if parent_item is None:
                    item = QtWidgets.QTreeWidgetItem(tree_widget)
                else:
                    item = QtWidgets.QTreeWidgetItem(parent_item)
                
                # Format name with parallel indicator if applicable
                if is_parallel:
                    item.setText(0, f"⚡ {name}  ({proc_uid[:8]}...)")
                else:
                    item.setText(0, f"{name}  ({proc_uid[:8]}...)")
                
                # Column 1: Epoch
                item.setText(1, str(epoch))
                item.setTextAlignment(1, QtCore.Qt.AlignmentFlag.AlignCenter)
                
                # Column 2: Layer    
                item.setText(2, str(layer))
                item.setTextAlignment(2, QtCore.Qt.AlignmentFlag.AlignCenter)
                
                # Column 3: Duration
                if duration > 0:
                    item.setText(3, f"{duration:.3f}s")
                else:
                    item.setText(3, "ongoing")
                item.setTextAlignment(3, QtCore.Qt.AlignmentFlag.AlignCenter)
                
                # Column 4: Execution Type
                if is_parallel:
                    item.setText(4, "⚡ Parallel")
                    # Apply parallel group color to all columns
                    if parallel_color:
                        for col in range(5):
                            item.setBackground(col, parallel_color)
                else:
                    item.setText(4, "Sequential")
                
                # Get subtasks and categorize them
                subtasks = proc_data.get("subtasks", []) or []
                subtask_layers = {}
                
                for subtask_uid in subtasks:
                    if subtask_uid in processes and subtask_uid not in visited:
                        # Get layer from UID for consistency
                        st_layer = get_layer_from_uid(subtask_uid)
                        if st_layer not in subtask_layers:
                            subtask_layers[st_layer] = []
                        subtask_layers[st_layer].append(subtask_uid)
                
                # Separate parallel groups from sequential tasks
                parallel_groups = {layer: tasks for layer, tasks in subtask_layers.items() if len(tasks) > 1}
                sequential_tasks = [uid for layer, tasks in subtask_layers.items() 
                                   for uid in tasks if len(tasks) == 1]
                
                # Generate unique key for this parent to track parallel group colors
                parent_key = parent_id if parent_id else "root"
                if parent_key not in parallel_group_counter:
                    parallel_group_counter[parent_key] = 0
                
                # Add sequential tasks first (as direct children)
                for subtask_uid in sequential_tasks:
                    add_process_to_tree(item, subtask_uid, depth + 1, is_parallel=False, parent_id=proc_uid)
                
                # Add parallel tasks as DIRECT CHILDREN (not in a container)
                # Each parallel task is independently expandable/collapsible
                for layer, task_uids in sorted(parallel_groups.items()):
                    # Assign color for this parallel group
                    color_idx = parallel_group_counter[parent_key] % len(parallel_bg_colors)
                    group_color = parallel_bg_colors[color_idx]
                    parallel_group_counter[parent_key] += 1
                    
                    # Add each parallel task as a direct child with parallel coloring
                    for subtask_uid in task_uids:
                        add_process_to_tree(item, subtask_uid, depth + 1, 
                                          is_parallel=True, parallel_color=group_color, 
                                          parent_id=proc_uid)
                
                return item
            
            # Build tree from root processes
            for uid, proc in processes.items():
                parent_uid = proc.get("parent_uid")
                if parent_uid is None or parent_uid not in processes:
                    add_process_to_tree(None, uid, 0)
            
            tree_widget.expandAll()
            viz_layout.addWidget(tree_widget, stretch=1)
            
            # Legend and info
            info_frame = QtWidgets.QFrame()
            info_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            info_frame.setStyleSheet("background-color: #f8f8f8; border-radius: 5px; padding: 5px;")
            info_layout = QtWidgets.QVBoxLayout(info_frame)
            info_layout.setSpacing(5)
            
            # Legend
            legend_layout = QtWidgets.QHBoxLayout()
            legend_layout.addWidget(QtWidgets.QLabel("<b>Structure:</b>"))
            
            seq_label = QtWidgets.QLabel("Normal (Sequential)")
            seq_label.setStyleSheet("padding: 2px 8px; background-color: white; border: 1px solid #ccc; border-radius: 3px;")
            legend_layout.addWidget(seq_label)
            
            par_label = QtWidgets.QLabel("⚡ Parallel (colored = same layer = concurrent)")
            par_label.setStyleSheet("padding: 2px 8px; background-color: #c8ffc8; border: 2px solid #4CAF50; border-radius: 3px; font-weight: bold;")
            legend_layout.addWidget(par_label)
            
            legend_layout.addStretch(1)
            info_layout.addLayout(legend_layout)
            
            # Statistics
            stats_layout = QtWidgets.QHBoxLayout()
            total_procs = len(processes)
            root_procs = len([p for p in processes.values() if p.get("parent_uid") is None])
            
            # Count parallel processes
            parallel_proc_count = 0
            for proc in processes.values():
                subtasks = proc.get("subtasks", []) or []
                subtask_layers = {}
                for st_uid in subtasks:
                    if st_uid in processes:
                        st_layer = processes[st_uid].get("layer", 0)
                        subtask_layers[st_layer] = subtask_layers.get(st_layer, 0) + 1
                parallel_proc_count += sum(count for count in subtask_layers.values() if count > 1)
            
            stats_layout.addWidget(QtWidgets.QLabel(f"<b>Total:</b> {total_procs}"))
            stats_layout.addWidget(QtWidgets.QLabel(f"<b>Root:</b> {root_procs}"))
            stats_layout.addWidget(QtWidgets.QLabel(f"<b>Sequential:</b> {total_procs - root_procs - parallel_proc_count}"))
            if parallel_proc_count > 0:
                stats_layout.addWidget(QtWidgets.QLabel(f"<b style='color: #2E7D32;'>⚡ Parallel:</b> {parallel_proc_count}"))
            stats_layout.addStretch(1)
            
            info_layout.addLayout(stats_layout)
            viz_layout.addWidget(info_frame)
        
        return viz_widget

    def _print_mindmap_to_pdf(self):
        """Print the process hierarchy to PDF."""
        from PyQt6.QtPrintSupport import QPrinter
        from PyQt6.QtGui import QPainter, QPageLayout
        from PyQt6.QtCore import QMarginsF
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Process Hierarchy to PDF", "process_hierarchy.pdf", "PDF Files (*.pdf)"
        )
        if not file_path:
            return
        
        try:
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(file_path)
            
            painter = QPainter(printer)
            
            # Render the tree widget
            if self.mindmap_widget:
                printer.setPageLayout(QPageLayout(
                    QPageLayout.PageSize.A4,
                    QPageLayout.Orientation.Portrait,
                    QMarginsF(20, 20, 20, 20)
                ))
                
                # Render the widget to PDF
                self.mindmap_widget.render(painter)
            else:
                # No widget to render - draw a message
                painter.drawText(painter.window(), QtCore.Qt.AlignmentFlag.AlignCenter, "No process data loaded")
            
            painter.end()
            self.mindmap_status_label.setText(f"Saved to PDF: {file_path}")
        except Exception as e:
            self.mindmap_status_label.setText(f"PDF export failed: {str(e)}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = ResourceMonitorApp()
    win.show()
    sys.exit(app.exec())
