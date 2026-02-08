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


def get_bleuscore(predicted: list, expected: list):
    bleu = BLEUScore()
    print(f"Target: {predicted}\nexpected: {expected}")
    
    return bleu(predicted, expected)


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

    inference_stats["bleu"] = get_bleuscore(TARGET_TEXT, predicted_text)

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

        # Connect tab change signal to update placeholders when Performance Reports tab is selected
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Setup Performance Reports tab with controls and scroll area
        self._setup_performance_reports_tab()
        
        self.scroll_LPR, self.scroll_layout_LPR = self._make_scroll_area(self.tab_LPR)
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
                {"epoch": entry["epoch"], "cer": entry["cer"], "wer": entry["wer"], "bleu": entry["bleu"]}
                for entry in data
            ])
            df.rename(columns={"cer": "CER", "wer": "WER", "bleu": "BLEU"}, inplace=True)
            epochs = df["epoch"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        self.plot_linguistic, _ = self._make_plot("Translation Metrics Over Epochs", "k", self.scroll_layout_LPR)

        self.line_cer = self.plot_linguistic.plot(pen=pg.mkPen('r', width=2), symbol='o', symbolBrush='r')
        self.line_wer = self.plot_linguistic.plot(pen=pg.mkPen('b', width=2), symbol='o', symbolBrush='b')
        self.line_bleu = self.plot_linguistic.plot(pen=pg.mkPen('g', width=2), symbol='o', symbolBrush='g')

        legend = self.plot_linguistic.addLegend()

        legend.addItem(self.line_cer, "CER")
        legend.addItem(self.line_wer, "WER")
        legend.addItem(self.line_bleu, "BLEU")

        if not df.empty:
            self.line_cer.setData(epochs, df["CER"])
            self.line_wer.setData(epochs, df["WER"])
            self.line_bleu.setData(epochs, df["BLEU"])

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
        self.gpu_infer_line = self.p_gpu_infer.plot(pen=pg.mkPen('g', width=2), symbol='o', symbolBrush='g')
        plots_layout.addWidget(self.p_gpu_infer)

            # Storage plot
        self.p_storage_infer = pg.PlotWidget(title="RAM Memory During Inference( GB)")
        self.p_storage_infer.setBackground("w")
        self.p_storage_infer.showGrid(x=True, y=True, alpha=0.3)
        self.storage_line = self.p_storage_infer.plot(pen=pg.mkPen('b', width=2), symbol='o', symbolBrush='b')
        plots_layout.addWidget(self.p_storage_infer)

        control_layout.addWidget(plots_frame)

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
        """Setup the Performance Reports tab with epoch range controls and separate sections."""
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

        control_layout.addStretch(1)

        # Status label showing current range
        self.range_status_label = QtWidgets.QLabel("⚠️ Select a data range or enable LIVE MODE")
        self.range_status_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
        control_layout.addWidget(self.range_status_label)

        tab_layout.addWidget(control_frame)

        # --- Scroll Area for Plots ---
        self.scroll_pReport = QtWidgets.QScrollArea()
        self.scroll_pReport.setWidgetResizable(True)
        tab_layout.addWidget(self.scroll_pReport)

        # Content widget for scroll area
        content = QtWidgets.QWidget()
        self.scroll_layout_pReport = QtWidgets.QVBoxLayout(content)
        self.scroll_layout_pReport.setSpacing(30)
        self.scroll_layout_pReport.setContentsMargins(20, 20, 20, 20)
        self.scroll_pReport.setWidget(content)

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

    def _on_tab_changed(self, index):
        """Handle tab selection change."""
        print(f"DEBUG: Tab changed to index {index}")
        # If Performance Reports tab is selected (index 0), update placeholders
        if index == 0:
            print(f"DEBUG: Updating placeholders for Performance Reports tab")
            self._update_epoch_input_placeholders()

    def _update_epoch_input_placeholders(self):
        """Update input placeholders to show available epoch range."""
        print(f"DEBUG: _update_epoch_input_placeholders called")
        try:
            if not self.file_sec.exists():
                print(f"DEBUG: File doesn't exist: {self.file_sec}")
                self.epoch_start_input.setPlaceholderText("1")
                self.epoch_end_input.setPlaceholderText("max")
                return
            
            # Read just the epoch_marker column to find range
            df = pd.read_csv(self.file_sec, usecols=["epoch_marker"])
            
            if "epoch_marker" in df.columns:
                epochs = df["epoch_marker"].dropna().apply(
                    lambda x: int(x.split("_")[1]) if isinstance(x, str) and x.startswith("epoch_") else None
                ).dropna()
                
                print(f"DEBUG: Found epochs: {list(epochs.unique())[:5]}... min={epochs.min()}, max={epochs.max()}")
                
                if not epochs.empty:
                    min_epoch = int(epochs.min())
                    max_epoch = int(epochs.max())
                    self.epoch_start_input.setPlaceholderText(str(min_epoch))
                    self.epoch_end_input.setPlaceholderText(str(max_epoch))
                    print(f"DEBUG: Set placeholders to {min_epoch} - {max_epoch}")
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
        line = plot.plot(pen=pg.mkPen(color, width=2))

        vbox.addWidget(plot)
        layout.addWidget(frame)
        
        # Store reference to point count label for updates
        setattr(self, f"_point_count_label_{plot_id}", point_count_label)
        
        return plot, line, sampling_combo

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
        print(f"DEBUG: _update_average_plots called, _epoch_range={self._epoch_range}")
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
                print(f"DEBUG: Average plots - filtering by range {start_epoch}-{end_epoch}, df_epoch has {len(df_epoch)} epochs before filter")
                if start_epoch is not None:
                    df_epoch = df_epoch[df_epoch["epoch"] >= start_epoch]
                if end_epoch is not None:
                    df_epoch = df_epoch[df_epoch["epoch"] <= end_epoch]
                print(f"DEBUG: Average plots - after filter, df_epoch has {len(df_epoch)} epochs")
                
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

            for col in ["epoch", "CER", "WER", "BLEU"]:
                if col not in df.columns:
                    return

            self.line_cer.setData(df["epoch"], df["CER"])
            self.line_wer.setData(df["epoch"], df["WER"])
            self.line_bleu.setData(df["epoch"], df["BLEU"])

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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = ResourceMonitorApp()
    win.show()
    sys.exit(app.exec())

