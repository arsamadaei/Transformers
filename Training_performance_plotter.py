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
from train import get_model, get_ds, greedy_decode
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

        self.scroll_pReport, self.scroll_layout_pReport = self._make_scroll_area(self.tab_pReport)
        self.scroll_LPR, self.scroll_layout_LPR = self._make_scroll_area(self.tab_LPR)
        self.scroll_attn, self.scroll_layout_attn = self._make_scroll_area(self.tab_attn)

        try:
            self.dash_line = QtCore.Qt.PenStyle.DashLine
        except AttributeError:
            self.dash_line = QtCore.Qt.DashLine

        self.p_cpu_sec, self.cpu_line_sec = self._make_plot("CPU Usage (%)", "b", self.scroll_layout_pReport)
        self.p_gpu_sec, self.gpu_line_sec = self._make_plot("GPU Usage (GB)", "g", self.scroll_layout_pReport)
        self.p_ram_sec, self.ram_line_sec = self._make_plot("RAM Usage (GB)", "orange", self.scroll_layout_pReport)

        self.p_cpu_epoch, self.cpu_line_epoch = self._make_plot("CPU Usage (%) per Epoch", "b", self.scroll_layout_pReport, symbol="o")
        self.p_gpu_epoch, self.gpu_line_epoch = self._make_plot("GPU Usage (GB) per Epoch", "g", self.scroll_layout_pReport, symbol="o")
        self.p_ram_epoch, self.ram_line_epoch = self._make_plot("RAM Usage (GB) per Epoch", "orange", self.scroll_layout_pReport, symbol="o")

        self.epoch_lines = []

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

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_pReport_plots)
        self.timer.start(1000)

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
        self.decoding_time = QtWidgets.QLabel("<b>Decoding Time of Srouce to Target:</b> No data loaded.")
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
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model Weights File", "", "PyTorch Weights (*.pt)")
        if file_name:
            self.weights_path = file_name
            self.path_label.setText(f"Weights Path: {self.weights_path}")
            self.status_label.setText(f"File selected: {Path(file_name).name}. Click 'Generate Attention Maps'.")
            self.submit_button.setEnabled(True)
            self._clear_attention_widgets()

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
        self.decoding_time.setText("<b>Decoding Time:</b> No data loaded.")
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
            inference_stats = {"gpu_samples": [], "storage_gb": 0.0, "predicted_text": "<prediction unavailable>", "bleu": 0.0, "decode_time": 0.0}
        
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
        self.decoding_time.setText(f"<b>Decoding Time:</b> {inference_stats['decode_time']:.3f}")
        self.pred_time.setText(f"<b>Generation of Tokens Per Second:</b> {inference_stats['TPS']:.3f}")

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

    def update_pReport_plots(self):
        try:
            if not self.file_sec.exists():
                return
            df = pd.read_csv(self.file_sec)
            if df.empty:
                return

            if "epoch_marker" in df.columns:
                df["epoch"] = df["epoch_marker"].fillna("").apply(
                    lambda x: int(x.split("_")[1]) if isinstance(x, str) and x.startswith("epoch_") else np.nan
                )
            else:
                df["epoch"] = np.nan

            df = df.dropna(subset=["time"])
            if df.empty:
                return

            df["time"] -= df["time"].iloc[0]

            self._update_visibility(self.p_cpu_sec, self.cpu_line_sec, df["time"], df["cpu_percent"])
            self._update_visibility(self.p_gpu_sec, self.gpu_line_sec, df["time"], df["gpu_gb"])
            self._update_visibility(self.p_ram_sec, self.ram_line_sec, df["time"], df["ram_gb"])

            self._clear_epoch_lines()
            self._draw_epoch_lines(df)

            if df["epoch"].notna().any():
                df_epoch = df.groupby("epoch").agg({
                    "cpu_percent": "mean",
                    "gpu_gb": "mean",
                    "ram_gb": "mean",
                }).reset_index()
                self._update_visibility(self.p_cpu_epoch, self.cpu_line_epoch, df_epoch["epoch"], df_epoch["cpu_percent"])
                self._update_visibility(self.p_gpu_epoch, self.gpu_line_epoch, df_epoch["epoch"], df_epoch["gpu_gb"])
                self._update_visibility(self.p_ram_epoch, self.ram_line_epoch, df_epoch["epoch"], df_epoch["ram_gb"])
        except Exception as e:
            print("Update error:", e)

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
        epochs = sorted(df["epoch"].dropna().unique())
        if not epochs:
            return
        for epoch in epochs:
            idx = df[df["epoch"] == epoch].index
            if len(idx) == 0:
                continue
            t = df.loc[idx[0], "time"]
            for p in [self.p_cpu_sec, self.p_gpu_sec, self.p_ram_sec]:
                line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen("gray", style=self.dash_line))
                label = pg.TextItem(f"epoch_{int(epoch)}", anchor=(0, 1), color=(100, 100, 100))
                label.setPos(t, p.viewRange()[1][1] * 0.9)
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

