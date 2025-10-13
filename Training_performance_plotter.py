import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
import json

class ResourceMonitorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # --- File setup ---
        self.log_dir = Path("eval_results/resource_usage")
        self.file_sec = self.log_dir / "usage_seconds.csv"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # --- Window setup ---
        self.setWindowTitle("Resource Usage Monitor")
        self.resize(1400, 1000)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # --- Tabs setup ---
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Per-second plots
        self.tab_pReport = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_pReport, "Performance Reports")

        # Tab 2: Per-epoch plots
        self.tab_LPR = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_LPR, "Linguistic Performance Report")

        # --- Scroll areas for both tabs ---
        self.scroll_pReport, self.scroll_layout_pReport = self._make_scroll_area(self.tab_pReport)
        self.scroll_LPR, self.scroll_layout_LPR = self._make_scroll_area(self.tab_LPR)

        # --- Dash line style fallback ---
        try:
            self.dash_line = QtCore.Qt.PenStyle.DashLine
        except AttributeError:
            self.dash_line = QtCore.Qt.DashLine

        # --- Per-second plots ---
        self.p_cpu_sec, self.cpu_line_sec = self._make_plot("CPU Usage (%)", "b", self.scroll_layout_pReport)
        self.p_gpu_sec, self.gpu_line_sec = self._make_plot("GPU Usage (GB)", "g", self.scroll_layout_pReport)
        self.p_ram_sec, self.ram_line_sec = self._make_plot("RAM Usage (GB)", "orange", self.scroll_layout_pReport)

        # --- Per-epoch plots ---
        self.p_cpu_epoch, self.cpu_line_epoch = self._make_plot("CPU Usage (%) per Epoch", "b", self.scroll_layout_pReport, symbol="o")
        self.p_gpu_epoch, self.gpu_line_epoch = self._make_plot("GPU Usage (GB) per Epoch", "g", self.scroll_layout_pReport, symbol="o")
        self.p_ram_epoch, self.ram_line_epoch = self._make_plot("RAM Usage (GB) per Epoch", "orange", self.scroll_layout_pReport, symbol="o")

        # --- Epoch marker tracking ---
        self.epoch_lines = []
        
        # --- Linguistic report plots ---
        with open("eval_results/eval_metrics.json", "r") as f:
            data = json.load(f)

        # Extract metrics only
        metrics = []
        for entry in data:
            metrics.append({
                "epoch": entry["epoch"],
                "cer": entry["cer"],
                "wer": entry["wer"],
                "bleu": entry["bleu"]
            })

        # Create DataFrame
        df = pd.DataFrame(metrics)

        # Make sure column names match what you will use in setData
        df.rename(columns={"cer": "CER", "wer": "WER", "bleu": "BLEU"}, inplace=True)
        
        self.plot_linguistic, _ = self._make_plot("Translation Metrics Over Epochs", "k", self.scroll_layout_LPR)

        self.line_cer = self.plot_linguistic.plot(pen=pg.mkPen('r', width=2), symbol='o', symbolBrush='r')
        self.line_wer = self.plot_linguistic.plot(pen=pg.mkPen('b', width=2), symbol='o', symbolBrush='b')
        self.line_bleu = self.plot_linguistic.plot(pen=pg.mkPen('g', width=2), symbol='o', symbolBrush='g')

        legend = self.plot_linguistic.addLegend()
        
        legend.addItem(self.line_cer, "CER")
        legend.addItem(self.line_wer, "WER")
        legend.addItem(self.line_bleu, "BLEU")
        
        # X-axis: epoch
        epochs = df["epoch"]

        self.line_cer.setData(epochs, df["CER"])
        self.line_wer.setData(epochs, df["WER"])
        self.line_bleu.setData(epochs, df["BLEU"])

        # --- Timer for auto-refresh ---
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_pReport_plots)
        self.timer.start(1000)  # every second
        
        

    # --- Scrollable area builder ---
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

    # --- Plot builder ---
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

    # --- Update loops ---
    def update_pReport_plots(self):
        try:
            if not self.file_sec.exists():
                return
            df = pd.read_csv(self.file_sec)
            if df.empty:
                return

            # Parse epochs
            if "epoch_marker" in df.columns:
                df["epoch"] = df["epoch_marker"].fillna("").apply(
                    lambda x: int(x.split("_")[1]) if isinstance(x, str) and x.startswith("epoch_") else np.nan
                )
            else:
                df["epoch"] = np.nan

            df = df.dropna(subset=["time"])
            if df.empty:
                return

            # Normalize time
            df["time"] -= df["time"].iloc[0]

            # --- Update per-second lines ---
            self._update_visibility(self.p_cpu_sec, self.cpu_line_sec, df["time"], df["cpu_percent"])
            self._update_visibility(self.p_gpu_sec, self.gpu_line_sec, df["time"], df["gpu_gb"])
            self._update_visibility(self.p_ram_sec, self.ram_line_sec, df["time"], df["ram_gb"])

            # --- Epoch markers ---
            self._clear_epoch_lines()
            self._draw_epoch_lines(df)

            # --- Epoch averages ---
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

            # Check required columns
            for col in ["epoch", "CER", "WER", "BLEU"]:
                if col not in df.columns:
                    return

            # Update lines
            self.line_cer.setData(df["epoch"], df["CER"])
            self.line_wer.setData(df["epoch"], df["WER"])
            self.line_bleu.setData(df["epoch"], df["BLEU"])

            # Set labels
            self.plot_linguistic.setLabel("left", "Score")
            self.plot_linguistic.setLabel("bottom", "Epoch")
            self.plot_linguistic.setTitle("Translation Metrics Over Epochs")

        except Exception as e:
            print("Update error:", e)

    # --- Utility functions ---
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

