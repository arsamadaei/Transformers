import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import panel as pn
import time
from pathlib import Path
import numpy as np
import sys
import os
import socket 
import subprocess 
import json
import matplotlib.pyplot as plt

# --- CONFIGURATION & SETUP ---
pn.extension(sizing_mode="stretch_width")

log_dir = Path("eval_results/resource_usage")
file_sec = log_dir / "usage_seconds.csv"
file_epoch = log_dir / "usage_epochs.csv"
eval_results_file = Path("eval_results/eval_metrics.json") 

PORTS = [5006, 5007, 5008, 5009]

# --- OPTIMIZATION: GLOBAL DATA CACHE ---
# These DataFrames will hold the data read from the files. 
# We update them incrementally instead of rereading the whole file.
data_cache = {
    'sec': pd.DataFrame(),
    'epoch': pd.DataFrame(),
    'metrics': pd.DataFrame()
}
# Keep track of how many rows we've read from each file
last_read_rows = {
    'sec': 0,
    'epoch': 0,
    'metrics': 0
}

# --- PLOTTING FUNCTION (RESOURCE USAGE) ---

def create_plotly_figure(df, is_per_second=True):
    """Creates a Plotly figure (fixed position error)."""
    
    color_cpu = 'blue'
    color_ram = 'orange'
    color_gpu = 'green'
    
    x_data = df["time"] if is_per_second else df["epoch"]
    x_title = "Time (s)" if is_per_second else "Epoch"
    title_text = "Resource Usage — Per Second (RAM & GPU on Right Axes)" if is_per_second else "Resource Usage — Per Epoch (RAM & GPU on Right Axes)"
    
    # --- 3 AXES (Y1 CPU, Y2 RAM, Y3 GPU) ---
    fig = go.Figure()

    # 1. CPU (Y1: Left)
    fig.add_trace(go.Scatter(x=x_data, y=df["cpu_percent"], name="CPU %", mode='lines+markers' if not is_per_second else 'lines', line=dict(color=color_cpu), yaxis='y1'))
    # 2. RAM (Y2: Right, Visible)
    fig.add_trace(go.Scatter(x=x_data, y=df["ram_gb"], name="RAM (GB)", mode='lines+markers' if not is_per_second else 'lines', line=dict(color=color_ram), yaxis='y2'))
    # 3. GPU (Y3: Right, Hidden)
    fig.add_trace(go.Scatter(x=x_data, y=df["gpu_gb"], name="GPU (GB)", mode='lines+markers' if not is_per_second else 'lines', line=dict(color=color_gpu, dash='dot', width=1.5), yaxis='y3'))

    # --- AXIS CONFIGURATION ---
    ram_max = df["ram_gb"].max() * 1.1 if not df.empty else 12.0
    gpu_max = df["gpu_gb"].max() * 1.1 if not df.empty else 3.5
    
    if is_per_second:
        cpu_y_range = [0, 85] 
    else:
        cpu_y_range = [12, 29]

    # Y1: CPU 
    fig.update_layout(yaxis1=dict(title="CPU %", color=color_cpu, side='left', showgrid=True, range=cpu_y_range, domain=[0, 1]))
    # Y2: RAM 
    fig.update_layout(yaxis2=dict(title="RAM (GB)", color=color_ram, overlaying="y1", side="right", anchor="x", showgrid=False, range=[0, ram_max], position=1.0))
    # Y3: GPU - FIXED: position is now 1.0
    fig.update_layout(
        yaxis3=dict(
            title="GPU (GB)", color=color_gpu, overlaying="y1", side="right", anchor="x", showgrid=False, range=[0, gpu_max],
            position=1.0, # FIXED: Must be in [0, 1]
            linecolor="rgba(0,0,0,0)", tickfont=dict(color='rgba(0,0,0,0)'),
            title_font=dict(color='rgba(0,0,0,0)'), showticklabels=False
        )
    )

    # Epoch Markers (Simplified)
    if is_per_second and "epoch" in df.columns and not df["epoch"].empty:
        epochs = sorted(df["epoch"].dropna().unique())
        max_y_val = cpu_y_range[1] 
        for epoch in epochs:
            first_idx_df = df[df["epoch"] == epoch].iloc[:1]
            if not first_idx_df.empty:
                t = first_idx_df["time"].iloc[0]
                fig.add_vline(x=t, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_annotation(x=t, y=max_y_val, text=f"Epoch {int(epoch)}", showarrow=False, textangle=-45, font=dict(size=9, color="gray"), yref="y1", xref="x")

    # Final Layout
    legend_y = 1.15 if is_per_second else 1.05
    fig.update_layout(
        title_text=title_text, xaxis=dict(title=x_title, showgrid=True), height=600,
        hovermode="x unified", template="plotly_white", 
        legend=dict(x=0, y=legend_y, traceorder="normal", orientation="h", bgcolor='rgba(0,0,0,0)')
    )
    return fig

# ----------------------------------------------------------------------
# --- NEW PLOTTING FUNCTION (METRICS) ---
# ----------------------------------------------------------------------

def create_metrics_figure(df):
    """Generates a Plotly figure for metrics from the cached DataFrame."""
    if df.empty or 'epoch' not in df.columns:
        return pn.pane.Markdown("Metrics data is empty or missing 'epoch' column.")

    fig = go.Figure()
    score_cols = [col for col in df.columns if col.lower() not in ['epoch', 'time', 'step']]

    for col in score_cols:
        fig.add_trace(go.Scatter(
            x=df['epoch'], y=df[col], mode='lines+markers', name=col
        ))

    fig.update_layout(
        title_text="Translation Metrics Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Score",
        height=600,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0, y=1.1, traceorder="normal", orientation="h", bgcolor='rgba(0,0,0,0)')
    )
    return fig

# --- PROCESS MANAGEMENT ---

def kill_process_on_port(port):
    """Safely checks if port is free and attempts to kill if used by a prior run."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if s.connect_ex(('127.0.0.1', port)) != 0:
            s.close()
            return f"Port {port} is free."
        s.close()
        
        cmd = f"lsof -t -i :{port}" 
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout.strip() and result.returncode == 0:
            pid = result.stdout.strip().split('\n')[0]
            os.kill(int(pid), 15)
            time.sleep(1) 
            try:
                os.kill(int(pid), 0)
                os.kill(int(pid), 9)
            except OSError:
                return f"Successfully killed PID {pid} on port {port}."
        return f"Port {port} is in use."

    except Exception as e:
        return f"Error during port check/kill: {e}"

# ----------------------------------------------------------------------
# --- CORE OPTIMIZATION: INCREMENTAL DATA UPDATE ---
# ----------------------------------------------------------------------

def update_data_cache():
    """Reads only NEW data from files and updates the global cache."""
    global data_cache, last_read_rows
    
    # 1. Resource Usage (Seconds)
    if file_sec.exists():
        try:
            # Read only new data using skiprows and header=None (then reassign headers)
            df_new = pd.read_csv(file_sec, skiprows=last_read_rows['sec'], header=None)
            if not df_new.empty:
                # Read header separately to correctly name columns
                if last_read_rows['sec'] == 0:
                    header = pd.read_csv(file_sec, nrows=0).columns.tolist()
                else:
                    # Assume column order is consistent if not the first read
                    if not data_cache['sec'].empty:
                        header = data_cache['sec'].columns.tolist()
                    else:
                        header = pd.read_csv(file_sec, nrows=0).columns.tolist()

                df_new.columns = header
                data_cache['sec'] = pd.concat([data_cache['sec'], df_new], ignore_index=True)
                last_read_rows['sec'] += len(df_new)
        except Exception as e:
            print(f"Error reading {file_sec}: {e}")
            pass

    # 2. Resource Usage (Epochs) - SIMPLER: just read entire file if it's small/updated
    if file_epoch.exists():
        try:
            current_rows = len(pd.read_csv(file_epoch))
            if current_rows != last_read_rows['epoch']:
                df_epoch = pd.read_csv(file_epoch)
                data_cache['epoch'] = df_epoch
                last_read_rows['epoch'] = current_rows
        except Exception as e:
            print(f"Error reading {file_epoch}: {e}")
            pass
            
    # 3. Metrics (JSON) - Read entire file if it exists and has changed
    if eval_results_file.exists():
        try:
            with open(eval_results_file, "r") as f:
                data = json.load(f)
            
            if len(data) != last_read_rows['metrics']:
                data_cache['metrics'] = pd.DataFrame(data).sort_values(by="epoch")
                last_read_rows['metrics'] = len(data)
        except Exception as e:
            print(f"Error reading {eval_results_file}: {e}")
            pass
            
    # Process the 'sec' data once here to clean up the logic in update_plots
    if not data_cache['sec'].empty:
        df_sec = data_cache['sec']
        if "epoch_marker" in df_sec.columns and "epoch" not in df_sec.columns:
            df_sec["epoch"] = (
                df_sec["epoch_marker"]
                .fillna("")
                .apply(lambda x: int(x.split("_")[1]) if isinstance(x, str) and x.startswith("epoch_") else np.nan)
            )
        
        # Normalize time based on the first recorded time
        if "time" in df_sec.columns:
            t0 = df_sec["time"].iloc[0]
            df_sec["time"] = df_sec["time"].apply(lambda t: t - t0)
        
        data_cache['sec'] = df_sec # Update the cache with processed data

# --- MAIN APPLICATION LOGIC ---

@pn.depends()
def update_plots():
    """Uses the cached data to quickly generate plots and updates the Panel tabs."""
    
    # Run the efficient data update first
    update_data_cache()

    tabs_content = []
    
    # 1. Per-Second (Live)
    if not data_cache['sec'].empty:
        fig_sec = create_plotly_figure(data_cache['sec'], is_per_second=True)
        tabs_content.append(("Per-Second (Live)", pn.pane.Plotly(fig_sec, config={'responsive': True})))
    else:
        tabs_content.append(("Per-Second (Live)", pn.pane.Markdown("## ⏳ Waiting for per-second resource data...")))

    # 2. Per-Epoch
    if not data_cache['epoch'].empty:
        fig_epoch = create_plotly_figure(data_cache['epoch'].sort_values(by="epoch"), is_per_second=False)
        tabs_content.append(("Per-Epoch", pn.pane.Plotly(fig_epoch, config={'responsive': True})))
    else:
        tabs_content.append(("Per-Epoch", pn.pane.Markdown("## ⏳ Waiting for per-epoch resource data...")))


    # 3. Metrics (New Tab)
    if not data_cache['metrics'].empty:
        metrics_figure = create_metrics_figure(data_cache['metrics'])
        tabs_content.append(("Metrics", pn.pane.Plotly(metrics_figure, config={'responsive': True})))
    else:
        tabs_content.append(("Metrics", pn.pane.Markdown("## ⏳ Waiting for metrics data (eval_metrics.json)...")))
    
    
    if not tabs_content:
        return pn.pane.Markdown("## ⏳ Waiting for any data files to be created...", sizing_mode='stretch_width')

    return pn.Tabs(*tabs_content, sizing_mode="stretch_width")


# --- SERVER SETUP ---

template = pn.template.FastListTemplate(
    title='Live Resource Usage Dashboard',
    sidebar=[],
    main=[update_plots],
    header_background='#EEEEEE',
    theme=pn.theme.DefaultTheme
)

print("\n--- Starting Panel Server ---")

# Run initial data load to populate cache before server starts
update_data_cache()
print("Initial data cache loaded.")

for port in PORTS:
    kill_result = kill_process_on_port(port)
    print(f"Port {port} status: {kill_result}")
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_free = s.connect_ex(('127.0.0.1', port)) != 0
    s.close()
    
    if is_free:
        try:
            if 'panel serve' not in sys.argv:
                print(f"NOTE: Running using template.show(). Opening browser on port {port}...")
                template.show(port=port) 
                break 
            else:
                template.servable()
                break
        except OSError as e:
            if e.errno == 98: 
                print(f"Port {port} is still locked. Trying next port...")
            else:
                print(f"An unexpected OS error occurred on port {port}: {e}")
                sys.exit(1)
        except Exception as e:
            print(f"\nFATAL ERROR: Could not start Panel server on port {port}. Detail: {e}")
            sys.exit(1)

else: 
    print("\nFATAL ERROR: Failed to start Panel server on any port (5006-5009).")
    sys.exit(1)


while True:
    time.sleep(2)