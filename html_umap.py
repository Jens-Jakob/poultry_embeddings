"""
Interactive UMAP visualization of image embeddings — HTML export version.

Loads patch embeddings, averages per image, projects to 2D with UMAP,
generates base64 thumbnails, and writes a self-contained HTML file with
Plotly. Hover over any point to see the image and filepath.

Usage:
    python umap_viz_html.py
    python umap_viz_html.py --embeddings-dir ./embeddings --n-neighbors 15 --min-dist 0.1
    python umap_viz_html.py --output report_umap.html
"""

import argparse
import base64
import io
import json
import os
import re
import sys

import numpy as np
from PIL import Image
import umap


def _to_long_path(path: str) -> str:
    """Prefix path with extended-length format to bypass the Windows 260-char MAX_PATH limit."""
    if path.startswith('\\\\?\\'):
        return path
    if path.startswith('\\\\'):
        return '\\\\?\\UNC\\' + path[2:]
    return '\\\\?\\' + os.path.abspath(path)


def load_embeddings(embeddings_dir):
    patches = np.load(os.path.join(embeddings_dir, "patch_embeddings.npy")).astype(np.float32)
    filenames = np.load(os.path.join(embeddings_dir, "filenames.npy"))
    cls_approx = patches.mean(axis=1)
    return cls_approx, filenames


def run_umap(vectors, n_neighbors, min_dist):
    print(f"Running UMAP on {vectors.shape[0]} images ({vectors.shape[1]}-dim) ...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=42)
    coords = reducer.fit_transform(vectors)
    print("UMAP done.")
    return coords


def make_thumbnail_b64(path, size=180):
    """Return a base64-encoded JPEG thumbnail, or empty string on failure."""
    try:
        with open(_to_long_path(path), 'rb') as f:
            img = Image.open(f)
            img.load()
            img = img.convert("RGB")
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        return ""


# Regex to pull YYYYMMDD from filenames like Cam=B-FN=20240408...
_DATE_RE = re.compile(r"FN=(\d{8})")


def parse_date_from_filename(filename: str) -> str:
    """Extract YYYYMMDD date string from filename, or '' if not found."""
    m = _DATE_RE.search(filename)
    if m:
        return m.group(1)
    return ""


def generate_html(coords, filenames, output_path, thumbnail_size=180):
    """Build a self-contained HTML file with Plotly scatter + hover images."""
    n = len(filenames)
    print(f"Generating thumbnails for {n} images ...")

    thumb_b64_list = []
    hover_texts = []
    dates = []
    for i, fn in enumerate(filenames):
        path = str(fn)
        basename = os.path.basename(path)
        b64 = make_thumbnail_b64(path, size=thumbnail_size)
        thumb_b64_list.append(b64)
        hover_texts.append(basename)
        dates.append(parse_date_from_filename(basename))
        if (i + 1) % 500 == 0 or i == n - 1:
            print(f"  {i + 1}/{n} thumbnails done")

    x_vals = coords[:, 0].tolist()
    y_vals = coords[:, 1].tolist()
    norms = np.linalg.norm(coords, axis=1).tolist()

    # Get sorted unique dates for the slider
    unique_dates = sorted(set(d for d in dates if d))
    print(f"  found {len(unique_dates)} unique dates in filenames")

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>UMAP — Image Embeddings</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        background: #0f0f1a;
        color: #ccc;
        font-family: 'Consolas', 'SF Mono', 'Fira Code', monospace;
        overflow: hidden;
        height: 100vh;
    }
    #plot { width: 100vw; height: 100vh; }

    #hover-card {
        display: none;
        position: fixed;
        z-index: 1000;
        pointer-events: none;
        background: #16213e;
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 6px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        max-width: 330px;
    }
    #hover-card img {
        display: block;
        width: 100%;
        border-radius: 4px;
    }
    #hover-card .label {
        margin-top: 4px;
        font-size: 10px;
        color: #8899aa;
        word-break: break-all;
        line-height: 1.3;
        text-align: center;
    }

    .toolbar {
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 50;
        background: rgba(15, 15, 26, 0.95);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid rgba(255,255,255,0.06);
        padding: 10px 24px;
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .toolbar-title {
        color: #fff;
        font-size: 14px;
        font-weight: bold;
        letter-spacing: 0.5px;
    }
    .toolbar-meta {
        font-size: 12px;
        color: #556;
    }
    .mode-btn {
        background: rgba(255,255,255,0.08);
        color: #888;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 5px;
        padding: 5px 12px;
        font-size: 12px;
        font-family: inherit;
        cursor: pointer;
        transition: all 0.15s;
    }
    .mode-btn:hover { background: rgba(255,255,255,0.15); color: #ccc; }
    .mode-btn.active { background: rgba(224, 92, 92, 0.3); color: #e05c5c; border-color: #e05c5c; }

    .modebar { display: none !important; }

    .date-filter {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-left: auto;
    }
    .date-filter label {
        font-size: 11px;
        color: #888;
    }
    .date-filter input[type="date"] {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 4px;
        color: #ccc;
        padding: 3px 8px;
        font-size: 11px;
        font-family: inherit;
    }
    .date-filter input[type="date"]::-webkit-calendar-picker-indicator {
        filter: invert(0.7);
    }
    .date-filter .date-info {
        font-size: 10px;
        color: #e05c5c;
        min-width: 80px;
    }

    #selection-panel {
        display: none;
        position: fixed;
        bottom: 16px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 100;
        background: rgba(22, 33, 62, 0.95);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 12px 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        backdrop-filter: blur(8px);
        text-align: center;
        max-width: 90vw;
    }
    #selection-panel .sel-count {
        font-size: 16px;
        color: #fff;
        font-weight: bold;
        margin-bottom: 8px;
    }
    #selection-panel .sel-count span {
        color: #e05c5c;
    }
    #selection-panel .sel-thumbs {
        display: flex;
        gap: 4px;
        justify-content: center;
        flex-wrap: wrap;
        max-height: 200px;
        overflow-y: auto;
        margin-bottom: 8px;
    }
    #selection-panel .sel-thumbs img {
        width: 60px;
        height: 60px;
        object-fit: cover;
        border-radius: 4px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    #selection-panel .sel-hint {
        font-size: 10px;
        color: #556;
    }
    .sel-export-btn {
        background: #e05c5c;
        color: #fff;
        border: none;
        border-radius: 5px;
        padding: 6px 14px;
        font-size: 12px;
        font-weight: bold;
        cursor: pointer;
        margin-top: 6px;
        font-family: inherit;
    }
    .sel-export-btn:hover { background: #c94444; }
</style>
</head>
<body>
<div class="toolbar">
    <span class="toolbar-title">UMAP — Image Embeddings</span>
    <span class="toolbar-meta" id="meta-info"></span>
    <button class="mode-btn active" id="btn-pan" onclick="setMode('pan')">Pan</button>
    <button class="mode-btn" id="btn-box" onclick="setMode('select')">Box Select</button>
    <button class="mode-btn" id="btn-lasso" onclick="setMode('lasso')">Lasso</button>
    <button class="mode-btn" id="btn-clear" onclick="clearSelection()">Clear Selection</button>
    <button class="mode-btn" id="btn-reset" onclick="resetZoom()">Reset Zoom</button>
    <div class="date-filter">
        <label>From</label>
        <input type="date" id="date-from" onchange="applyDateFilter()" />
        <label>To</label>
        <input type="date" id="date-to" onchange="applyDateFilter()" />
        <button class="mode-btn" id="btn-date-clear" onclick="clearDateFilter()">Clear Dates</button>
        <span class="date-info" id="date-info"></span>
    </div>
</div>
<div id="plot"></div>
<div id="hover-card">
    <img id="hover-img" src="" alt="" />
    <div class="label" id="hover-label"></div>
</div>
<div id="selection-panel">
    <div class="sel-count"><span id="sel-num">0</span> images selected</div>
    <div class="sel-thumbs" id="sel-thumbs"></div>
    <button class="sel-export-btn" onclick="exportSelection()">Export CSV</button>
    <div class="sel-hint">Use box select or lasso in the toolbar to select points</div>
</div>

<script>
// ---- DATA (injected by Python) ----
var X = __X_VALS__;
var Y = __Y_VALS__;
var NORMS = __NORMS__;
var LABELS = __LABELS__;
var THUMBS = __THUMBS__;
var DATES = __DATES__;
var UNIQUE_DATES = __UNIQUE_DATES__;
// ---- END DATA ----

document.getElementById('meta-info').textContent = X.length + ' images';

// Set date picker min/max from data
if (UNIQUE_DATES.length > 0) {
    var minD = UNIQUE_DATES[0];
    var maxD = UNIQUE_DATES[UNIQUE_DATES.length - 1];
    var fmtDate = function(d) { return d.slice(0,4) + '-' + d.slice(4,6) + '-' + d.slice(6,8); };
    document.getElementById('date-from').min = fmtDate(minD);
    document.getElementById('date-from').max = fmtDate(maxD);
    document.getElementById('date-to').min = fmtDate(minD);
    document.getElementById('date-to').max = fmtDate(maxD);
}

// Base colors and sizes — used to reset after filtering
var baseOpacity = [];
var baseColors = [];
var baseSizes = [];
for (var i = 0; i < X.length; i++) {
    baseOpacity.push(0.7);
    baseColors.push(NORMS[i]);
    baseSizes.push(4);
}

var trace = {
    x: X, y: Y,
    mode: 'markers',
    type: 'scattergl',
    marker: {
        size: baseSizes.slice(),
        color: baseColors.slice(),
        colorscale: 'Plasma',
        opacity: baseOpacity.slice(),
        line: { width: 0 }
    },
    text: LABELS,
    hoverinfo: 'none',
    customdata: THUMBS
};

var layout = {
    paper_bgcolor: '#0f0f1a',
    plot_bgcolor: '#0f0f1a',
    margin: { l: 0, r: 0, t: 50, b: 0 },
    xaxis: { visible: false },
    yaxis: { visible: false },
    hovermode: 'closest',
    dragmode: 'pan'
};

var config = {
    scrollZoom: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['autoScale2d'],
    displaylogo: false
};

Plotly.newPlot('plot', [trace], layout, config);

function setMode(mode) {
    Plotly.relayout('plot', { dragmode: mode });
    document.querySelectorAll('.mode-btn').forEach(function(btn) { btn.classList.remove('active'); });
    if (mode === 'pan') document.getElementById('btn-pan').classList.add('active');
    else if (mode === 'select') document.getElementById('btn-box').classList.add('active');
    else if (mode === 'lasso') document.getElementById('btn-lasso').classList.add('active');
}

function resetZoom() {
    Plotly.relayout('plot', { 'xaxis.autorange': true, 'yaxis.autorange': true });
}

var hoverCard = document.getElementById('hover-card');
var hoverImg = document.getElementById('hover-img');
var hoverLabel = document.getElementById('hover-label');
var plotDiv = document.getElementById('plot');
var selPanel = document.getElementById('selection-panel');
var selNum = document.getElementById('sel-num');
var selThumbs = document.getElementById('sel-thumbs');
var currentSelection = [];

function clearSelection() {
    Plotly.restyle('plot', { selectedpoints: [null] }, [0]);
    Plotly.relayout('plot', { selections: [] });
    selPanel.style.display = 'none';
    currentSelection = [];
    setMode('pan');
}

// ---- DATE FILTERING ----
function applyDateFilter() {
    var fromVal = document.getElementById('date-from').value.replace(/-/g, '');
    var toVal = document.getElementById('date-to').value.replace(/-/g, '');

    if (!fromVal && !toVal) {
        clearDateFilter();
        return;
    }

    var newOpacity = [];
    var newColors = [];
    var newSizes = [];
    var matchCount = 0;

    for (var i = 0; i < DATES.length; i++) {
        var d = DATES[i];
        var inRange = true;
        if (!d) {
            inRange = false;  // no date parsed = dim
        } else {
            if (fromVal && d < fromVal) inRange = false;
            if (toVal && d > toVal) inRange = false;
        }

        if (inRange) {
            newOpacity.push(0.9);
            newColors.push('#e05c5c');
            newSizes.push(6);
            matchCount++;
        } else {
            newOpacity.push(0.08);
            newColors.push('#333344');
            newSizes.push(3);
        }
    }

    Plotly.restyle('plot', {
        'marker.opacity': [newOpacity],
        'marker.color': [newColors],
        'marker.size': [newSizes],
        'marker.colorscale': null
    }, [0]);

    document.getElementById('date-info').textContent = matchCount + ' in range';
}

function clearDateFilter() {
    document.getElementById('date-from').value = '';
    document.getElementById('date-to').value = '';
    document.getElementById('date-info').textContent = '';

    Plotly.restyle('plot', {
        'marker.opacity': [baseOpacity],
        'marker.color': [baseColors],
        'marker.size': [baseSizes],
        'marker.colorscale': [['Plasma']]
    }, [0]);
}

// ---- SELECTION ----
plotDiv.on('plotly_selected', function(data) {
    if (!data || !data.points || data.points.length === 0) {
        selPanel.style.display = 'none';
        currentSelection = [];
        return;
    }

    currentSelection = data.points.map(function(pt) {
        return { index: pt.pointIndex, label: pt.text, b64: pt.customdata };
    });

    selNum.textContent = currentSelection.length;

    selThumbs.innerHTML = '';
    var showCount = Math.min(currentSelection.length, 30);
    for (var i = 0; i < showCount; i++) {
        var b64 = currentSelection[i].b64;
        if (b64) {
            var img = document.createElement('img');
            img.src = 'data:image/jpeg;base64,' + b64;
            img.title = currentSelection[i].label;
            selThumbs.appendChild(img);
        }
    }
    if (currentSelection.length > 30) {
        var more = document.createElement('span');
        more.style.cssText = 'color:#888;font-size:11px;align-self:center;padding:0 8px;';
        more.textContent = '+' + (currentSelection.length - 30) + ' more';
        selThumbs.appendChild(more);
    }

    selPanel.style.display = 'block';
});

plotDiv.on('plotly_deselect', function() {
    selPanel.style.display = 'none';
    currentSelection = [];
});

function exportSelection() {
    if (currentSelection.length === 0) return;
    var rows = [['image_path']];
    currentSelection.forEach(function(pt) {
        rows.push([pt.label]);
    });
    var csv = rows.map(function(r) {
        return r.map(function(v) { return '"' + v.replace(/"/g, '""') + '"'; }).join(',');
    }).join('\\n');
    var blob = new Blob([csv], { type: 'text/csv' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'umap_selection.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ---- HOVER ----
plotDiv.on('plotly_hover', function(data) {
    var pt = data.points[0];
    var b64 = pt.customdata;
    if (!b64) {
        hoverCard.style.display = 'none';
        return;
    }

    hoverImg.src = 'data:image/jpeg;base64,' + b64;
    hoverLabel.textContent = pt.text;

    hoverCard.style.display = 'block';

    var evt = data.event;
    var vw = window.innerWidth;
    var vh = window.innerHeight;
    var cardW = 330;
    var cardH = 390;
    var gap = 20;

    var left, top;
    if (evt.clientX > vw / 2) {
        left = evt.clientX - cardW - gap;
    } else {
        left = evt.clientX + gap;
    }
    if (evt.clientY > vh / 2) {
        top = evt.clientY - cardH - gap;
    } else {
        top = evt.clientY + gap;
    }

    left = Math.max(4, Math.min(left, vw - cardW - 4));
    top = Math.max(4, Math.min(top, vh - cardH - 4));

    hoverCard.style.left = left + 'px';
    hoverCard.style.top = top + 'px';
});

plotDiv.on('plotly_unhover', function() {
    hoverCard.style.display = 'none';
});
</script>
</body>
</html>"""

    html = html.replace("__X_VALS__", json.dumps(x_vals))
    html = html.replace("__Y_VALS__", json.dumps(y_vals))
    html = html.replace("__NORMS__", json.dumps(norms))
    html = html.replace("__LABELS__", json.dumps(hover_texts))
    html = html.replace("__THUMBS__", json.dumps(thumb_b64_list))
    html = html.replace("__DATES__", json.dumps(dates))
    html = html.replace("__UNIQUE_DATES__", json.dumps(unique_dates))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved to: {output_path}")
    print(f"  file size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", default=r"C:\Users\jjs\Desktop\embedding program\poultry_embeddings\embeddings\embeddings_EmptyShackle")
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--output", default="umap_report.html",
        help="Output HTML file path")
    parser.add_argument("--thumbnail-size", type=int, default=270,
        help="Thumbnail size in pixels (default: 270). Smaller = smaller file.")
    args = parser.parse_args()

    vectors, filenames = load_embeddings(args.embeddings_dir)
    coords = run_umap(vectors, args.n_neighbors, args.min_dist)
    generate_html(coords, filenames, args.output, thumbnail_size=args.thumbnail_size)