<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>RoDLA — 90s Frontend README</title>
  <link href="https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root{
      --bg:#000; --panel:#0b2b2b; --text:#7cff7c; --muted:#9bd1d1; --accent:#00ffff; --error:#ff4d4d;
      --card-bg: rgba(255,255,255,0.02);
    }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-family:'Courier Prime', 'IBM Plex Mono', monospace;line-height:1.45;padding:28px}

    .container{max-width:1100px;margin:0 auto;display:grid;grid-template-columns:1fr 320px;gap:20px}
    header{grid-column:1/-1;display:flex;align-items:center;gap:16px}

    .title-box{border:3px solid var(--accent);padding:12px 14px;border-radius:6px;background:linear-gradient(180deg, rgba(0,255,255,0.02), transparent)}
    h1{margin:0;font-size:20px;color:var(--text)}
    h2,h3{color:var(--accent);margin:10px 0}
    p.lead{color:var(--muted);margin:8px 0}

    nav{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}
    .nav-btn{border:1px solid var(--accent);padding:6px 8px;border-radius:6px;color:var(--text);text-decoration:none;font-size:13px}

    main{background:var(--card-bg);padding:18px;border-radius:8px;border:1px solid rgba(255,255,255,0.03)}
    .sidebar{position:sticky;top:28px;background:var(--card-bg);padding:14px;border-radius:8px;border:1px solid rgba(255,255,255,0.03)}

    .section{margin-bottom:14px}
    pre.code{background:#001515;color:var(--text);padding:12px;border-radius:6px;overflow:auto;border:1px dashed rgba(0,255,255,0.08)}
    code.inline{background:#001b1b;padding:2px 6px;border-radius:4px;color:var(--accent)}

    table{width:100%;border-collapse:collapse;margin-top:8px}
    th,td{border:1px solid rgba(255,255,255,0.03);padding:8px;text-align:left;color:var(--muted)}
    th{color:var(--accent)}

    .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .card{background:linear-gradient(180deg, rgba(255,255,255,0.01), transparent);padding:12px;border-radius:6px;border:1px solid rgba(255,255,255,0.02)}

    .check{color:var(--text)}
    .footer{grid-column:1/-1;margin-top:8px;color:var(--muted);font-size:13px;text-align:center}

    /* retro CRT scanlines */
    .crt{position:relative}
    .crt:after{content:'';position:absolute;left:0;right:0;top:0;bottom:0;background-image:linear-gradient(transparent 50%, rgba(0,0,0,0.06) 50%);background-size:100% 4px;pointer-events:none;border-radius:6px}

    @media (max-width:980px){.container{grid-template-columns:1fr}.sidebar{order:2}}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="title-box">
        <h1>🎮 RoDLA 90s Frontend — Complete Project Documentation</h1>
        <div style="font-size:12px;color:var(--muted)">Retro terminal UI • CVPR 2024 • v2.1.0 (90s Edition)</div>
      </div>

      <nav>
        <a class="nav-btn" href="#analysis">Analysis</a>
        <a class="nav-btn" href="#frontend">Frontend</a>
        <a class="nav-btn" href="#structure">Project Structure</a>
        <a class="nav-btn" href="#quickstart">Quick Start</a>
        <a class="nav-btn" href="#api">API</a>
      </nav>
    </header>

    <main>
      <section id="analysis" class="section crt">
        <h2>📊 Project Analysis Summary</h2>
        <p class="lead">RoDLA (Robust Document Layout Analysis) benchmarks how well layout detectors handle realistic perturbations. Large dataset and clear metrics enable rigorous model comparison.</p>

        <div class="grid">
          <div class="card">
            <h3>What is RoDLA?</h3>
            <p class="muted">A robustness benchmark for document layout analysis: detects text, tables, figures, headers, footers and measures degradation under perturbations.</p>
            <ul style="color:var(--muted)">
              <li>mAP (clean): <strong>70.0</strong></li>
              <li>mAP (perturbed avg): <strong>61.7</strong></li>
              <li>mRD Score: <strong>147.6</strong></li>
              <li>Backbone: <strong>InternImage-XL + DINO</strong></li>
            </ul>
          </div>

          <div class="card">
            <h3>Key Features</h3>
            <ol style="color:var(--muted)">
              <li>Blur / Noise</li>
              <li>Rotation / Scaling</li>
              <li>Perspective / Photometric</li>
              <li>Large-scale dataset (450k+ docs)</li>
            </ol>
          </div>
        </div>

        <div style="display:flex;gap:12px;margin-top:12px">
          <div class="card" style="flex:1;text-align:center">
            <div style="font-size:20px;color:var(--text)">70.0</div>
            <div style="color:var(--muted)">mAP (clean)</div>
          </div>
          <div class="card" style="flex:1;text-align:center">
            <div style="font-size:20px;color:var(--text)">61.7</div>
            <div style="color:var(--muted)">mAP (perturbed)</div>
          </div>
          <div class="card" style="flex:1;text-align:center">
            <div style="font-size:20px;color:var(--text)">147.6</div>
            <div style="color:var(--muted)">mRD</div>
          </div>
        </div>

        <h3 style="margin-top:14px">System Architecture</h3>
        <pre class="code">┌─────────────────────────────────────────────────────────────┐
│                    RoDLA System (90s Edition)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐              ┌──────────────────┐     │
│  │   Frontend       │  (HTTP)      │   Backend        │     │
│  │  90s Terminal    │──────────────│   FastAPI        │     │
│  │  Port: 8080      │  (JSON/Image)│   Port: 8000     │     │
│  └──────────────────┘              └──────────────────┘     │
│         │                                    │              │
│         │                                    ▼              │
│         │                          ┌──────────────────┐     │
│         │                          │   PyTorch Model  │     │
│         │                          │   InternImage-XL │     │
│         │                          └──────────────────┘     │
│         │                                    │              │
│         └────────────────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘</pre>

      </section>

      <section id="frontend" class="section">
        <h2>🎨 Frontend Design (90s)</h2>

        <div class="card">
          <h3>Color Scheme</h3>
          <ul style="color:var(--muted)">
            <li><strong>Primary:</strong> Teal (#008080)</li>
            <li><strong>Text:</strong> Lime Green (#00FF00)</li>
            <li><strong>Accent:</strong> Cyan (#00FFFF)</li>
            <li><strong>Background:</strong> Black (#000000)</li>
            <li><strong>Error:</strong> Red (#FF0000)</li>
            <li><strong>No gradients:</strong> Flat 90s style (except subtle panel textures)</li>
          </ul>

          <h3>Design Elements</h3>
          <p class="muted">CRT scanlines, blinking status, Windows 95 borders, monospace fonts, MS Sans-inspired UI and a terminal-like interaction model.</p>

          <h3>Responsive Breakpoints</h3>
          <p class="muted">Desktop (full), Tablet (≥768px) grid adjustments, Mobile (&lt;768px) stacked single-column.</p>
        </div>
      </section>

      <section id="structure" class="section">
        <h2>📁 Project Structure</h2>
        <pre class="code">rodla-academic/
├── SETUP_GUIDE.md
├── PROJECT_ANALYSIS.md
├── start.sh
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   ├── server.py
│   └── README.md
├── deployment/
│   └── backend/
│       ├── backend.py
│       ├── config/
│       │   └── settings.py
│       ├── api/
│       │   ├── routes.py
│       │   └── schemas.py
│       ├── core/
│       ├── services/
│       ├── perturbations/
│       └── tests/
├── model/
│   ├── configs/
│   ├── ops_dcnv3/
│   └── train.py
└── perturbation/
    └── *.py</pre>
      </section>

      <section id="datasets" class="section">
        <h2>Datasets</h2>
        <div class="card">
          <h3>Training</h3>
          <p class="muted">Download RoDLA-related and related pretraining datasets.</p>
          <ul style="color:var(--muted)">
            <li><a class="nav-btn" href="https://drive.google.com/file/d/1bfjaxb5fAjU7sFqtM3GfNYm0ynrB5Vwo/view?usp=drive_link">PubLayNet-P</a></li>
          </ul>

          <h3>Finetuning</h3>
          <p class="muted">DocBank/DockBank links.</p>
          <ul style="color:var(--muted)">
            <li><a class="nav-btn" href="https://drive.google.com/file/d/1bfjaxb5fAjU7sFqtM3GfNYm0ynrB5Vwo/view?usp=drive_link">DockBank</a></li>
          </ul>
        </div>
      </section>

      <section id="weights" class="section">
        <h2>Weights</h2>
        <div class="card">
          <h3>Training</h3>
          <p class="muted">Pretrained checkpoints on PubLayNet.</p>
          <a class="nav-btn" href="https://drive.google.com/file/d/1I2CafA-wRKAWCqFgXPgtoVx3OQcRWEjp/view?usp=sharing">Checkpoints for PubLayNet</a>

          <h3 style="margin-top:8px">Finetuning</h3>
          <p class="muted">DocBank finetuned checkpoints.</p>
          <a class="nav-btn" href="https://drive.google.com/file/d/1BHyz2jH52Irt6izCeTRb4g2J5lXsA9cz/view?usp=drive_link">Checkpoints for DocBank</a>
        </div>
      </section>

      <section id="quickstart" class="section">
        <h2>🚀 Quick Start</h2>

        <h3>Option 1: Automated Startup (Recommended)</h3>
        <pre class="code">cd /home/admin/CV/rodla-academic
./start.sh</pre>
        <p class="muted">Starts backend (8000) and frontend (8080) and prints access points.</p>

        <h3>Option 2: Manual Startup</h3>
        <pre class="code"># Backend
cd deployment/backend
python backend.py

# Frontend
cd frontend
python3 server.py

# Browser
Open: http://localhost:8080</pre>

        <h3>Option 3: Alternative HTTP Servers</h3>
        <pre class="code"># http.server
python3 -m http.server 8080

# npx http-server
npx http-server -p 8080 -c-1

# php built-in
php -S localhost:8080</pre>
      </section>

      <section id="ui" class="section">
        <h2>🎮 User Interface Guide</h2>
        <div class="card">
          <h3>Main Sections</h3>
          <ol style="color:var(--muted)">
            <li>Header — branding, version, status</li>
            <li>Upload — drag & drop, preview</li>
            <li>Analysis Options — threshold, mode, perturbations</li>
            <li>Action Buttons — Analyze, Clear All</li>
            <li>Status — progress + blinking</li>
            <li>Results — annotated image, stats, table, metrics, downloads</li>
          </ol>

          <h3>Workflow Example</h3>
          <pre class="code">1. Upload image
2. Configure options
3. Click Analyze
4. Review results
5. Download (PNG / JSON)
6. Clear and repeat</pre>
        </div>
      </section>

      <section id="api" class="section">
        <h2>🔌 API Integration</h2>
        <div class="card">
          <h3>Backend Endpoints</h3>
          <table>
            <tr><th>Method</th><th>Endpoint</th><th>Purpose</th></tr>
            <tr><td>GET</td><td>/api/health</td><td>Health check</td></tr>
            <tr><td>GET</td><td>/api/model-info</td><td>Model information</td></tr>
            <tr><td>POST</td><td>/api/detect</td><td>Standard detection</td></tr>
            <tr><td>GET</td><td>/api/perturbations/info</td><td>Perturbation info</td></tr>
            <tr><td>POST</td><td>/api/detect-with-perturbation</td><td>Detection + perturbation</td></tr>
            <tr><td>POST</td><td>/api/batch</td><td>Batch processing</td></tr>
          </table>

          <h3 style="margin-top:10px">Request / Response</h3>
          <pre class="code">Request (multipart/form-data):
{ file: image_file, score_threshold: 0.3 }

Response (JSON):
{
  "detections": [ { "class": "Text", "confidence": 0.95, "box": {"x1":10, "y1":20, "x2":100, "y2":200} } ],
  "class_distribution": {"Text":5, "Table":2},
  "annotated_image": "base64...",
  "metrics": { }
}</pre>
        </div>
      </section>

      <section id="features" class="section">
        <h2>💡 Features</h2>
        <div class="grid">
          <div class="card">
            <h4>Standard Detection</h4>
            <p class="muted">Real-time detection, bounding boxes, confidence, class labels.</p>
          </div>
          <div class="card">
            <h4>Perturbation Analysis</h4>
            <p class="muted">Apply multiple perturbations, benchmark degradation, compare clean vs perturbed.</p>
          </div>
        </div>
      </section>

      <section id="demo" class="section">
        <h2>🎯 Demo Mode</h2>
        <div class="card">
          <p class="muted">If backend unavailable, frontend switches to Demo Mode. Generates realistic sample output, no network required.</p>
          <p class="muted">Status indicator: <code class="inline">● DEMO MODE</code> (Yellow)</p>
        </div>
      </section>

      <section id="config" class="section">
        <h2>⚙️ Configuration</h2>
        <div class="card">
          <h4>Backend (deployment/backend/config/settings.py)</h4>
          <pre class="code">API_HOST = "0.0.0.0"
API_PORT = 8000
DEFAULT_SCORE_THRESHOLD = 0.3
MAX_DETECTIONS_PER_IMAGE = 300</pre>

          <h4>Frontend (frontend/script.js)</h4>
          <pre class="code">const API_BASE_URL = 'http://localhost:8000/api';</pre>

          <h4>Style (frontend/styles.css)</h4>
          <pre class="code">:root{ --primary-color:#008080; --text-color:#00FF00; --accent-color:#00FFFF; --bg-color:#000; }</pre>
        </div>
      </section>

      <section id="performance" class="section">
        <h2>📊 Performance Metrics</h2>
        <table>
          <tr><th>Metric</th><th>Value</th></tr>
          <tr><td>Detection Speed (GPU)</td><td>3-5 seconds/image</td></tr>
          <tr><td>Detection Speed (CPU)</td><td>10-15 seconds/image</td></tr>
          <tr><td>Model mAP (Clean)</td><td>70.0</td></tr>
          <tr><td>Model mAP (Perturbed Avg)</td><td>61.7</td></tr>
          <tr><td>mRD Score</td><td>147.6</td></tr>
          <tr><td>Max Batch Size</td><td>300 images</td></tr>
          <tr><td>Max File Size</td><td>50 MB</td></tr>
        </table>
      </section>

      <section id="troubleshoot" class="section">
        <h2>🐛 Troubleshooting</h2>
        <div class="card">
          <h4>Frontend can't connect</h4>
          <pre class="code">✗ Backend not running
  → Start: cd deployment/backend && python backend.py

✗ Wrong port
  → Check API_BASE_URL in frontend/script.js

✗ CORS error
  → Configure CORS_ORIGINS in settings.py</pre>

          <h4>Analysis slow</h4>
          <pre class="code">✗ Image too large → Resize
✗ CPU only → Install PyTorch with CUDA
✗ Multiple jobs queued → Wait or increase workers</pre>

          <h4>Port already in use</h4>
          <pre class="code">lsof -ti :8000 | xargs kill -9
lsof -ti :8080 | xargs kill -9
# or run front on 8081
python3 -m http.server 8081</pre>
        </div>
      </section>

      <section id="security" class="section">
        <h2>🔒 Security Considerations</h2>
        <div class="card">
          <h4>Frontend</h4>
          <p class="muted">No sensitive data stored locally; processing on backend; downloads are client-side only.</p>

          <h4>Backend</h4>
          <p class="muted">File upload limits, input validation, CORS restrictions, no direct FS access in production.</p>

          <h4>Deployment</h4>
          <p class="muted">Use HTTPS, authentication, rate limiting and file type validation.</p>
        </div>
      </section>

      <section id="browser" class="section">
        <h2>📝 Browser Support</h2>
        <table>
          <tr><th>Browser</th><th>Version</th><th>Status</th></tr>
          <tr><td>Chrome</td><td>90+</td><td>✓ Fully supported</td></tr>
          <tr><td>Firefox</td><td>88+</td><td>✓ Fully supported</td></tr>
          <tr><td>Safari</td><td>14+</td><td>✓ Fully supported</td></tr>
          <tr><td>Edge</td><td>90+</td><td>✓ Fully supported</td></tr>
          <tr><td>IE 11</td><td>-</td><td>✗ Not supported</td></tr>
        </table>
      </section>

      <section id="model" class="section">
        <h2>🎓 Model Details</h2>
        <div class="card">
          <h4>Architecture</h4>
          <p class="muted">Backbone: InternImage-XL • Detection: DINO • Attention: Channel attention + pooling • Pretraining: ImageNet-22K</p>

          <h4>Training Data</h4>
          <p class="muted">PubLayNet-P, DocLayNet-P and 450k+ perturbed documents used for evaluation and finetuning.</p>

          <h4>Classes</h4>
          <p class="muted">Text, Tables, Figures, Headers, Footers, Page numbers, Captions (model-dependent)</p>
        </div>
      </section>

      <section id="deploy" class="section">
        <h2>🚀 Deployment Options</h2>
        <div class="card">
          <h4>Local</h4>
          <pre class="code">./start.sh</pre>

          <h4>Docker</h4>
          <pre class="code">FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000 8080
CMD ["./start.sh"]</pre>

          <h4>Production</h4>
          <p class="muted">Use HTTPS, auth, rate-limiting, WSGI server, logging and monitoring.</p>
        </div>
      </section>

      <section id="refs" class="section">
        <h2>📚 References</h2>
        <ul style="color:var(--muted)">
          <li>Paper: RoDLA: Benchmarking the Robustness of Document Layout Analysis Models (CVPR 2024)</li>
          <li>Frameworks: FastAPI, PyTorch, OpenCV</li>
          <li>Frontend: HTML5, CSS3, Vanilla JS</li>
          <li>License: Apache 2.0</li>
        </ul>
      </section>

      <section id="success" class="section">
        <h2>🎉 Success Indicators</h2>
        <ul style="color:var(--muted)">
          <li class="check">Backend starts without errors</li>
          <li class="check">Frontend loads at <code class="inline">http://localhost:8080</code></li>
          <li class="check">Upload & analyze images</li>
          <li class="check">Download annotated PNG & JSON</li>
        </ul>
      </section>

      <section id="help" class="section">
        <h2>📞 Getting Help</h2>
        <ol style="color:var(--muted)">
          <li>Read project README files</li>
          <li>Check logs: <code class="inline">/tmp/rodla_*.log</code></li>
          <li>Open DevTools (F12)</li>
          <li>API docs: <code class="inline">http://localhost:8000/docs</code></li>
          <li>GitHub Issues</li>
        </ol>
      </section>

      <section id="future" class="section">
        <h2>🎨 Future Enhancements</h2>
        <ul style="color:var(--muted)">
          <li>[ ] Multiple model selection</li>
          <li>[ ] Batch processing UI</li>
          <li>[ ] Real-time preview</li>
          <li>[ ] Export to COCO</li>
          <li>[ ] Database integration</li>
          <li>[ ] WebSocket support</li>
        </ul>
      </section>

      <section id="summary" class="section">
        <h2>🎯 Summary</h2>
        <p class="muted">RoDLA 90s Edition — Retro UI + Complete backend & frontend, demo mode, responsive and production-ready recommendations. Version: <strong>v2.1.0</strong></p>
      </section>

      <div class="footer">Created with ❤️ for retro computing enthusiasts and document analysis professionals.</div>
    </main>

    <aside class="sidebar">
      <div style="font-size:14px;color:var(--accent);margin-bottom:8px">Quick Links</div>
      <div style="display:flex;flex-direction:column;gap:8px">
        <a class="nav-btn" href="https://arxiv.org/pdf/2403.14442.pdf">Paper (arXiv)</a>
        <a class="nav-btn" href="https://yufanchen96.github.io/projects/RoDLA/">Project Homepage</a>
        <a class="nav-btn" href="https://github.com/yufanchen96/RoDLA">GitHub</a>
        <a class="nav-btn" href="#quickstart">Quick Start</a>
      </div>

      <div style="margin-top:12px;padding-top:12px;border-top:1px dashed rgba(255,255,255,0.03);color:var(--muted)">
        <div style="font-weight:700;color:var(--text)">Status</div>
        <div style="margin-top:6px">● ONLINE (or ● DEMO MODE)</div>

        <h4 style="margin-top:12px;color:var(--accent)">Contact</h4>
        <div style="color:var(--muted);font-size:13px">See GitHub issues or project contacts in repository.</div>
      </div>
    </aside>

  </div>
</body>
</html>
