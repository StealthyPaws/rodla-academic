/* ============================================
   90s RETRO RODLA FRONTEND JAVASCRIPT - DEMO MODE
   Falls back to demo data if backend unavailable
   ============================================ */

// Configuration
const API_BASE_URL = 'http://localhost:8000/api';
let currentMode = 'standard';
let currentFile = null;
let lastResults = null;
let demoMode = false;

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('[RODLA] System initialized...');
    setupEventListeners();
    checkBackendStatus();
});

// ============================================
// EVENT LISTENERS
// ============================================

function setupEventListeners() {
    // File upload
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFileSelect(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Mode buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;
            
            // Toggle perturbation options
            const pertOptions = document.getElementById('perturbationOptions');
            if (currentMode === 'perturbation') {
                pertOptions.style.display = 'block';
            } else {
                pertOptions.style.display = 'none';
            }
        });
    });

    // Confidence threshold
    document.getElementById('confidenceThreshold').addEventListener('input', (e) => {
        document.getElementById('thresholdValue').textContent = e.target.value;
    });

    // Buttons
    document.getElementById('analyzeBtn').addEventListener('click', handleAnalysis);
    document.getElementById('resetBtn').addEventListener('click', handleReset);
    document.getElementById('dismissErrorBtn').addEventListener('click', hideError);
    document.getElementById('downloadImageBtn').addEventListener('click', downloadImage);
    document.getElementById('downloadJsonBtn').addEventListener('click', downloadJson);
    document.getElementById('generatePerturbationsBtn')?.addEventListener('click', handleGeneratePerturbations);
}

// ============================================
// FILE HANDLING
// ============================================

function handleFileSelect(file) {
    // Validate file
    if (!file.type.startsWith('image/')) {
        showError('Invalid file type. Please upload an image.');
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        showError('File too large. Maximum size is 50MB.');
        return;
    }

    currentFile = file;
    showPreview(file);
    document.getElementById('analyzeBtn').disabled = false;
}

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewContainer = document.getElementById('previewContainer');
        const previewImage = document.getElementById('previewImage');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');

        previewImage.src = e.target.result;
        fileName.textContent = `Filename: ${file.name}`;
        fileSize.textContent = `Size: ${(file.size / 1024).toFixed(2)} KB`;
        previewContainer.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// ============================================
// ANALYSIS
// ============================================

async function handleAnalysis() {
    if (!currentFile) {
        showError('Please select an image first.');
        return;
    }

    const analysisType = currentMode === 'standard' ? 'Standard Detection' : 'Perturbation Analysis';
    updateStatus(`> INITIATING ${analysisType.toUpperCase()}...`);
    showStatus();
    hideError();

    try {
        const startTime = Date.now();
        const results = await runAnalysis();
        const processingTime = Date.now() - startTime;

        lastResults = {
            ...results,
            processingTime: processingTime,
            timestamp: new Date().toISOString(),
            mode: currentMode,
            fileName: currentFile.name
        };

        displayResults(results, processingTime);
        hideStatus();
    } catch (error) {
        console.error('[ERROR]', error);
        showError(`Analysis failed: ${error.message}`);
        hideStatus();
    }
}

async function handleAnalysis() {
    if (!currentFile) {
        showError('Please select an image first.');
        return;
    }

    const analysisType = currentMode === 'standard' ? 'Standard Detection' : 'Perturbation Analysis';
    updateStatus(`> INITIATING ${analysisType.toUpperCase()}...`);
    showStatus();
    hideError();

    try {
        const startTime = Date.now();
        let results;
        
        if (demoMode) {
            results = generateDemoResults();
            await new Promise(r => setTimeout(r, 2000)); // Simulate processing
        } else {
            results = await runAnalysis();
        }
        
        const processingTime = Date.now() - startTime;

        lastResults = {
            ...results,
            processingTime: processingTime,
            timestamp: new Date().toISOString(),
            mode: currentMode,
            fileName: currentFile.name
        };

        displayResults(results, processingTime);
        hideStatus();
    } catch (error) {
        console.error('[ERROR]', error);
        showError(`Analysis failed: ${error.message}`);
        hideStatus();
    }
}

async function runAnalysis() {
    const formData = new FormData();
    formData.append('file', currentFile);
    
    const threshold = parseFloat(document.getElementById('confidenceThreshold').value);
    formData.append('score_threshold', threshold);

    if (currentMode === 'perturbation') {
        // Get selected perturbation types
        const perturbationTypes = [];
        document.querySelectorAll('.checkbox-label input[type="checkbox"]:checked').forEach(checkbox => {
            perturbationTypes.push(checkbox.value);
        });

        if (perturbationTypes.length === 0) {
            throw new Error('Please select at least one perturbation type.');
        }

        formData.append('perturbation_types', perturbationTypes.join(','));
        
        updateStatus('> APPLYING PERTURBATIONS...');
        return await fetch(`${API_BASE_URL}/detect-with-perturbation`, {
            method: 'POST',
            body: formData
        }).then(r => {
            if (!r.ok) throw new Error(`API Error: ${r.status}`);
            return r.json();
        });
    } else {
        updateStatus('> RUNNING STANDARD DETECTION...');
        return await fetch(`${API_BASE_URL}/detect`, {
            method: 'POST',
            body: formData
        }).then(r => {
            if (!r.ok) throw new Error(`API Error: ${r.status}`);
            return r.json();
        });
    }
}

// ============================================
// PERTURBATIONS GENERATION
// ============================================

async function handleGeneratePerturbations() {
    if (!currentFile) {
        showError('Please select an image first.');
        return;
    }

    updateStatus('> GENERATING ALL 12 PERTURBATIONS (3 DEGREES EACH)...');
    showStatus();
    hideError();

    try {
        const formData = new FormData();
        formData.append('file', currentFile);

        updateStatus('> REQUESTING PERTURBATION GRID FROM BACKEND... ▌▐');

        const response = await fetch(`${API_BASE_URL}/generate-perturbations`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }

        const results = await response.json();
        
        if (!results.success) {
            throw new Error(results.message || 'Failed to generate perturbations');
        }
        
        displayPerturbations(results);
        hideStatus();

    } catch (error) {
        console.error('[ERROR]', error);
        showError(`Failed to generate perturbations: ${error.message}`);
        hideStatus();
    }
}

function displayPerturbations(results) {
    const container = document.getElementById('perturbationsPreviewContainer');
    const section = document.getElementById('perturbationsPreviewSection');
    
    // Update section title with grid info
    const titleElement = section.querySelector('.section-title') || section.parentElement.querySelector('.section-title');
    if (titleElement) {
        titleElement.textContent = `[::] PERTURBATION GRID: 12 TYPES × 3 DEGREES [::]`;
    }
    
    let html = `<div style="font-size: 0.9em; color: #00FFFF; margin-bottom: 15px; padding: 10px; border: 1px dashed #00FFFF;">
        TOTAL: 12 Perturbation Types × 3 Degree Levels (1=Mild, 2=Moderate, 3=Severe)
    </div>`;

    // Add original
    html += `
        <div class="perturbation-grid-section">
            <div class="perturbation-type-label">[ORIGINAL IMAGE]</div>
            <div style="padding: 10px;">
                <img src="data:image/png;base64,${results.perturbations.original.original}" 
                     alt="Original" class="perturbation-preview-image" style="width: 200px; height: auto;">
            </div>
        </div>
    `;

    // Group by perturbation type
    const perturbationTypes = [
        "defocus", "vibration", "speckle", "texture",
        "watermark", "background", "ink_holdout", "ink_bleeding",
        "illumination", "rotation", "keystoning", "warping"
    ];
    
    const categories = {
        "blur": ["defocus", "vibration"],
        "noise": ["speckle", "texture"],
        "content": ["watermark", "background"],
        "inconsistency": ["ink_holdout", "ink_bleeding", "illumination"],
        "spatial": ["rotation", "keystoning", "warping"]
    };

    // Display by category
    Object.entries(categories).forEach(([catName, types]) => {
        html += `<div style="margin-top: 20px; padding: 10px; border-top: 2px solid #008080;">
            <div style="color: #00FF00; font-weight: bold; margin-bottom: 10px;">▼ ${catName.toUpperCase()} ▼</div>`;
        
        types.forEach(ptype => {
            if (results.perturbations[ptype]) {
                html += `<div class="perturbation-type-group" style="margin-bottom: 15px;">
                    <div class="perturbation-type-label" style="margin-bottom: 8px;">${ptype.toUpperCase()}</div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">`;
                
                // Three degree levels
                for (let degree = 1; degree <= 3; degree++) {
                    const degreeKey = `degree_${degree}`;
                    const degreeLabel = ['MILD', 'MODERATE', 'SEVERE'][degree - 1];
                    
                    if (results.perturbations[ptype][degreeKey]) {
                        html += `
                            <div style="text-align: center;">
                                <div style="color: #00FFFF; font-size: 0.8em; margin-bottom: 5px;">DEG ${degree}: ${degreeLabel}</div>
                                <img src="data:image/png;base64,${results.perturbations[ptype][degreeKey]}" 
                                     alt="${ptype} degree ${degree}" 
                                     class="perturbation-preview-image"
                                     style="width: 150px; height: auto; border: 1px solid #008080; padding: 2px;">
                            </div>
                        `;
                    }
                }
                
                html += `</div></div>`;
            }
        });
        
        html += `</div>`;
    });

    container.innerHTML = html;
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth' });
}

// ============================================


function displayResults(results, processingTime) {
    updateStatus(`> DISPLAYING RESULTS... [${processingTime}ms]`);

    // Update stats
    const detections = results.detections || [];
    const confidences = detections.map(d => d.confidence || 0);
    const avgConfidence = confidences.length > 0 
        ? (confidences.reduce((a, b) => a + b) / confidences.length * 100).toFixed(1)
        : 0;

    document.getElementById('detectionCount').textContent = detections.length;
    document.getElementById('avgConfidence').textContent = `${avgConfidence}%`;
    document.getElementById('processingTime').textContent = `${processingTime}ms`;

    // Display image
    if (results.annotated_image) {
        document.getElementById('resultImage').src = `data:image/png;base64,${results.annotated_image}`;
    }

    // Class distribution
    displayClassDistribution(results.class_distribution || {});

    // Detection table
    displayDetectionsTable(detections);

    // Metrics
    displayMetrics(results.metrics || {});

    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function displayClassDistribution(distribution) {
    const chart = document.getElementById('classChart');
    
    if (Object.keys(distribution).length === 0) {
        chart.innerHTML = '<p class="no-data">No class distribution data</p>';
        return;
    }

    const maxCount = Math.max(...Object.values(distribution));
    let html = '';

    Object.entries(distribution).forEach(([className, count]) => {
        const percentage = (count / maxCount) * 100;
        html += `
            <div class="chart-item">
                <div class="chart-label">${className}</div>
                <div class="chart-bar-container">
                    <div class="chart-bar" style="width: ${percentage}%;">
                        <span class="chart-count">${count}</span>
                    </div>
                </div>
            </div>
        `;
    });

    chart.innerHTML = html;
}

function displayDetectionsTable(detections) {
    const tbody = document.getElementById('detectionsTableBody');
    
    if (detections.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="no-data">NO DETECTIONS</td></tr>';
        return;
    }

    let html = '';
    detections.slice(0, 50).forEach((det, idx) => {
        const box = det.box || {};
        const x1 = box.x1 ? box.x1.toFixed(0) : '?';
        const y1 = box.y1 ? box.y1.toFixed(0) : '?';
        const x2 = box.x2 ? box.x2.toFixed(0) : '?';
        const y2 = box.y2 ? box.y2.toFixed(0) : '?';
        
        html += `
            <tr>
                <td>${idx + 1}</td>
                <td>${det.class || 'Unknown'}</td>
                <td>${(det.confidence * 100).toFixed(1)}%</td>
                <td>[${x1},${y1},${x2},${y2}]</td>
            </tr>
        `;
    });

    if (detections.length > 50) {
        html += `<tr><td colspan="4" class="no-data">... and ${detections.length - 50} more</td></tr>`;
    }

    tbody.innerHTML = html;
}

function displayMetrics(metrics) {
    const metricsBox = document.getElementById('metricsBox');
    
    if (Object.keys(metrics).length === 0) {
        metricsBox.innerHTML = '<p class="no-data">No metrics available</p>';
        return;
    }

    let html = '';
    Object.entries(metrics).forEach(([key, value]) => {
        const displayValue = typeof value === 'number' ? value.toFixed(3) : value;
        html += `
            <div class="metric-line">
                <span class="metric-label">${key}:</span>
                <span class="metric-value">${displayValue}</span>
            </div>
        `;
    });

    metricsBox.innerHTML = html;
}

// ============================================
// UI HELPERS
// ============================================

function updateStatus(message) {
    document.getElementById('statusText').textContent = message;
}

function showStatus() {
    document.getElementById('statusSection').style.display = 'block';
    document.getElementById('statusSection').scrollIntoView({ behavior: 'smooth' });
}

function hideStatus() {
    document.getElementById('statusSection').style.display = 'none';
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorSection').style.display = 'block';
    document.getElementById('errorSection').scrollIntoView({ behavior: 'smooth' });
}

function hideError() {
    document.getElementById('errorSection').style.display = 'none';
}

function handleReset() {
    currentFile = null;
    lastResults = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('previewContainer').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('statusSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';
    document.getElementById('analyzeBtn').disabled = true;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================
// DOWNLOADS
// ============================================

function downloadImage() {
    if (!lastResults || !lastResults.annotated_image) {
        showError('No image to download');
        return;
    }

    const link = document.createElement('a');
    link.href = `data:image/png;base64,${lastResults.annotated_image}`;
    link.download = `rodla-result-${Date.now()}.png`;
    link.click();
}

function downloadJson() {
    if (!lastResults) {
        showError('No results to download');
        return;
    }

    const jsonData = {
        timestamp: lastResults.timestamp,
        fileName: lastResults.fileName,
        mode: lastResults.mode,
        processingTime: lastResults.processingTime,
        detections: lastResults.detections,
        metrics: lastResults.metrics,
        classDistribution: lastResults.class_distribution
    };

    const link = document.createElement('a');
    link.href = `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(jsonData, null, 2))}`;
    link.download = `rodla-result-${Date.now()}.json`;
    link.click();
}

// ============================================
// DEMO MODE - Generate sample results
// ============================================

function generateDemoResults() {
    const classes = ['Title', 'Text', 'Figure', 'Table', 'Header', 'Footer'];
    const detectionCount = Math.floor(Math.random() * 15) + 5;
    const detections = [];

    for (let i = 0; i < detectionCount; i++) {
        detections.push({
            class: classes[Math.floor(Math.random() * classes.length)],
            confidence: Math.random() * 0.5 + 0.5,
            box: {
                x1: Math.floor(Math.random() * 500),
                y1: Math.floor(Math.random() * 500),
                x2: Math.floor(Math.random() * 500 + 200),
                y2: Math.floor(Math.random() * 500 + 200)
            }
        });
    }

    const distribution = {};
    classes.forEach(cls => {
        distribution[cls] = Math.floor(Math.random() * detectionCount);
    });

    // Create a simple demo image (black canvas with green boxes)
    const canvas = document.createElement('canvas');
    canvas.width = 800;
    canvas.height = 600;
    const ctx = canvas.getContext('2d');
    
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, 800, 600);
    
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 2;
    
    // Draw demo boxes
    detections.forEach((det, idx) => {
        ctx.strokeRect(det.box.x1, det.box.y1, det.box.x2 - det.box.x1, det.box.y2 - det.box.y1);
        ctx.fillStyle = '#00FF00';
        ctx.font = '12px Courier New';
        ctx.fillText(`${det.class} ${(det.confidence * 100).toFixed(0)}%`, det.box.x1, det.box.y1 - 5);
    });

    const imageData = canvas.toDataURL('image/png').split(',')[1];

    return {
        detections: detections,
        class_distribution: distribution,
        annotated_image: imageData,
        metrics: {
            'Total Detections': detections.length,
            'Average Confidence': (detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length).toFixed(3),
            'Processing Mode': currentMode === 'standard' ? 'Standard' : 'Perturbation',
            'Image Size': `${800}x${600}`
        }
    };
}

// ============================================
// BACKEND STATUS CHECK
// ============================================

async function checkBackendStatus() {
    try {
        console.log('[RODLA] Checking backend connection...');
        const response = await fetch(`${API_BASE_URL}/model-info`, { 
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });
        
        if (response.ok) {
            demoMode = false;
            console.log('[RODLA] Backend connection: OK');
            console.log('[RODLA] Using live backend');
        } else {
            throw new Error('Backend responded with error');
        }
    } catch (error) {
        console.warn('[RODLA] Backend not available:', error.message);
        console.log('[RODLA] Switching to DEMO MODE - showing sample results');
        demoMode = true;
        
        // Update status indicator in UI
        const statusElement = document.querySelector('.status-online');
        if (statusElement) {
            statusElement.textContent = '● DEMO MODE';
            statusElement.style.color = '#FFFF00'; // Yellow for demo
        }
    }
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

console.log('[RODLA] Frontend loaded successfully. Ready for analysis.');
console.log('[RODLA] Demo mode available if backend is unavailable.');
