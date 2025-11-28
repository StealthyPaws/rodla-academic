/* ============================================
   90s RETRO RODLA FRONTEND JAVASCRIPT - DEMO MODE
   Falls back to demo data if backend unavailable
   ============================================ */

// Configuration - dynamically detect API base URL
// Works with: localhost:8000, localhost:7860, HuggingFace Spaces, etc.
const getAPIBaseURL = () => {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const port = window.location.port;
    
    // On HuggingFace Spaces or same-origin deployment
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
        // Local development - try both ports
        return `${protocol}//${hostname}:8000/api`;
    } else {
        // HuggingFace Spaces or production - use same origin
        return `${protocol}//${hostname}:${port}/api`;
    }
};

const API_BASE_URL = getAPIBaseURL();
let currentMode = 'standard';
let currentFile = null;
let lastResults = null;
let demoMode = false;

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('[RODLA] System initialized...');
    console.log('[RODLA] API Base URL:', API_BASE_URL);
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
            
            // Toggle perturbation options and hint
            const pertOptions = document.getElementById('perturbationOptions');
            const modeHint = document.getElementById('modeHint');
            const standardModeHint = document.getElementById('standardModeHint');
            const analyzeBtn = document.getElementById('analyzeBtn');
            
            if (currentMode === 'perturbation') {
                // PERTURBATION MODE - allow analysis of original or perturbation images
                pertOptions.style.display = 'block';
                modeHint.style.display = 'block';
                standardModeHint.style.display = 'none';
                analyzeBtn.style.opacity = currentFile ? '1' : '0.5';
                analyzeBtn.style.cursor = currentFile ? 'pointer' : 'not-allowed';
                analyzeBtn.disabled = !currentFile;
                analyzeBtn.title = 'Click to generate perturbations, then click on any image to analyze it';
            } else {
                // STANDARD MODE
                pertOptions.style.display = 'none';
                modeHint.style.display = 'none';
                standardModeHint.style.display = 'block';
                analyzeBtn.style.opacity = currentFile ? '1' : '0.5';
                analyzeBtn.style.cursor = currentFile ? 'pointer' : 'not-allowed';
                analyzeBtn.disabled = !currentFile;
                analyzeBtn.title = 'Click to analyze the document layout';
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
    
    // Enable analyze button only if in standard mode
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (currentMode === 'standard') {
        analyzeBtn.disabled = false;
    }
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
        let results;
        
        if (demoMode) {
            results = generateDemoResults();
            await new Promise(r => setTimeout(r, 2000)); // Simulate processing
        } else {
            results = await runAnalysis();
        }
        
        const processingTime = Date.now() - startTime;

        // Read original image as base64 for annotation
        const originalImageBase64 = await readFileAsBase64(currentFile);

        lastResults = {
            ...results,
            original_image: originalImageBase64,
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

    // Only standard detection mode
    updateStatus('> RUNNING STANDARD DETECTION...');
    return await fetch(`${API_BASE_URL}/detect`, {
        method: 'POST',
        body: formData
    }).then(r => {
        if (!r.ok) throw new Error(`API Error: ${r.status}`);
        return r.json();
    });
}

async function analyzePerturbationImage(imageBase64, perturbationType, degree) {
    // Analyze a specific perturbation image
    updateStatus(`> ANALYZING ${perturbationType.toUpperCase()} (DEGREE ${degree})...`);
    showStatus();
    hideError();

    try {
        const startTime = Date.now();
        
        // Convert base64 to blob and create file
        const binaryString = atob(imageBase64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: 'image/png' });
        const file = new File([blob], `${perturbationType}_degree_${degree}.png`, { type: 'image/png' });
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        const threshold = parseFloat(document.getElementById('confidenceThreshold').value);
        formData.append('score_threshold', threshold);
        
        // Send to backend
        const response = await fetch(`${API_BASE_URL}/detect`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        const results = await response.json();
        const processingTime = Date.now() - startTime;
        
        // Store results with perturbation info
        lastResults = {
            ...results,
            original_image: imageBase64,
            processingTime: processingTime,
            timestamp: new Date().toISOString(),
            mode: 'perturbation',
            perturbation_type: perturbationType,
            perturbation_degree: degree,
            fileName: `${perturbationType}_degree_${degree}.png`
        };
        
        displayResults(results, processingTime);
        hideStatus();
    } catch (error) {
        console.error('[ERROR]', error);
        showError(`Perturbation analysis failed: ${error.message}`);
        hideStatus();
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
        TOTAL: 12 Perturbation Types × 3 Degree Levels (1=Mild, 2=Moderate, 3=Severe) - CLICK ON ANY IMAGE TO ANALYZE
    </div>`;

    // Store all perturbation images for clickable analysis
    const perturbationImages = [];

    // Add original
    perturbationImages.push({
        name: 'original',
        image: results.perturbations.original.original
    });
    
    html += `
        <div class="perturbation-grid-section">
            <div class="perturbation-type-label">[ORIGINAL IMAGE]</div>
            <div style="padding: 10px;">
                <img src="data:image/png;base64,${results.perturbations.original.original}" 
                     alt="Original" class="perturbation-preview-image" 
                     data-perturbation="original" data-degree="0"
                     style="width: 200px; height: auto; cursor: pointer; border: 2px solid transparent; transition: all 0.2s;" 
                     title="Click to analyze this image">
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
                        perturbationImages.push({
                            name: ptype,
                            degree: degree,
                            image: results.perturbations[ptype][degreeKey]
                        });
                        
                        html += `
                            <div style="text-align: center;">
                                <div style="color: #00FFFF; font-size: 0.8em; margin-bottom: 5px;">DEG ${degree}: ${degreeLabel}</div>
                                <img src="data:image/png;base64,${results.perturbations[ptype][degreeKey]}" 
                                     alt="${ptype} degree ${degree}" 
                                     class="perturbation-preview-image"
                                     data-perturbation="${ptype}"
                                     data-degree="${degree}"
                                     style="width: 150px; height: auto; border: 2px solid #008080; padding: 2px; cursor: pointer; transition: all 0.2s;" 
                                     title="Click to analyze this perturbation"
                                     onmouseover="this.style.borderColor='#00FF00'; this.style.boxShadow='0 0 10px #00FF00';"
                                     onmouseout="this.style.borderColor='#008080'; this.style.boxShadow='none';">
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
    
    // Add click handlers to perturbation images
    const perturbationImgs = container.querySelectorAll('[data-perturbation]');
    perturbationImgs.forEach(img => {
        img.addEventListener('click', async function() {
            const perturbationType = this.dataset.perturbation;
            const degree = this.dataset.degree;
            
            // Find the image data
            let imageBase64 = null;
            if (perturbationType === 'original') {
                imageBase64 = results.perturbations.original.original;
            } else {
                const degreeKey = `degree_${degree}`;
                imageBase64 = results.perturbations[perturbationType][degreeKey];
            }
            
            if (!imageBase64) {
                showError('Failed to load image for analysis');
                return;
            }
            
            // Convert base64 to File object and analyze
            await analyzePerturbationImage(imageBase64, perturbationType, degree);
        });
    });
    
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
    document.getElementById('processingTime').textContent = `${processingTime.toFixed(0)}ms`;

    // Draw annotated image with bounding boxes
    if (lastResults && lastResults.original_image) {
        drawAnnotatedImage(lastResults.original_image, detections, results.image_width, results.image_height);
    } else {
        // Fallback: try to use previewImage
        const previewImg = document.getElementById('previewImage');
        if (previewImg && previewImg.src) {
            drawAnnotatedImageFromSrc(previewImg.src, detections, results.image_width, results.image_height);
        }
    }

    // Class distribution
    displayClassDistribution(results.class_distribution || {});

    // Detection table
    displayDetectionsTable(detections);

    // Metrics
    displayMetrics(results, processingTime);

    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function drawAnnotatedImage(imageBase64, detections, imgWidth, imgHeight) {
    // Draw bounding boxes on image and display
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Load image
    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        // Draw bounding boxes
        detections.forEach((det, idx) => {
            const bbox = det.bbox || {};
            
            // Convert normalized coordinates to pixel coordinates
            const x = bbox.x * img.width;
            const y = bbox.y * img.height;
            const w = bbox.width * img.width;
            const h = bbox.height * img.height;
            
            // Draw box
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);
            
            // Draw label
            const label = `${det.class_name || 'Unknown'} (${(det.confidence * 100).toFixed(1)}%)`;
            const fontSize = Math.max(12, Math.min(18, Math.floor(img.height / 30)));
            ctx.font = `bold ${fontSize}px monospace`;
            ctx.fillStyle = '#000000';
            ctx.fillRect(x, y - fontSize - 5, ctx.measureText(label).width + 10, fontSize + 5);
            ctx.fillStyle = '#00FF00';
            ctx.fillText(label, x + 5, y - 5);
        });
        
        // Display canvas as image
        const resultImage = document.getElementById('resultImage');
        resultImage.src = canvas.toDataURL('image/png');
        resultImage.style.display = 'block';
    };
    
    img.src = `data:image/png;base64,${imageBase64}`;
}

function drawAnnotatedImageFromSrc(imageSrc, detections, imgWidth, imgHeight) {
    // Draw bounding boxes on image from data URL
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        // Draw bounding boxes with colors based on class
        const colors = ['#00FF00', '#00FFFF', '#FF00FF', '#FFFF00', '#FF6600', '#00FF99'];
        
        detections.forEach((det, idx) => {
            const bbox = det.bbox || {};
            
            // Convert normalized coordinates to pixel coordinates
            const x = bbox.x * img.width;
            const y = bbox.y * img.height;
            const w = bbox.width * img.width;
            const h = bbox.height * img.height;
            
            // Select color
            const color = colors[idx % colors.length];
            
            // Draw box
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);
            
            // Draw label background
            const label = `${idx + 1}. ${det.class_name || 'Unknown'} (${(det.confidence * 100).toFixed(1)}%)`;
            const fontSize = 14;
            ctx.font = `bold ${fontSize}px monospace`;
            const textWidth = ctx.measureText(label).width;
            
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(x, y - fontSize - 8, textWidth + 8, fontSize + 6);
            ctx.fillStyle = color;
            ctx.fillText(label, x + 4, y - 4);
        });
        
        // Display canvas as image
        const resultImage = document.getElementById('resultImage');
        resultImage.src = canvas.toDataURL('image/png');
        resultImage.style.display = 'block';
        resultImage.style.maxWidth = '100%';
        resultImage.style.height = 'auto';
        resultImage.style.border = '2px solid #00FF00';
    };
    
    img.src = imageSrc;
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
        tbody.innerHTML = '<tr><td colspan="5" class="no-data">NO DETECTIONS</td></tr>';
        return;
    }

    let html = '';
    detections.slice(0, 50).forEach((det, idx) => {
        // Handle different bbox formats
        const bbox = det.bbox || det.box || {};
        
        // Convert normalized coordinates to pixel coordinates
        let x = '?', y = '?', w = '?', h = '?';
        if (bbox.x !== undefined && bbox.y !== undefined && bbox.width !== undefined && bbox.height !== undefined) {
            x = bbox.x.toFixed(3);
            y = bbox.y.toFixed(3);
            w = bbox.width.toFixed(3);
            h = bbox.height.toFixed(3);
        } else if (bbox.x1 !== undefined && bbox.y1 !== undefined && bbox.x2 !== undefined && bbox.y2 !== undefined) {
            x = bbox.x1.toFixed(0);
            y = bbox.y1.toFixed(0);
            w = (bbox.x2 - bbox.x1).toFixed(0);
            h = (bbox.y2 - bbox.y1).toFixed(0);
        }
        
        const className = det.class_name || det.class || 'Unknown';
        const confidence = det.confidence ? (det.confidence * 100).toFixed(1) : '0.0';
        
        html += `
            <tr>
                <td>${idx + 1}</td>
                <td>${className}</td>
                <td>${confidence}%</td>
                <td title="x: ${x}, y: ${y}, w: ${w}, h: ${h}">[${x.substring(0,5)}, ${y.substring(0,5)}, ${w.substring(0,5)}, ${h.substring(0,5)}]</td>
            </tr>
        `;
    });

    if (detections.length > 50) {
        html += `<tr><td colspan="5" class="no-data">... and ${detections.length - 50} more</td></tr>`;
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

function readFileAsBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const result = reader.result;
            // Extract base64 data without the data:image/png;base64, prefix
            const base64 = result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function displayMetrics(results, processingTime) {
    const metricsDiv = document.getElementById('metricsBox');
    if (!metricsDiv) return;

    const detections = results.detections || [];
    const confidences = detections.map(d => d.confidence || 0);
    const avgConfidence = confidences.length > 0 
        ? (confidences.reduce((a, b) => a + b) / confidences.length * 100).toFixed(1)
        : 0;
    const maxConfidence = confidences.length > 0 
        ? (Math.max(...confidences) * 100).toFixed(1)
        : 0;
    const minConfidence = confidences.length > 0 
        ? (Math.min(...confidences) * 100).toFixed(1)
        : 0;

    // Determine detection mode
    let detectionMode = 'HEURISTIC (CPU Fallback)';
    let modelType = 'Heuristic Layout Detection';
    
    if (results.detection_mode === 'mmdet') {
        detectionMode = 'MMDET Neural Network';
        modelType = 'DINO (InternImage-XL)';
    }

    const metricsHTML = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
            <div style="background: #1a1a1a; border: 2px solid #00FF00; border-radius: 4px; padding: 12px;">
                <div style="color: #00FFFF; font-size: 12px; font-weight: bold;">DETECTION MODE</div>
                <div style="color: #00FF00; font-size: 14px; margin-top: 4px;">${detectionMode}</div>
            </div>
            <div style="background: #1a1a1a; border: 2px solid #00FF00; border-radius: 4px; padding: 12px;">
                <div style="color: #00FFFF; font-size: 12px; font-weight: bold;">MODEL TYPE</div>
                <div style="color: #00FF00; font-size: 14px; margin-top: 4px;">${modelType}</div>
            </div>
            <div style="background: #1a1a1a; border: 2px solid #00FF00; border-radius: 4px; padding: 12px;">
                <div style="color: #00FFFF; font-size: 12px; font-weight: bold;">PROCESSING TIME</div>
                <div style="color: #00FF00; font-size: 14px; margin-top: 4px;">${processingTime.toFixed(0)}ms</div>
            </div>
            <div style="background: #1a1a1a; border: 2px solid #00FF00; border-radius: 4px; padding: 12px;">
                <div style="color: #00FFFF; font-size: 12px; font-weight: bold;">AVG CONFIDENCE</div>
                <div style="color: #00FF00; font-size: 14px; margin-top: 4px;">${avgConfidence}%</div>
            </div>
            <div style="background: #1a1a1a; border: 2px solid #00FF00; border-radius: 4px; padding: 12px;">
                <div style="color: #00FFFF; font-size: 12px; font-weight: bold;">MAX CONFIDENCE</div>
                <div style="color: #00FF00; font-size: 14px; margin-top: 4px;">${maxConfidence}%</div>
            </div>
            <div style="background: #1a1a1a; border: 2px solid #00FF00; border-radius: 4px; padding: 12px;">
                <div style="color: #00FFFF; font-size: 12px; font-weight: bold;">MIN CONFIDENCE</div>
                <div style="color: #00FF00; font-size: 14px; margin-top: 4px;">${minConfidence}%</div>
            </div>
        </div>
    `;

    metricsDiv.innerHTML = metricsHTML;
}

console.log('[RODLA] Frontend loaded successfully. Ready for analysis.');
console.log('[RODLA] Demo mode available if backend is unavailable.');
