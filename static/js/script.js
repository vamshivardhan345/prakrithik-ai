/**
 * PRAKRITIK AI - FRONTEND CONTROLLER
 * Handles Image Upload, AI Diagnostics, Guardrails & Organic Remedy Catalog Search
 */

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------
    // 1. Navigation & Mobile Menu
    // -------------------------------------------------------------
    const navToggle = document.getElementById('nav-toggle');
    const navMenu = document.getElementById('nav-menu');
    const navLinks = document.querySelectorAll('.nav-link');

    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            navMenu.classList.toggle('open');
        });
    }

    // Smooth scroll and active tab handling
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    navLinks.forEach(l => l.classList.remove('active'));
                    this.classList.add('active');
                    if (navMenu) navMenu.classList.remove('open');
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
        });
    });

    // -------------------------------------------------------------
    // 2. Image Upload & Drag-and-Drop
    // -------------------------------------------------------------
    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');
    const browseBtn = document.getElementById('browse-btn');
    const predictBtn = document.getElementById('predict-btn');
    const removeBtn = document.getElementById('remove-btn');
    
    const uploadIdle = document.getElementById('upload-idle');
    const uploadPreview = document.getElementById('upload-preview');
    const imagePreview = document.getElementById('image-preview');
    const loadingState = document.getElementById('loading-state');
    
    const resultsEmpty = document.getElementById('results-empty');
    const resultsWarning = document.getElementById('results-warning');
    const warningTitle = document.getElementById('warning-title');
    const warningMessage = document.getElementById('warning-message');
    const resultsContent = document.getElementById('results-content');

    let currentFile = null;

    if (browseBtn) {
        browseBtn.addEventListener('click', () => fileInput.click());
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                processFile(e.target.files[0]);
            }
        });
    }

    // Drag and drop events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files[0]) {
            processFile(dt.files[0]);
        }
    });

    function processFile(file) {
        if (!file.type.startsWith('image/')) {
            showWarning("Invalid File Type", "Please select a valid image file (JPG, PNG, JPEG, or WEBP).");
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadIdle.style.display = 'none';
            uploadPreview.style.display = 'block';
            predictBtn.disabled = false;

            // Reset results display for the new upload
            resetResultsDisplay();
        };
        reader.readAsDataURL(file);
    }

    if (removeBtn) {
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetUpload();
        });
    }

    function resetUpload() {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        uploadPreview.style.display = 'none';
        uploadIdle.style.display = 'flex';
        predictBtn.disabled = true;
        resetResultsDisplay();
    }

    function resetResultsDisplay() {
        if (resultsEmpty) resultsEmpty.style.display = 'flex';
        if (resultsWarning) resultsWarning.style.display = 'none';
        if (resultsContent) resultsContent.style.display = 'none';
    }

    function showWarning(title, msg) {
        if (resultsEmpty) resultsEmpty.style.display = 'none';
        if (resultsContent) resultsContent.style.display = 'none';
        if (resultsWarning) {
            warningTitle.textContent = title;
            warningMessage.textContent = msg;
            resultsWarning.style.display = 'flex';
        }
    }

    // -------------------------------------------------------------
    // 3. Disease Diagnosis & Prediction Request
    // -------------------------------------------------------------
    if (predictBtn) {
        predictBtn.addEventListener('click', async () => {
            if (!currentFile) return;

            // UI state: loading
            predictBtn.disabled = true;
            uploadIdle.style.display = 'none';
            uploadPreview.style.display = 'none';
            loadingState.style.display = 'flex';
            resetResultsDisplay();

            const formData = new FormData();
            formData.append('image', currentFile);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                console.log("Prediction Response:", data);

                if (!response.ok || !data.success) {
                    // Show rejection or error alert
                    const title = data.uncertain ? "Diagnosis Rejected (Low Confidence)" : "Analysis Notice";
                    const message = data.error || "Unable to classify the uploaded image reliably.";
                    showWarning(title, message);
                } else {
                    // Render successful diagnostic & treatment data
                    renderDiagnosis(data);
                }
            } catch (error) {
                console.error("Diagnosis Error:", error);
                showWarning("Network or Server Error", "Could not connect to the diagnostic backend. Please check server connection.");
            } finally {
                loadingState.style.display = 'none';
                uploadPreview.style.display = 'block';
                predictBtn.disabled = false;
            }
        });
    }

    function renderDiagnosis(data) {
        if (resultsEmpty) resultsEmpty.style.display = 'none';
        if (resultsWarning) resultsWarning.style.display = 'none';
        if (resultsContent) resultsContent.style.display = 'block';

        // 1. Primary Diagnosis Details
        document.getElementById('disease-name').textContent = data.display_name || data.disease;
        document.getElementById('crop-name').textContent = data.crop || "--";
        document.getElementById('condition-name').textContent = data.disease || "--";

        // 2. Confidence Indicator
        const confPercent = Math.round((data.confidence || 0) * 100);
        document.getElementById('confidence-val').textContent = `${confPercent}%`;

        // 3. Top Probability Breakdown
        const barsContainer = document.getElementById('prediction-bars');
        barsContainer.innerHTML = '';
        if (data.top_predictions && data.top_predictions.length > 0) {
            data.top_predictions.forEach(p => {
                const item = document.createElement('div');
                item.className = 'prediction-item';
                const pVal = (p.prob * 100).toFixed(1);
                item.innerHTML = `
                    <div class="prediction-labels">
                        <span>${p.name}</span>
                        <span>${pVal}%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: ${pVal}%"></div>
                    </div>
                `;
                barsContainer.appendChild(item);
            });
        }

        // 4. Organic Treatment & Recommendation
        const rec = data.recommendation || {};
        document.getElementById('organic-rec-title').textContent = rec.organic_recommendation ? "Targeted Organic Solution" : "Recommended Cultural Action";
        document.getElementById('issue-type-badge').textContent = rec.issue_type || "Disease";
        document.getElementById('rec-solution').textContent = rec.organic_recommendation || "--";
        document.getElementById('rec-active-ingredients').textContent = rec.active_ingredients || "N/A";
        document.getElementById('rec-dosage').textContent = rec.dosage || "Label-specific";
        document.getElementById('rec-frequency').textContent = rec.frequency || "As directed on product label";
        document.getElementById('rec-validation').textContent = rec.validation_status || "Extension verified";
        document.getElementById('rec-guidance').textContent = rec.application_guidance || "--";
        document.getElementById('rec-prevention').textContent = rec.integrated_prevention || "--";
        document.getElementById('rec-safety').textContent = rec.safety_precautions || "--";

        if (rec.disclaimer) {
            document.getElementById('rec-disclaimer').textContent = rec.disclaimer;
        }

        // Smooth scroll to results on mobile/desktop
        resultsContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // -------------------------------------------------------------
    // 4. Organic Recommendations Catalog Browser
    // -------------------------------------------------------------
    const cropFilter = document.getElementById('crop-filter');
    const diseaseSearch = document.getElementById('disease-search');
    const searchBtn = document.getElementById('search-btn');
    const resetBtn = document.getElementById('reset-btn');
    const catalogGrid = document.getElementById('catalog-grid');

    // Populate crops dropdown
    async function loadCropsDropdown() {
        try {
            const res = await fetch('/api/crops');
            const data = await res.json();
            if (data.success && cropFilter) {
                data.crops.forEach(crop => {
                    const opt = document.createElement('option');
                    opt.value = crop;
                    opt.textContent = crop;
                    cropFilter.appendChild(opt);
                });
            }
        } catch (e) {
            console.error("Could not load crops dropdown:", e);
        }
    }

    // Fetch and render catalog cards
    async function fetchCatalog(crop = '', disease = '') {
        if (!catalogGrid) return;
        catalogGrid.innerHTML = `
            <div class="loading-catalog text-center" style="grid-column: 1/-1; padding: 3rem;">
                <div class="spinner" style="margin: 0 auto 1rem;"></div>
                <p>Loading organic remedies...</p>
            </div>
        `;

        try {
            const queryParams = new URLSearchParams();
            if (crop) queryParams.append('crop', crop);
            if (disease) queryParams.append('disease', disease);

            const res = await fetch(`/api/recommendations?${queryParams.toString()}`);
            const data = await res.json();

            if (data.success && data.recommendations && data.recommendations.length > 0) {
                catalogGrid.innerHTML = '';
                data.recommendations.forEach(item => {
                    const card = document.createElement('div');
                    card.className = 'catalog-card';
                    card.innerHTML = `
                        <div>
                            <div class="catalog-card-header">
                                <span class="catalog-crop"><i class="fa-solid fa-wheat-awn"></i> ${item.crop}</span>
                                <span class="badge badge-info">${item.issue_type || 'Pest/Disease'}</span>
                            </div>
                            <h3 class="catalog-title">${item.disease_or_pest}</h3>
                            <div class="catalog-body">
                                <p><strong>Organic Remedy:</strong> ${item.organic_recommendation}</p>
                                <p><strong>Active Bio-Agent:</strong> ${item.active_ingredients || 'Biological / Cultural'}</p>
                                <p><strong>Application:</strong> ${item.application_guidance}</p>
                                <p><strong>Dosage:</strong> ${item.dosage}</p>
                                <p><strong>Prevention:</strong> ${item.integrated_prevention}</p>
                                <p><strong>Safety:</strong> ${item.safety_precautions}</p>
                            </div>
                        </div>
                        <div class="catalog-meta">
                            <i class="fa-solid fa-landmark"></i> Reference: ${item.reference_to_verify || 'ICAR / KVK'}
                        </div>
                    `;
                    catalogGrid.appendChild(card);
                });
            } else {
                catalogGrid.innerHTML = `
                    <div style="grid-column: 1/-1; text-align: center; padding: 4rem 1rem; color: var(--text-muted);">
                        <i class="fa-solid fa-seedling" style="font-size: 2.5rem; margin-bottom: 1rem; color: #94a3b8;"></i>
                        <h3>No Matching Organic Remedies Found</h3>
                        <p>Try searching with a broader term or select 'All Crops'.</p>
                    </div>
                `;
            }
        } catch (e) {
            console.error("Catalog Fetch Error:", e);
            catalogGrid.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; padding: 2rem; color: var(--red);">
                    <p>Failed to load catalog records from backend.</p>
                </div>
            `;
        }
    }

    if (searchBtn) {
        searchBtn.addEventListener('click', () => {
            const crop = cropFilter ? cropFilter.value : '';
            const disease = diseaseSearch ? diseaseSearch.value : '';
            fetchCatalog(crop, disease);
        });
    }

    if (diseaseSearch) {
        diseaseSearch.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const crop = cropFilter ? cropFilter.value : '';
                const disease = diseaseSearch.value;
                fetchCatalog(crop, disease);
            }
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            if (cropFilter) cropFilter.value = '';
            if (diseaseSearch) diseaseSearch.value = '';
            fetchCatalog();
        });
    }

    // Initialize catalog & crops
    loadCropsDropdown();
    fetchCatalog();
});
