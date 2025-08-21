// VocationVector Frontend JavaScript

// Initialize Socket.IO connection with reconnection options
const socket = io({
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    reconnectionAttempts: 5,  // Limit reconnection attempts to avoid spam
    transports: ['polling', 'websocket'],  // Start with polling for reliability
    upgrade: true,
    rememberUpgrade: true,
    timeout: 20000  // Connection timeout
});

// Connection state tracking
let isConnected = false;

// WebSocket keepalive
let pingInterval = null;
let lastPongTime = Date.now();
const PING_INTERVAL = 10000; // Ping every 10 seconds
const PONG_TIMEOUT = 30000;  // Consider dead if no pong for 30 seconds

// Current active tab
let activeTab = 'upload';

// Process tracking
let activeProcesses = new Map();

// Polling intervals for active processes
let pollingIntervals = new Map();

// Store current data
let currentJobs = [];
let currentResumes = [];
let currentMatches = [];
let filteredMatches = [];
let matchSearchTerm = '';
let currentPreferences = null;

// Hidden items tracking (shared between jobs and matches)
let hiddenJobIds = new Set();
let showHiddenItems = false;

// Event handler functions (defined here so they can be removed/added properly)
function handleMatchFilterInput(e) {
    matchSearchTerm = e.target.value;
    displayFilteredMatches();
}

function handleMatchSortChange(e) {
    sortMatchesByValue(e.target.value);
}

// Loading states
let loadingStates = {
    resumes: false,
    jobs: false,
    matches: false,
    analytics: false,
    preferences: false
};

// Track which tabs have been loaded at least once
let loadedTabs = {
    upload: false,
    search: false,
    matches: false,
    analytics: false,
    preferences: false
};

// Helper function to format metadata display
function formatMetadata(text) {
    if (!text) return text;
    // Replace underscores with spaces and capitalize each word
    return text.toString()
        .replace(/_/g, ' ')
        .replace(/\b\w/g, char => char.toUpperCase());
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Restore the last active tab from localStorage
    const savedTab = localStorage.getItem('activeTab');
    if (savedTab) {
        activeTab = savedTab;
    }
    
    // Set up match filter event listener
    const matchFilterInput = document.getElementById('matchFilter');
    if (matchFilterInput) {
        matchFilterInput.addEventListener('input', handleMatchFilterInput);
    }
    
    // Set up match sort event listener
    const matchSortSelect = document.getElementById('matchSort');
    if (matchSortSelect) {
        matchSortSelect.addEventListener('change', handleMatchSortChange);
    }
    
    // Load hidden job IDs
    loadHiddenJobIds();
    
    initializeTabs();
    initializeFileUpload();
    initializeWeightSliders();
    loadInitialData();
    setupSocketListeners();
    setupModalHandlers();
    
    // Add connection status indicator
    addConnectionStatusIndicator();
    
    // Add resume filter listener
    const resumeFilter = document.getElementById('resumeFilter');
    if (resumeFilter) {
        resumeFilter.addEventListener('input', filterResumes);
    }
});

// Tab Navigation
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const tabName = this.dataset.tab;
            switchTab(tabName);
        });
    });
    
    // Set initial active tab
    switchTab(activeTab);
}

function switchTab(tabName) {
    // Update buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        btn.classList.add('bg-white', 'text-gray-700', 'border-gray-200', 'hover:border-blue-300', 'hover:text-blue-600');
    });
    
    const activeBtn = document.querySelector(`[data-tab="${tabName}"]`);
    if (activeBtn) {
        activeBtn.classList.remove('bg-white', 'text-gray-700', 'border-gray-200', 'hover:border-blue-300', 'hover:text-blue-600');
        activeBtn.classList.add('active');
    }
    
    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
        content.style.display = 'none';
    });
    
    const activeContent = document.getElementById(`${tabName}-tab`);
    if (activeContent) {
        activeContent.classList.add('active');
        activeContent.style.display = 'block';
    }
    
    activeTab = tabName;
    
    // Save the active tab to localStorage
    localStorage.setItem('activeTab', tabName);
    
    // Load tab-specific data
    loadTabData(tabName);
}

// Load data for specific tab
function loadTabData(tabName, forceRefresh = false) {
    // Always force refresh for upload and search tabs to get latest data from database
    if (tabName === 'upload' || tabName === 'search') {
        forceRefresh = true;
    }
    
    // Skip if already loaded and not forcing refresh
    if (loadedTabs[tabName] && !forceRefresh) {
        return;
    }
    
    switch(tabName) {
        case 'upload':
            loadResumes();
            // Don't mark as loaded so it always refreshes
            break;
        case 'preferences':
            loadPreferences();
            loadedTabs.preferences = true;
            break;
        case 'search':
            loadJobs();
            // Don't mark as loaded so it always refreshes with latest jobs
            break;
        case 'matches':
            // Check if there's a pending refresh
            if (window.pendingMatchRefresh) {
                console.log('Loading matches with pending refresh');
                window.pendingMatchRefresh = false;
                loadedTabs.matches = false; // Force reload
            }
            loadMatches();
            loadedTabs.matches = true;
            break;
        case 'analytics':
            loadAnalytics();
            loadedTabs.analytics = true;
            break;
    }
}

// File Upload
function initializeFileUpload() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    
    console.log('Initializing file upload:', { dropZone, fileInput, browseBtn });
    
    if (!dropZone || !fileInput) {
        console.error('File upload elements not found');
        return;
    }
    
    // Browse button click
    if (browseBtn) {
        browseBtn.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('Browse button clicked');
            fileInput.click();
        });
    }
    
    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    });
}

// Global function for inline onchange handler
window.handleFileSelect = function(event) {
    console.log('handleFileSelect called:', event.target.files);
    if (event.target.files.length > 0) {
        const fileName = event.target.files[0].name;
        console.log('Selected file:', fileName);
        updateUploadStatus(`Selected: ${fileName}`);
        handleFiles(event.target.files);
    }
}

function handleFiles(files) {
    console.log('handleFiles called with:', files);
    
    if (!files || files.length === 0) {
        console.log('No files to handle');
        return;
    }
    
    try {
        // Convert FileList to Array
        const fileArray = Array.from(files);
        console.log('Files to process:', fileArray.length);
        
        // Process each file
        fileArray.forEach((file, index) => {
            console.log(`Processing file ${index}:`, file.name, file.size, file.type);
            
            if (file.size > 10 * 1024 * 1024) {
                console.error('File too large:', file.name, file.size);
                if (typeof showNotification === 'function') {
                    showNotification('File too large. Maximum size is 10MB.', 'error');
                } else {
                    alert('File too large. Maximum size is 10MB.');
                }
                return;
            }
            
            // Upload immediately
            console.log(`Starting upload for: ${file.name}`);
            uploadFile(file);
        });
        
        // Clear the file input to allow re-selecting the same file
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.value = '';
        }
    } catch (error) {
        console.error('Error in handleFiles:', error);
        console.error('Error stack:', error.stack);
        alert('Error processing files: ' + error.message);
    }
}

async function uploadFile(file) {
    console.log('uploadFile called with:', file.name, 'size:', file.size, 'type:', file.type);
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Debug: Log FormData contents
    for (let [key, value] of formData.entries()) {
        console.log('FormData entry:', key, value);
    }
    
    // Show upload status
    console.log('Setting upload status');
    updateUploadStatus(`Uploading ${file.name}...`);
    
    try {
        console.log('Starting fetch to /api/upload_resume');
        const response = await fetch('/api/upload_resume', {
            method: 'POST',
            body: formData
        });
        
        console.log('Response received:', response.status, response.statusText);
        
        if (!response.ok) {
            console.error('Response not OK:', response.status);
            const errorText = await response.text();
            console.error('Error response body:', errorText);
            throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('Response data:', data);
        
        if (data.success) {
            showNotification(`Successfully uploaded ${file.name}`, 'success');
            activeProcesses.set(data.process_id, {
                type: 'resume',
                filename: file.name,
                startTime: new Date()
            });
            
            updateUploadStatus(`Processing ${file.name}... (this may take a few minutes)`);
            
            // Store the process ID to track this specific upload
            window.currentUploadProcessId = data.process_id;
            
            // Note if this was a replacement
            if (data.replaced) {
                showNotification(`Replacing existing resume: ${file.name}`, 'info');
            }
            
            // Trigger a refresh of the resumes list after a short delay
            setTimeout(() => {
                loadResumes();
            }, 2000);
            
        } else {
            showNotification(data.error || 'Upload failed', 'error');
            updateUploadStatus('Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        console.error('Error details:', {
            message: error.message,
            stack: error.stack,
            name: error.name
        });
        showNotification(`Upload failed: ${error.message}`, 'error');
        updateUploadStatus(`Upload failed: ${error.message}`);
    }
}

// Weight Sliders
function initializeWeightSliders() {
    const sliders = document.querySelectorAll('input[type="range"]');
    
    sliders.forEach(slider => {
        slider.addEventListener('input', function() {
            const valueSpan = this.parentElement.querySelector('.weight-value');
            if (valueSpan) {
                valueSpan.textContent = this.value;
            }
        });
    });
}

// API Functions
async function loadResumes() {
    const container = document.getElementById('resumesList');
    if (container && !loadingStates.resumes) {
        container.innerHTML = '<div class="text-center py-8"><div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div><p class="text-sm text-gray-500 mt-2">Loading resumes...</p></div>';
        loadingStates.resumes = true;
    }
    
    try {
        const response = await fetch('/api/get_resumes');
        const data = await response.json();
        
        if (data.success) {
            currentResumes = data.resumes;
            displayResumes(data.resumes);
        }
    } catch (error) {
        console.error('Error loading resumes:', error);
        if (container) {
            container.innerHTML = '<p class="text-red-500 text-sm p-4">Error loading resumes</p>';
        }
    } finally {
        loadingStates.resumes = false;
    }
}

function displayResumes(resumes) {
    const container = document.getElementById('resumesList');
    if (!container) return;
    
    if (resumes.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-sm">No resumes uploaded yet</p>';
        return;
    }
    
    container.innerHTML = resumes.map(resume => `
        <div class="resume-card border rounded-lg p-3 hover:shadow-md transition-shadow" data-resume-name="${(resume.filename || '').toLowerCase()}" data-resume-skills="${((resume.skills || []).join(' ') + ' ' + (resume.email || '')).toLowerCase()}">
            <div class="flex flex-col gap-2">
                <div class="flex items-start justify-between gap-2">
                    <div class="flex-1 min-w-0" onclick="showResumeDetail('${resume.id}')" style="cursor: pointer;">
                        <div class="text-sm font-semibold text-gray-900 truncate" title="${resume.filename || 'Unknown File'}">
                            ${resume.filename || 'Unknown File'}
                        </div>
                    </div>
                    <div class="flex items-center gap-1 flex-shrink-0">
                        <span class="px-2 py-1 bg-green-100 text-green-700 text-xs rounded">Processed</span>
                        <button onclick="event.stopPropagation(); deleteResume('${resume.id}')" 
                                class="text-red-500 hover:text-red-700 p-1" title="Delete">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="text-xs text-gray-500" onclick="showResumeDetail('${resume.id}')" style="cursor: pointer;">
                    <div>Uploaded: ${formatDate(resume.processed_at)}</div>
                    ${resume.years_experience ? `<div>${resume.years_experience} years experience</div>` : ''}
                    ${resume.email ? `<div>${resume.email}</div>` : ''}
                </div>
            </div>
        </div>
    `).join('');
}

function filterResumes() {
    const filterInput = document.getElementById('resumeFilter');
    if (!filterInput) return;
    
    const filterValue = filterInput.value.toLowerCase().trim();
    const resumeCards = document.querySelectorAll('.resume-card');
    
    let visibleCount = 0;
    resumeCards.forEach(card => {
        const name = card.getAttribute('data-resume-name') || '';
        const skills = card.getAttribute('data-resume-skills') || '';
        
        if (filterValue === '' || name.includes(filterValue) || skills.includes(filterValue)) {
            card.style.display = '';
            visibleCount++;
        } else {
            card.style.display = 'none';
        }
    });
    
    // Show message if no resumes match
    const container = document.getElementById('resumesList');
    if (visibleCount === 0 && filterValue !== '') {
        if (!container.querySelector('.no-results-message')) {
            container.insertAdjacentHTML('beforeend', '<p class="no-results-message text-gray-500 text-sm p-4">No resumes match your filter</p>');
        }
    } else {
        const noResultsMsg = container.querySelector('.no-results-message');
        if (noResultsMsg) {
            noResultsMsg.remove();
        }
    }
}

// Track jobs and matches that have been added to avoid duplicates
const addedJobIds = new Set();
const addedMatchIds = new Set();

function addJobToList(job) {
    const container = document.getElementById('jobsList');
    if (!container) return;
    
    // Skip if job already added
    if (addedJobIds.has(job.job_id)) return;
    addedJobIds.add(job.job_id);
    
    // Remove loading spinner if it's the first job
    if (container.querySelector('.animate-spin')) {
        container.innerHTML = '';
    }
    
    // Create job item HTML
    const jobHtml = `
        <li class="p-4 hover:bg-gray-50 cursor-pointer transition-colors" onclick="showJobDetail('${job.job_id}')">
            <div class="flex items-start justify-between">
                <div class="flex-1">
                    <h3 class="text-base font-semibold text-gray-900">${job.title || 'Unknown Position'}</h3>
                    <p class="text-sm text-gray-600 mt-1">${job.company || 'Unknown Company'}</p>
                    <div class="flex items-center gap-4 mt-2 text-xs text-gray-500">
                        ${job.location ? `<span>📍 ${job.location}</span>` : ''}
                        ${job.skills && job.skills.length > 0 ? `
                            <span>🔧 ${job.skills.slice(0, 3).join(', ')}${job.skills.length > 3 ? '...' : ''}</span>
                        ` : ''}
                    </div>
                </div>
                <div class="text-right">
                    <span class="inline-block px-2 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800">
                        New
                    </span>
                </div>
            </div>
        </li>
    `;
    
    // Add to list (prepend for newest first)
    container.insertAdjacentHTML('afterbegin', jobHtml);
    
    // Update job count
    window.jobCount = (window.jobCount || 0) + 1;
    
    // Show notification
    showNotification(`Added: ${job.title} at ${job.company}`, 'success');
}

function addMatchToList(match) {
    const container = document.getElementById('matchesList');
    if (!container) return;
    
    // Create unique match ID
    const matchId = `${match.job_id}_${match.resume_id}`;
    
    // Skip if match already added
    if (addedMatchIds.has(matchId)) return;
    addedMatchIds.add(matchId);
    
    // Remove "no matches" message if it exists
    const noMatches = container.querySelector('.text-gray-500');
    if (noMatches && noMatches.textContent.includes('No matches')) {
        noMatches.remove();
    }
    
    // Create match item HTML
    const matchHtml = `
        <li class="p-4 hover:bg-gray-50 transition-colors">
            <div class="flex items-start justify-between">
                <div class="flex-1">
                    <div class="flex items-center gap-2">
                        <h3 class="text-base font-semibold text-gray-900">${match.job_title}</h3>
                        <span class="inline-block px-2 py-0.5 text-xs font-medium rounded-full bg-green-100 text-green-800">
                            ${match.score}%
                        </span>
                    </div>
                    <p class="text-sm text-gray-600 mt-1">${match.company}</p>
                    <p class="text-xs text-gray-500 mt-1">Matched with: ${match.resume_name}</p>
                </div>
                <div class="text-right">
                    <span class="inline-block px-2 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800">
                        New Match
                    </span>
                </div>
            </div>
        </li>
    `;
    
    // Add to list (prepend for newest first)
    container.insertAdjacentHTML('afterbegin', matchHtml);
    
    // Update match count
    window.matchCount = (window.matchCount || 0) + 1;
    
    // Show notification for high-scoring matches
    if (match.score >= 70) {
        showNotification(`Great match found: ${match.job_title} (${match.score}%)`, 'success');
    }
}

async function loadJobs() {
    const container = document.getElementById('jobsList');
    if (container && !loadingStates.jobs) {
        // Only show loading if we don't have jobs yet
        if (addedJobIds.size === 0) {
            container.innerHTML = '<div class="text-center py-8"><div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div><p class="text-sm text-gray-500 mt-2">Loading jobs...</p></div>';
        }
        loadingStates.jobs = true;
    }
    
    try {
        const response = await fetch('/api/get_jobs');
        const data = await response.json();
        
        if (data.success) {
            currentJobs = data.jobs;
            displayJobs(data.jobs);
        }
    } catch (error) {
        console.error('Error loading jobs:', error);
        if (container) {
            container.innerHTML = '<p class="text-red-500 text-sm p-4">Error loading jobs</p>';
        }
    } finally {
        loadingStates.jobs = false;
    }
}

function displayJobs(jobs) {
    const container = document.getElementById('jobsList');
    if (!container) return;
    
    // Filter jobs based on hidden state
    const visibleJobs = showHiddenItems ? jobs : jobs.filter(job => !hiddenJobIds.has(job.id));
    const hiddenCount = jobs.filter(job => hiddenJobIds.has(job.id)).length;
    
    // Add toggle button if there are hidden items
    let headerHtml = '';
    if (hiddenCount > 0) {
        headerHtml = `
            <div class="mb-4 flex items-center justify-between">
                <span class="text-sm text-gray-600">${hiddenCount} job(s) archived</span>
                <button id="toggleHiddenJobs" onclick="toggleShowHidden()" 
                        class="px-3 py-1 text-xs text-white rounded ${showHiddenItems ? 'bg-blue-500' : 'bg-gray-500'}">
                    ${showHiddenItems ? 'Hide Archived' : 'Show Archived'}
                </button>
            </div>
        `;
    }
    
    if (visibleJobs.length === 0) {
        container.innerHTML = headerHtml + '<p class="text-gray-500 text-sm ml-2 mb-2">No jobs found. Start a search above.</p>';
        return;
    }
    
    container.innerHTML = headerHtml + visibleJobs.map(job => {
        const salaryDisplay = formatSalary(job.salary_min, job.salary_max);
        const isHidden = hiddenJobIds.has(job.id);
        
        return `
        <div class="job-card p-4 hover:bg-gray-50 transition-colors ${isHidden ? 'opacity-50' : ''}" onclick="showJobDetail('${job.id}')" style="cursor: pointer;">
            <div class="job-header">
                <div class="flex items-start gap-3">
                    <input type="checkbox" 
                           ${isHidden ? 'checked' : ''}
                           onclick="event.stopPropagation(); toggleJobVisibility('${job.id}')"
                           class="mt-1 cursor-pointer"
                           title="${isHidden ? 'Unarchive job' : 'Archive job'}">
                    <div class="job-main-info flex-1">
                        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
                            <div class="job-title">${job.title}</div>
                            ${salaryDisplay ? `
                            <div class="job-salary" style="background: linear-gradient(to right, #2563eb, #4f46e5); color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.9em; font-weight: 600;">
                                ${salaryDisplay}
                            </div>
                            ` : ''}
                        </div>
                        <div class="job-company">${job.company}</div>
                        <div class="job-meta">
                            ${job.location ? `
                            <div class="job-meta-item">
                                <i class="fas fa-map-marker-alt"></i>
                                <span>${job.location}</span>
                            </div>
                            ` : ''}
                            ${job.employment_type ? `
                            <div class="job-meta-item">
                                <i class="fas fa-briefcase"></i>
                                <span>${formatMetadata(job.employment_type)}</span>
                            </div>
                            ` : ''}
                            ${job.posted_date ? `
                            <div class="job-meta-item">
                                <i class="fas fa-calendar-alt"></i>
                                <span>${job.posted_date}</span>
                            </div>
                            ` : ''}
                            ${job.via ? `
                            <div class="job-meta-item">
                                <i class="fas fa-link"></i>
                                <span>${job.via}</span>
                            </div>
                            ` : ''}
                            ${job.years_experience_required ? `
                            <div class="job-meta-item">
                                <i class="fas fa-clock"></i>
                                <span>${job.years_experience_required} years experience</span>
                            </div>
                            ` : ''}
                            ${job.remote_policy ? `
                            <div class="job-meta-item">
                                <i class="fas fa-home"></i>
                                <span>${formatMetadata(job.remote_policy)}</span>
                            </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
                <button onclick="event.stopPropagation(); deleteJob('${job.id}')" 
                        class="text-red-500 hover:text-red-700 ml-4" title="Delete">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                    </svg>
                </button>
            </div>
            
            <!-- Job Description Preview -->
            ${job.description ? `
            <div class="mt-3">
                <p class="text-xs text-gray-600 line-clamp-2">${job.description.substring(0, 150)}${job.description.length > 150 ? '...' : ''}</p>
            </div>
            ` : ''}
            
            <!-- Additional Job Info -->
            ${(job.education_requirements && job.education_requirements.length > 0) || job.equity || job.bonus || job.team_size ? `
            <div class="mt-3">
                <div class="flex flex-wrap gap-3 text-xs text-gray-600">
                    ${job.education_requirements && job.education_requirements.length > 0 ? `
                        <span><i class="fas fa-graduation-cap"></i> ${
                            typeof job.education_requirements[0] === 'string' 
                                ? job.education_requirements[0] 
                                : (job.education_requirements[0].degree || 'Degree required')
                        }</span>
                    ` : ''}
                    ${job.equity ? `<span><i class="fas fa-chart-line"></i> ${job.equity}</span>` : ''}
                    ${job.bonus ? `<span><i class="fas fa-gift"></i> ${job.bonus}</span>` : ''}
                    ${job.team_size ? `<span><i class="fas fa-users"></i> Team: ${job.team_size}</span>` : ''}
                </div>
            </div>
            ` : ''}
            
            <div class="skills-container mt-3">
                ${job.skills.slice(0, 5).map(skill => {
                    // Handle both string skills and object skills
                    const skillText = typeof skill === 'string' ? skill : (skill.skill_name || skill.skill || skill.name || '');
                    return skillText ? `<span class="skill-tag">${skillText}</span>` : '';
                }).join('')}
                ${job.skills.length > 5 ? `<span class="text-xs text-gray-400">+${job.skills.length - 5} more</span>` : ''}
            </div>
        </div>
        `;
    }).join('');
}

// Global filter state - load from localStorage if available
let matchFilters = loadMatchFiltersFromStorage() || {
    minScore: 30,
    hiddenResumes: []
};

// Track matching process status
let matchingInProgress = false;
let matchingStatusMessage = '';

function loadMatchFiltersFromStorage() {
    try {
        const stored = localStorage.getItem('matchFilters');
        if (stored) {
            return JSON.parse(stored);
        }
    } catch (e) {
        console.error('Error loading match filters from storage:', e);
    }
    return null;
}

function saveMatchFiltersToStorage() {
    try {
        localStorage.setItem('matchFilters', JSON.stringify(matchFilters));
    } catch (e) {
        console.error('Error saving match filters to storage:', e);
    }
}

async function loadMatches() {
    // If matching is in progress, don't refresh - keep the status spinner
    if (matchingInProgress) {
        // Check if the spinner is still showing
        const matchesContent = document.getElementById('matchesContent');
        const hasSpinner = matchesContent && matchesContent.querySelector('#matchingStatus');
        
        if (!hasSpinner && matchesContent) {
            // Re-create the spinner if it was removed
            matchesContent.innerHTML = `
                <div class="border rounded-2xl overflow-hidden bg-white">
                    <div class="p-6">
                        <div class="flex flex-col items-center justify-center">
                            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
                            <p class="text-gray-700 font-semibold text-lg">Running Matching Algorithm</p>
                            <p class="text-gray-500 text-sm mt-1">Analyzing compatibility between resumes and jobs...</p>
                            <div id="matchingStatus" class="mt-6 w-full max-w-md space-y-3 text-sm">
                                <div class="flex items-center gap-3 p-3 bg-blue-50 rounded-lg animate-fadeIn">
                                    <div class="animate-pulse w-2 h-2 bg-blue-600 rounded-full"></div>
                                    <span class="text-gray-700">${matchingStatusMessage}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else if (hasSpinner) {
            // Just update the status message if spinner exists
            const statusSpan = document.querySelector('#matchingStatus span.text-gray-700');
            if (statusSpan) {
                statusSpan.textContent = matchingStatusMessage;
            }
        }
        return; // Don't load matches while processing
    }
    
    const container = document.getElementById('matchesList');
    if (container && !loadingStates.matches) {
        container.innerHTML = '<div class="text-center py-8"><div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div><p class="text-sm text-gray-500 mt-2">Loading matches...</p></div>';
        loadingStates.matches = true;
    }
    
    // Restore filter UI from saved state
    const scoreSlider = document.getElementById('minScoreSlider');
    if (scoreSlider && scoreSlider.value != matchFilters.minScore) {
        scoreSlider.value = matchFilters.minScore;
    }
    const scoreValueEl = document.getElementById('minScoreValue');
    if (scoreValueEl) {
        scoreValueEl.textContent = `${matchFilters.minScore}%`;
    }
    
    try {
        // Build query parameters with filters
        const params = new URLSearchParams({
            min_score: matchFilters.minScore / 100,  // Convert to decimal
            hidden_resumes: matchFilters.hiddenResumes.join(',')
        });
        
        const response = await fetch(`/api/get_matches?${params}`);
        const data = await response.json();
        
        if (data.success) {
            currentMatches = data.matches;
            // Preserve the search term if there's an input element
            const existingFilter = document.getElementById('matchFilter');
            if (existingFilter && existingFilter.value) {
                matchSearchTerm = existingFilter.value;
            }
            displayMatches(data.matches);
            
            // Update filter stats if elements exist (only on matches tab)
            const filteredCountEl = document.getElementById('filteredCount');
            const totalCountEl = document.getElementById('totalCount');
            if (filteredCountEl && totalCountEl) {
                filteredCountEl.textContent = data.total_after_filter || data.matches.length;
                totalCountEl.textContent = data.total_before_filter || data.matches.length;
            }
            
            // Load resume filters if not already loaded and container exists
            const resumeFiltersEl = document.getElementById('resumeFilters');
            if (resumeFiltersEl) {
                loadResumeFilters();
            }
        }
    } catch (error) {
        console.error('Error loading matches:', error);
        if (container) {
            container.innerHTML = '<p class="text-red-500 text-sm p-4">Error loading matches</p>';
        }
    } finally {
        loadingStates.matches = false;
    }
}

function filterMatchesBySearch(matches) {
    if (!matchSearchTerm || matchSearchTerm.trim() === '') {
        console.log('No search term, returning all matches:', matches.length);
        return matches;
    }
    
    const searchLower = matchSearchTerm.toLowerCase();
    console.log('Filtering matches with term:', searchLower);
    
    const filtered = matches.filter(match => {
        // Search in company name
        if (match.company && match.company.toLowerCase().includes(searchLower)) {
            return true;
        }
        
        // Search in job title
        if (match.job_title && match.job_title.toLowerCase().includes(searchLower)) {
            return true;
        }
        
        // Search in location
        if (match.location && match.location.toLowerCase().includes(searchLower)) {
            return true;
        }
        
        // Search in matched skills
        if (match.skills_matched && Array.isArray(match.skills_matched) && 
            match.skills_matched.some(skill => 
                skill && typeof skill === 'string' && skill.toLowerCase().includes(searchLower))) {
            return true;
        }
        
        // Search in resume name
        if (match.resume_name && match.resume_name.toLowerCase().includes(searchLower)) {
            return true;
        }
        
        return false;
    });
    
    console.log('Filtered results:', filtered.length, 'out of', matches.length);
    return filtered;
}

function displayFilteredMatches() {
    filteredMatches = filterMatchesBySearch(currentMatches);
    renderMatchesList(filteredMatches);
}

function sortMatchesByValue(sortBy) {
    if (sortBy === 'date') {
        // Sort by posted date (newest first)
        filteredMatches.sort((a, b) => {
            const dateA = a.posted_date ? new Date(a.posted_date) : new Date(0);
            const dateB = b.posted_date ? new Date(b.posted_date) : new Date(0);
            return dateB - dateA;
        });
    } else {
        // Sort by score (highest first)
        filteredMatches.sort((a, b) => b.overall_score - a.overall_score);
    }
    renderMatchesList(filteredMatches);
}

function displayMatches(matches) {
    // Store matches globally
    currentMatches = matches;
    
    // Apply search filter if there's a search term
    filteredMatches = filterMatchesBySearch(matches);
    // First check if we need to restore the normal matches structure
    const matchesContent = document.getElementById('matchesContent');
    if (matchesContent) {
        // Check if we're showing the spinner
        const hasSpinner = matchesContent.querySelector('#matchingStatus');
        if (hasSpinner) {
            // Restore the normal structure with search input
            matchesContent.innerHTML = `
                <div class="overflow-hidden bg-white rounded-2xl shadow-lg shadow-gray-200/40">
                    <div class="p-3 border-b flex items-center gap-3 text-sm">
                        <input id="matchFilter" class="rounded-lg border px-3 py-2 flex-1" 
                               placeholder="Filter by company, skill, or keyword..."
                               value="${matchSearchTerm}">
                        <select id="matchSort" class="rounded-lg border px-2 py-2">
                            <option value="score">Sort by Score</option>
                            <option value="date">Sort by Date</option>
                        </select>
                    </div>
                    <ul id="matchesList" class="divide-y">
                        <!-- Matches will be loaded here -->
                    </ul>
                </div>
            `;
            
            // Add event listener for search (remove old one first to avoid duplicates)
            const filterInput = document.getElementById('matchFilter');
            if (filterInput) {
                // Remove any existing listener
                filterInput.removeEventListener('input', handleMatchFilterInput);
                // Add new listener
                filterInput.addEventListener('input', handleMatchFilterInput);
            }
            
            // Add event listener for sort (remove old one first to avoid duplicates)
            const sortSelect = document.getElementById('matchSort');
            if (sortSelect) {
                // Remove any existing listener
                sortSelect.removeEventListener('change', handleMatchSortChange);
                // Add new listener
                sortSelect.addEventListener('change', handleMatchSortChange);
            }
        }
    }
    
    renderMatchesList(filteredMatches);
}

function renderMatchesList(matchesToRender) {
    const container = document.getElementById('matchesList');
    if (!container) return;
    
    // Filter matches based on hidden job IDs
    const visibleMatches = showHiddenItems ? matchesToRender : matchesToRender.filter(match => !hiddenJobIds.has(match.job_id));
    const hiddenCount = matchesToRender.filter(match => hiddenJobIds.has(match.job_id)).length;
    
    // Add toggle button if there are hidden matches
    let headerHtml = '';
    if (hiddenCount > 0) {
        headerHtml = `
            <div class="mb-4 flex items-center justify-between">
                <span class="text-sm text-gray-600">${hiddenCount} match(es) for archived jobs</span>
                <button id="toggleHiddenMatches" onclick="toggleShowHidden()" 
                        class="px-3 py-1 text-xs text-white rounded ${showHiddenItems ? 'bg-blue-500' : 'bg-gray-500'}">
                    ${showHiddenItems ? 'Hide Archived' : 'Show Archived'}
                </button>
            </div>
        `;
    }
    
    if (visibleMatches.length === 0) {
        // Check if it's due to filtering or no matches at all
        const isFiltered = matchSearchTerm && matchSearchTerm.trim() !== '';
        const message = isFiltered 
            ? `<p>No matches found for "${matchSearchTerm}".</p>
               <p class="text-sm mt-2">Try adjusting your search terms.</p>`
            : `<p>No matches found yet.</p>
               <p class="text-sm mt-2">Upload resumes and search for jobs, then run matching.</p>`;
        
        container.innerHTML = headerHtml + `
            <div class="text-center py-8 text-gray-500">
                ${message}
            </div>
        `;
        return;
    }
    
    container.innerHTML = headerHtml + visibleMatches.map((match, index) => {
        const isHidden = hiddenJobIds.has(match.job_id);
        
        return `
        <div class="match-item p-4 hover:bg-gray-50 transition-colors ${isHidden ? 'opacity-50' : ''}" onclick="toggleMatchDetails('match-${index}')" style="cursor: pointer;">
            <div class="flex items-start gap-3">
                <input type="checkbox" 
                       ${isHidden ? 'checked' : ''}
                       onclick="event.stopPropagation(); toggleJobVisibility('${match.job_id}')"
                       class="mt-1 cursor-pointer"
                       title="${isHidden ? 'Unarchive job and matches' : 'Archive job and matches'}">
                <div class="flex items-start gap-4 flex-1">
                    <div class="score-ring">
                        ${createScoreRing(match.overall_score)}
                    </div>
                    <div class="flex-1">
                        <div class="mb-2">
                            <div class="flex items-center gap-2">
                                <div class="job-title font-semibold">${match.job_title}</div>
                                ${match.location ? `<span class="text-xs text-gray-500">📍 ${match.location}</span>` : ''}
                            </div>
                            <div class="job-company text-gray-600">${match.company}</div>
                            <div class="flex items-center gap-3 text-sm mt-1">
                                <span class="text-gray-500"><span class="font-medium">Candidate:</span> ${match.resume_name}</span>
                                ${(() => {
                                    const dateInfo = getPostedDateInfo(match.posted_date, match.crawled_at);
                                    // Add border color based on background
                                    const borderClass = dateInfo.bgClass.includes('green') ? 'border-green-200' : 
                                                       dateInfo.bgClass.includes('yellow') ? 'border-yellow-200' : 
                                                       dateInfo.bgClass.includes('red') ? 'border-red-200' : 'border-gray-200';
                                    return `<span class="inline-flex items-center px-2 py-0.5 rounded-md ${dateInfo.bgClass} ${dateInfo.className} text-xs border ${borderClass}">${dateInfo.text}</span>`;
                                })()}
                                ${(() => {
                                    const salaryText = formatSalary(match.salary_min, match.salary_max) || 'Salary not specified';
                                    const salaryClass = getSalaryMatchClass(match.salary_score, match.salary_min, match.salary_max);
                                    return `<span class="inline-flex items-center px-2 py-0.5 rounded-md text-gray-900 text-xs border ${salaryClass}">${salaryText}</span>`;
                                })()}
                            </div>
                        </div>
                    
                    <!-- Bidirectional Scores -->
                    ${match.job_fit_score !== null || match.candidate_fit_score !== null ? `
                    <div class="flex gap-4 mb-3 text-sm">
                        ${match.job_fit_score !== null ? `
                        <div class="flex items-center gap-2">
                            <span class="text-gray-600">Job Fit:</span>
                            <span class="font-semibold text-blue-600">${match.job_fit_score}%</span>
                        </div>
                        ` : ''}
                        ${match.candidate_fit_score !== null ? `
                        <div class="flex items-center gap-2">
                            <span class="text-gray-600">Candidate Fit:</span>
                            <span class="font-semibold text-green-600">${match.candidate_fit_score}%</span>
                        </div>
                        ` : ''}
                        ${match.confidence_score !== null ? `
                        <div class="flex items-center gap-2">
                            <span class="text-gray-600">Confidence:</span>
                            <span class="font-semibold text-purple-600">${match.confidence_score}%</span>
                        </div>
                        ` : ''}
                    </div>
                    ` : ''}
                    
                    <!-- Score Breakdown -->
                    <div class="score-breakdown grid grid-cols-7 gap-2 text-xs">
                        <div class="score-item text-center">
                            <div class="score-label text-gray-500">Title</div>
                            <div class="score-value-text font-semibold">${match.title_match_score != null ? match.title_match_score + '%' : '--'}</div>
                        </div>
                        <div class="score-item text-center">
                            <div class="score-label text-gray-500">Skills</div>
                            <div class="score-value-text font-semibold">${match.skills_score != null ? match.skills_score + '%' : '--'}</div>
                        </div>
                        <div class="score-item text-center">
                            <div class="score-label text-gray-500">Experience</div>
                            <div class="score-value-text font-semibold">${match.experience_score != null ? match.experience_score + '%' : '--'}</div>
                        </div>
                        <div class="score-item text-center">
                            <div class="score-label text-gray-500">Education</div>
                            <div class="score-value-text font-semibold">${match.education_score != null ? match.education_score + '%' : '--'}</div>
                        </div>
                        <div class="score-item text-center">
                            <div class="score-label text-gray-500">Location</div>
                            <div class="score-value-text font-semibold">${match.location_score != null ? match.location_score + '%' : '--'}</div>
                        </div>
                        <div class="score-item text-center">
                            <div class="score-label text-gray-500">Salary</div>
                            <div class="score-value-text font-semibold">${match.salary_score != null ? match.salary_score + '%' : '--'}</div>
                        </div>
                    </div>
                    
                    <!-- Expandable Details -->
                    <div id="match-${index}" class="match-details hidden mt-4 pt-4 border-t border-gray-200">
                        <!-- Skills Analysis -->
                        <div class="mb-4">
                            <h4 class="text-sm font-semibold mb-2">Skills Analysis</h4>
                            ${match.skills_matched && match.skills_matched.length > 0 ? `
                            <div class="mb-2">
                                <span class="text-xs text-green-600 font-medium">Matched Skills (${match.skills_matched.length})</span>
                                <div class="flex flex-wrap gap-1 mt-1">
                                    ${match.skills_matched.map(skill => 
                                        `<span class="px-2 py-1 text-xs bg-green-100 text-green-700 rounded">${typeof skill === 'object' ? skill.skill : skill}</span>`
                                    ).join('')}
                                </div>
                            </div>
                            ` : ''}
                            
                            ${match.skills_gap && match.skills_gap.length > 0 ? `
                            <div class="mb-2">
                                <span class="text-xs text-red-600 font-medium">✗ Missing Skills (${match.skills_gap.length})</span>
                                <div class="flex flex-wrap gap-1 mt-1">
                                    ${match.skills_gap.map(skill => 
                                        `<span class="px-2 py-1 text-xs bg-red-100 text-red-700 rounded">${skill}</span>`
                                    ).join('')}
                                </div>
                            </div>
                            ` : ''}
                            
                            ${match.exceeded_skills && match.exceeded_skills.length > 0 ? `
                            <div class="mb-2">
                                <span class="text-xs text-blue-600 font-medium">+ Additional Skills (${match.exceeded_skills.length})</span>
                                <div class="flex flex-wrap gap-1 mt-1">
                                    ${match.exceeded_skills.map(skill => 
                                        `<span class="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded">${skill}</span>`
                                    ).join('')}
                                </div>
                            </div>
                            ` : ''}
                        </div>
                        
                        <!-- Requirements Analysis -->
                        ${(match.requirements_matched && match.requirements_matched.length > 0) || 
                          (match.requirements_gap && match.requirements_gap.length > 0) ? `
                        <div class="mb-4">
                            <h4 class="text-sm font-semibold mb-2">Requirements Analysis</h4>
                            ${match.requirements_matched && match.requirements_matched.length > 0 ? `
                            <div class="mb-2">
                                <span class="text-xs text-green-600 font-medium">Met Requirements</span>
                                <ul class="text-xs text-gray-600 mt-1 ml-4 list-disc">
                                    ${match.requirements_matched.map(req => `<li>${req}</li>`).join('')}
                                </ul>
                            </div>
                            ` : ''}
                            
                            ${match.requirements_gap && match.requirements_gap.length > 0 ? `
                            <div class="mb-2">
                                <span class="text-xs text-red-600 font-medium">✗ Missing Requirements</span>
                                <ul class="text-xs text-gray-600 mt-1 ml-4 list-disc">
                                    ${match.requirements_gap.map(req => `<li>${req}</li>`).join('')}
                                </ul>
                            </div>
                            ` : ''}
                        </div>
                        ` : ''}
                        
                        <!-- Education & Experience Gaps -->
                        ${(match.education_gaps && match.education_gaps.length > 0) || match.experience_gaps ? `
                        <div class="mb-4">
                            <h4 class="text-sm font-semibold mb-2">Gaps Analysis</h4>
                            ${match.education_gaps && match.education_gaps.length > 0 ? `
                            <div class="mb-2">
                                <span class="text-xs text-orange-600 font-medium">Education Gaps</span>
                                <ul class="text-xs text-gray-600 mt-1 ml-4 list-disc">
                                    ${match.education_gaps.map(gap => `<li>${gap}</li>`).join('')}
                                </ul>
                            </div>
                            ` : ''}
                            ${match.experience_gaps && match.experience_gaps.years_difference !== undefined ? `
                            <div class="mb-2">
                                <span class="text-xs text-orange-600 font-medium">Experience Gap</span>
                                <p class="text-xs text-gray-600 mt-1 ml-4">
                                    ${match.experience_gaps.years_difference > 0 ? 
                                        `Candidate has ${match.experience_gaps.years_difference} years more experience than required` :
                                        `Candidate needs ${Math.abs(match.experience_gaps.years_difference)} more years of experience`}
                                </p>
                            </div>
                            ` : ''}
                        </div>
                        ` : ''}
                        
                        <!-- Preferences Alignment -->
                        ${match.salary_match || match.location_preference_met !== undefined ? `
                        <div class="mb-4">
                            <h4 class="text-sm font-semibold mb-2">Preferences Alignment</h4>
                            ${match.salary_match && match.salary_match.alignment ? `
                            <div class="mb-2">
                                <span class="text-xs text-gray-600">Salary:</span>
                                <span class="text-xs ml-2 ${
                                    match.salary_match.alignment === 'aligned' ? 'text-green-600' :
                                    match.salary_match.alignment === 'below_expectations' ? 'text-red-600' :
                                    match.salary_match.alignment === 'above_expectations' ? 'text-orange-600' :
                                    match.salary_match.alignment === 'not_specified' ? 'text-red-600' :
                                    match.salary_match.alignment === 'no_preference' ? 'text-gray-600' :
                                    'text-gray-600'
                                }">${match.salary_match.alignment.replace(/_/g, ' ')}</span>
                                ${match.salary_match.job_range ? `<span class="text-xs text-gray-500 ml-2">(Job: ${match.salary_match.job_range})</span>` : ''}
                            </div>
                            ` : ''}
                            ${match.location_preference_met !== undefined ? `
                            <div class="mb-2">
                                <span class="text-xs text-gray-600">Location:</span>
                                <span class="text-xs ml-2 ${match.location_preference_met ? 'text-green-600' : 'text-red-600'}">
                                    ${match.location_preference_met ? 'Preference met' : '✗ Not in preferred location'}
                                </span>
                            </div>
                            ` : ''}
                            ${match.remote_preference_met !== undefined ? `
                            <div class="mb-2">
                                <span class="text-xs text-gray-600">Remote:</span>
                                <span class="text-xs ml-2 ${match.remote_preference_met ? 'text-green-600' : 'text-red-600'}">
                                    ${match.remote_preference_met ? 'Remote option available' : '✗ No remote option'}
                                </span>
                            </div>
                            ` : ''}
                        </div>
                        ` : ''}
                        
                        <!-- AI Assessment -->
                        ${match.llm_assessment ? `
                        <div class="mb-4">
                            <h4 class="text-sm font-semibold mb-2">AI Assessment</h4>
                            <div class="text-xs text-gray-600">
                                ${(() => {
                                    // Parse the LLM assessment into structured format
                                    const assessment = match.llm_assessment;
                                    let mainPoints = [];
                                    let recommendations = [];
                                    let isRecommendations = false;
                                    
                                    // Split by sentences and process
                                    const lines = assessment.split('\n');
                                    for (let line of lines) {
                                        line = line.trim();
                                        if (line.toLowerCase().includes('recommendation')) {
                                            isRecommendations = true;
                                            continue;
                                        }
                                        if (line) {
                                            if (isRecommendations) {
                                                // Handle bullet points in recommendations
                                                if (line.startsWith('•') || line.startsWith('-')) {
                                                    recommendations.push(line.substring(1).trim());
                                                } else {
                                                    recommendations.push(line);
                                                }
                                            } else {
                                                // Parse main assessment into key points
                                                // Split by periods but keep important context
                                                const sentences = line.split(/(?<=[.!?])\s+/);
                                                for (let sentence of sentences) {
                                                    sentence = sentence.trim();
                                                    if (sentence && sentence.length > 10) {
                                                        mainPoints.push(sentence);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    
                                    // Build the HTML
                                    let html = '';
                                    
                                    // Main assessment points
                                    if (mainPoints.length > 0) {
                                        html += '<div class="mb-2">';
                                        html += '<span class="text-xs font-medium">Key Findings:</span>';
                                        html += '<ul class="text-xs text-gray-600 mt-1 ml-4 list-disc">';
                                        for (let point of mainPoints) {
                                            html += `<li>${point}</li>`;
                                        }
                                        html += '</ul>';
                                        html += '</div>';
                                    }
                                    
                                    // Recommendations
                                    if (recommendations.length > 0) {
                                        html += '<div class="mb-2">';
                                        html += '<span class="text-xs font-medium">Recommendations:</span>';
                                        html += '<ul class="text-xs text-gray-600 mt-1 ml-4 list-disc">';
                                        for (let rec of recommendations) {
                                            html += `<li>${rec}</li>`;
                                        }
                                        html += '</ul>';
                                        html += '</div>';
                                    }
                                    
                                    // Fallback if parsing didn't work well
                                    if (!html) {
                                        html = `<p class="italic">"${assessment}"</p>`;
                                    }
                                    
                                    return html;
                                })()}
                            </div>
                        </div>
                        ` : ''}
                        
                        <!-- Match Reasons -->
                        ${match.match_reasons && match.match_reasons.length > 0 ? `
                        <div class="mb-4">
                            <h4 class="text-sm font-semibold mb-2">Key Match Factors</h4>
                            <ul class="text-xs text-gray-600 ml-4 list-disc">
                                ${match.match_reasons.map(reason => `<li>${reason}</li>`).join('')}
                            </ul>
                        </div>
                        ` : ''}
                        
                        <!-- Action Buttons -->
                        <div class="flex gap-2 mt-4">
                            <button onclick="event.stopPropagation(); showJobDetail('${match.job_id}')" 
                                    class="px-3 py-1 text-xs rounded-lg bg-blue-600 text-white hover:bg-blue-700">
                                View Job
                            </button>
                            <button onclick="event.stopPropagation(); showResumeDetail('${match.resume_id}')" 
                                    class="px-3 py-1 text-xs rounded-lg bg-gray-600 text-white hover:bg-gray-700">
                                View Resume
                            </button>
                        </div>
                    </div>
                    </div> <!-- Close flex-1 div -->
                </div>
            </div>
        </div>
        `;
    }).join('');
}

function createScoreRing(score) {
    const radius = 20;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score / 100) * circumference;
    const color = score >= 85 ? '#10b981' : score >= 70 ? '#3b82f6' : score >= 50 ? '#f59e0b' : '#ef4444';
    
    return `
        <svg width="60" height="60">
            <circle cx="30" cy="30" r="${radius}" stroke="#e5e7eb" stroke-width="4" fill="none" />
            <circle cx="30" cy="30" r="${radius}" stroke="${color}" stroke-width="4" 
                    stroke-linecap="round" fill="none" 
                    stroke-dasharray="${circumference}" 
                    stroke-dashoffset="${offset}"
                    style="transform: rotate(-90deg); transform-origin: center;" />
        </svg>
        <span class="score-value">${score}</span>
    `;
}

// Detail Modal Functions
function showJobDetail(jobId) {
    fetch(`/api/get_job/${jobId}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const job = data.job;
                const salaryDisplay = formatSalary(job.salary_min, job.salary_max);
                
                const modalContent = `
                    <div class="modal show" id="jobDetailModal">
                        <div class="modal-dialog" style="max-width: 900px;">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <div>
                                        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                                            <h2 class="modal-title" style="margin: 0;">${job.title}</h2>
                                            ${salaryDisplay ? `
                                            <div style="background: linear-gradient(to right, #2563eb, #4f46e5); color: white; padding: 4px 12px; border-radius: 6px; font-size: 1.1em; font-weight: 600;">
                                                ${salaryDisplay}
                                            </div>
                                            ` : ''}
                                        </div>
                                        <div class="text-secondary">${job.company} • ${job.location || 'Location not specified'}</div>
                                    </div>
                                    <button class="btn-close" onclick="closeModal('jobDetailModal')">×</button>
                                </div>
                                <div class="modal-body">
                                    <!-- Job Details Grid -->
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Job Details</h3>
                                        <div class="detail-grid">
                                            ${job.employment_type ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Employment Type</div>
                                                <div class="detail-value">${formatMetadata(job.employment_type)}</div>
                                            </div>
                                            ` : ''}
                                            ${job.remote_policy ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Remote Policy</div>
                                                <div class="detail-value">${formatMetadata(job.remote_policy)}</div>
                                            </div>
                                            ` : ''}
                                            ${job.years_experience_required ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Experience Required</div>
                                                <div class="detail-value">${job.years_experience_required} years</div>
                                            </div>
                                            ` : ''}
                                            ${job.equity ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Equity</div>
                                                <div class="detail-value">${job.equity}</div>
                                            </div>
                                            ` : ''}
                                            ${job.bonus ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Bonus</div>
                                                <div class="detail-value">${job.bonus}</div>
                                            </div>
                                            ` : ''}
                                            ${job.team_size ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Team Size</div>
                                                <div class="detail-value">${job.team_size}</div>
                                            </div>
                                            ` : ''}
                                            ${job.start_date ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Start Date</div>
                                                <div class="detail-value">${job.start_date}</div>
                                            </div>
                                            ` : ''}
                                            ${job.work_life_balance ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Work-Life Balance</div>
                                                <div class="detail-value">${job.work_life_balance}</div>
                                            </div>
                                            ` : ''}
                                            ${job.management_style ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Management Style</div>
                                                <div class="detail-value">${job.management_style}</div>
                                            </div>
                                            ` : ''}
                                            ${job.department ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Department</div>
                                                <div class="detail-value">${job.department}</div>
                                            </div>
                                            ` : ''}
                                            ${job.industry ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Industry</div>
                                                <div class="detail-value">${job.industry}</div>
                                            </div>
                                            ` : ''}
                                            ${job.search_location ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Search Location</div>
                                                <div class="detail-value">${job.search_location}</div>
                                            </div>
                                            ` : ''}
                                            ${job.posted_date ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Posted</div>
                                                <div class="detail-value">${job.posted_date}</div>
                                            </div>
                                            ` : ''}
                                            ${job.employment_type ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Employment Type</div>
                                                <div class="detail-value">${job.employment_type}</div>
                                            </div>
                                            ` : ''}
                                            ${job.via ? `
                                            <div class="detail-item">
                                                <div class="detail-label">Source</div>
                                                <div class="detail-value">${job.via}</div>
                                            </div>
                                            ` : ''}
                                        </div>
                                    </div>
                                    
                                    ${job.description ? `
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Description</h3>
                                        <div class="detail-text">
                                            ${job.description.replace(/\n/g, '<br>')}
                                        </div>
                                    </div>
                                    ` : ''}
                                    
                                    ${(() => {
                                        // Combine skills and key technologies into one section
                                        const allSkills = [];
                                        
                                        // Add regular skills
                                        if (job.skills && job.skills.length > 0) {
                                            job.skills.forEach(skill => {
                                                if (typeof skill === 'string') {
                                                    allSkills.push({name: skill, isString: true});
                                                } else if (typeof skill === 'object' && skill !== null) {
                                                    allSkills.push({...skill, isString: false});
                                                }
                                            });
                                        }
                                        
                                        // Add technical skills if not already in skills
                                        if (job.technical_skills && job.technical_skills.length > 0) {
                                            job.technical_skills.forEach(skill => {
                                                const skillName = typeof skill === 'string' ? skill : (skill.skill_name || skill.skill || skill.name);
                                                if (!allSkills.some(s => (s.name || s.skill_name || s.skill) === skillName)) {
                                                    if (typeof skill === 'string') {
                                                        allSkills.push({name: skill, isString: true});
                                                    } else {
                                                        allSkills.push({...skill, isString: false});
                                                    }
                                                }
                                            });
                                        }
                                        
                                        // Add key technologies
                                        if (job.key_technologies && job.key_technologies.length > 0) {
                                            job.key_technologies.forEach(tech => {
                                                if (!allSkills.some(s => (s.name || s.skill_name || s.skill) === tech)) {
                                                    allSkills.push({name: tech, isString: true, isTechnology: true});
                                                }
                                            });
                                        }
                                        
                                        if (allSkills.length === 0) return '';
                                        
                                        return `
                                        <div class="detail-section">
                                            <h3 class="detail-section-title">Required Skills & Technologies</h3>
                                            <div class="skills-container">
                                                ${allSkills.map(skill => {
                                                    if (skill.isString) {
                                                        return `<span class="skill-tag">${skill.name}</span>`;
                                                    } else {
                                                        const skillName = skill.skill_name || skill.skill || skill.name || 'Unknown';
                                                        const required = skill.required || skill.importance || '';
                                                        const level = skill.level || skill.proficiency || '';
                                                        
                                                        let skillHtml = `<span class="skill-tag">`;
                                                        skillHtml += `<span class="font-medium">${skillName}</span>`;
                                                        
                                                        // Add metadata if available
                                                        const metadata = [];
                                                        if (required === true || required === 'required') metadata.push('Required');
                                                        if (level) metadata.push(level);
                                                        
                                                        if (metadata.length > 0) {
                                                            skillHtml += ` <span class="text-xs text-gray-500">(${metadata.join(', ')})</span>`;
                                                        }
                                                        
                                                        skillHtml += `</span>`;
                                                        return skillHtml;
                                                    }
                                                }).join('')}
                                            </div>
                                        </div>
                                        `;
                                    })()}
                                    
                                    ${job.responsibilities && job.responsibilities.length > 0 ? `
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Responsibilities</h3>
                                        <ul class="list-disc pl-5 space-y-1">
                                            ${job.responsibilities.map(r => `<li class="text-sm">${r}</li>`).join('')}
                                        </ul>
                                    </div>
                                    ` : ''}
                                    
                                    ${job.requirements && job.requirements.length > 0 ? `
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Requirements</h3>
                                        <ul class="list-disc pl-5 space-y-1">
                                            ${job.requirements.map(r => `<li class="text-sm">${r}</li>`).join('')}
                                        </ul>
                                    </div>
                                    ` : ''}
                                    
                                    ${job.education_requirements && job.education_requirements.length > 0 ? `
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Education Requirements</h3>
                                        <ul class="list-disc pl-5 space-y-1">
                                            ${job.education_requirements.map(edu => {
                                                if (typeof edu === 'string') {
                                                    return `<li class="text-sm">${edu}</li>`;
                                                } else if (typeof edu === 'object' && edu !== null) {
                                                    const degree = edu.degree || edu.level || '';
                                                    const field = edu.field || edu.major || '';
                                                    const required = edu.required ? ' (Required)' : ' (Preferred)';
                                                    return `<li class="text-sm">${degree}${field ? ' in ' + field : ''}${required}</li>`;
                                                }
                                                return '';
                                            }).join('')}
                                        </ul>
                                    </div>
                                    ` : ''}
                                    
                                    ${job.certifications_required && job.certifications_required.length > 0 ? `
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Required Certifications</h3>
                                        <ul class="list-disc pl-5 space-y-1">
                                            ${job.certifications_required.map(cert => `<li class="text-sm">${cert}</li>`).join('')}
                                        </ul>
                                    </div>
                                    ` : ''}
                                    
                                    ${job.preferred_industries && job.preferred_industries.length > 0 ? `
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Preferred Industries</h3>
                                        <ul class="list-disc pl-5 space-y-1">
                                            ${job.preferred_industries.map(ind => `<li class="text-sm">${ind}</li>`).join('')}
                                        </ul>
                                    </div>
                                    ` : ''}
                                    
                                    ${job.growth_opportunities ? `
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Growth Opportunities</h3>
                                        <div class="detail-text">
                                            ${typeof job.growth_opportunities === 'string' ? job.growth_opportunities.replace(/\n/g, '<br>') : job.growth_opportunities}
                                        </div>
                                    </div>
                                    ` : ''}
                                    
                                    ${job.benefits && job.benefits.length > 0 ? `
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Benefits</h3>
                                        <ul class="list-disc pl-5 space-y-1">
                                            ${job.benefits.map(b => `<li class="text-sm">${b}</li>`).join('')}
                                        </ul>
                                    </div>
                                    ` : ''}
                                    
                                    ${job.apply_links && job.apply_links.length > 0 ? `
                                    <!-- Apply Links Section -->
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Apply for This Position</h3>
                                        <div class="flex gap-2 flex-wrap">
                                            ${job.apply_links.map(link => `
                                                <a href="${link.url}" target="_blank" rel="noopener noreferrer" 
                                                   class="inline-flex items-center px-3 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors">
                                                    <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                                              d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path>
                                                    </svg>
                                                    ${link.text || 'Apply Now'}
                                                </a>
                                            `).join('')}
                                        </div>
                                    </div>
                                    ` : ''}
                                    
                                    <!-- Original Job Text -->
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Original Job Description</h3>
                                        <div class="detail-text">
                                            ${job.full_text.replace(/\n/g, '<br>')}
                                        </div>
                                    </div>
                                </div>
                                <div class="modal-footer">
                                    <button class="btn btn-outline-primary" onclick="closeModal('jobDetailModal')">Close</button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                // Add modal to body
                const existingModal = document.getElementById('jobDetailModal');
                if (existingModal) {
                    existingModal.remove();
                }
                document.body.insertAdjacentHTML('beforeend', modalContent);
            }
        })
        .catch(error => {
            console.error('Error loading job details:', error);
            showNotification('Failed to load job details', 'error');
        });
}

function showResumeDetail(resumeId) {
    fetch(`/api/get_resume/${resumeId}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const resume = data.resume;
                
                // Store current resume globally for skill management
                window.currentResumeId = resumeId;
                window.currentResumeSkills = resume.skills || [];
                
                const modalContent = `
                    <div class="modal show" id="resumeDetailModal">
                        <div class="modal-dialog" style="max-width: 1000px;">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <div class="flex-1">
                                        <h2 class="modal-title">${resume.filename || 'Resume'}</h2>
                                        <div class="text-secondary mt-1">
                                            ${resume.name || 'Unknown'} 
                                            ${resume.years_experience ? `• ${resume.years_experience} years experience` : ''}
                                        </div>
                                        <div class="text-sm text-muted mt-2">
                                            ${resume.email || ''} ${resume.email && (resume.phone || resume.location) ? '•' : ''} 
                                            ${resume.phone || ''} ${resume.phone && resume.location ? '•' : ''} 
                                            ${resume.location || ''}
                                        </div>
                                        ${resume.linkedin || resume.github ? `
                                        <div class="text-sm text-muted mt-1">
                                            ${resume.linkedin ? `<a href="${resume.linkedin}" target="_blank" class="text-blue-600 hover:underline">LinkedIn</a>` : ''}
                                            ${resume.linkedin && resume.github ? ' • ' : ''}
                                            ${resume.github ? `<a href="${resume.github}" target="_blank" class="text-blue-600 hover:underline">GitHub</a>` : ''}
                                        </div>
                                        ` : ''}
                                    </div>
                                    <div class="flex gap-2">
                                        <button onclick="openResumeEditor('${resumeId}')" class="px-3 py-1 text-sm rounded-lg bg-blue-500 text-white hover:bg-blue-600">
                                            Edit Details
                                        </button>
                                        <button class="btn-close" onclick="closeModal('resumeDetailModal')">×</button>
                                    </div>
                                </div>
                                <div class="modal-body">
                                    ${resume.summary ? `
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Professional Summary</h3>
                                        <p class="text-sm">${resume.summary}</p>
                                    </div>
                                    ` : ''}
                                    
                                    ${resume.skills && resume.skills.length > 0 ? `
                                    <div class="detail-section">
                                        <div class="flex items-center justify-between mb-3">
                                            <h3 class="detail-section-title mb-0">Skills</h3>
                                            <button onclick="openSkillManager()" class="px-3 py-1 text-xs rounded-lg border border-gray-300 hover:bg-gray-50">
                                                Manage Skills
                                            </button>
                                        </div>
                                        <div class="skills-container" id="resumeSkillsContainer">
                                            ${resume.skills.map((skill, index) => {
                                                // Handle both string skills and object skills with metadata
                                                if (typeof skill === 'string') {
                                                    return `<span class="skill-tag">${skill}</span>`;
                                                } else if (typeof skill === 'object' && skill !== null) {
                                                    // Handle different skill object formats
                                                    const skillName = skill.skill_name || skill.skill || skill.name || 'Unknown';
                                                    const proficiency = skill.required_proficiency || skill.proficiency || skill.level || '';
                                                    const years = skill.years_required || skill.years || skill.experience || '';
                                                    
                                                    // Skip if skill name is actually 'Unknown'
                                                    if (skillName === 'Unknown') return '';
                                                    
                                                    let skillHtml = `<span class="skill-tag">`;
                                                    skillHtml += `<span class="font-medium">${skillName}</span>`;
                                                    
                                                    // Add metadata if available
                                                    const metadata = [];
                                                    if (proficiency && proficiency !== 'null') metadata.push(proficiency);
                                                    if (years && years !== 'null') metadata.push(`${years} yrs`);
                                                    
                                                    if (metadata.length > 0) {
                                                        skillHtml += ` <span class="text-xs text-gray-500">(${metadata.join(', ')})</span>`;
                                                    }
                                                    
                                                    skillHtml += `</span>`;
                                                    return skillHtml;
                                                }
                                                return '';
                                            }).join('')}
                                        </div>
                                    </div>
                                    ` : ''}
                                    
                                    ${resume.experience && resume.experience.length > 0 && resume.experience[0] !== null ? `
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Professional Experience</h3>
                                        <div class="space-y-4">
                                            ${resume.experience.map(exp => `
                                                <div class="border-l-2 border-gray-200 pl-4 hover:border-gray-400 transition-colors">
                                                    <div class="flex items-start justify-between">
                                                        <div>
                                                            <div class="font-semibold text-gray-900">${exp.title || exp.role || 'Position'}</div>
                                                            <div class="text-sm text-gray-600 mt-1">
                                                                ${exp.company || 'Company'} 
                                                                ${exp.location ? `• ${exp.location}` : ''}
                                                            </div>
                                                        </div>
                                                        <div class="text-sm text-gray-500 whitespace-nowrap ml-4">
                                                            ${exp.duration || exp.dates || exp.start_date || ''}
                                                            ${exp.end_date && exp.end_date !== exp.start_date ? ` - ${exp.end_date}` : ''}
                                                        </div>
                                                    </div>
                                                    ${exp.description || exp.responsibilities ? `
                                                    <div class="mt-2 text-sm text-gray-700">
                                                        ${exp.description || ''}
                                                        ${exp.responsibilities && Array.isArray(exp.responsibilities) ? 
                                                            `<ul class="mt-1 list-disc list-inside">
                                                                ${exp.responsibilities.map(r => `<li>${r}</li>`).join('')}
                                                            </ul>` : ''}
                                                    </div>
                                                    ` : ''}
                                                </div>
                                            `).join('')}
                                        </div>
                                    </div>
                                    ` : ''}
                                    
                                    ${resume.education && resume.education.length > 0 && resume.education[0] !== null ? `
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Education</h3>
                                        <div class="space-y-3">
                                            ${resume.education.map(edu => `
                                                <div class="border-l-2 border-gray-200 pl-4">
                                                    <div class="font-semibold text-gray-900">
                                                        ${edu.degree || edu.qualification || 'Degree'}
                                                        ${edu.field || edu.major ? ` in ${edu.field || edu.major}` : ''}
                                                    </div>
                                                    <div class="text-sm text-gray-600 mt-1">
                                                        ${edu.institution || edu.school || 'Institution'}
                                                        ${edu.location ? ` • ${edu.location}` : ''}
                                                    </div>
                                                    <div class="text-sm text-gray-500">
                                                        ${edu.year || edu.graduation_date || edu.dates || ''}
                                                        ${edu.gpa ? ` • GPA: ${edu.gpa}` : ''}
                                                    </div>
                                                </div>
                                            `).join('')}
                                        </div>
                                    </div>
                                    ` : ''}
                                    
                                    ${resume.certifications && resume.certifications.length > 0 ? `
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Certifications</h3>
                                        <div class="flex flex-wrap gap-2">
                                            ${resume.certifications.map(cert => `
                                                <span class="px-3 py-1 bg-blue-50 text-blue-700 rounded-lg text-sm">
                                                    ${typeof cert === 'object' ? (cert.name || cert.title || JSON.stringify(cert)) : cert}
                                                </span>
                                            `).join('')}
                                        </div>
                                    </div>
                                    ` : ''}
                                    
                                    ${resume.achievements && resume.achievements.length > 0 ? `
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Key Achievements</h3>
                                        <ul class="list-disc list-inside space-y-1">
                                            ${resume.achievements.map(achievement => `
                                                <li class="text-sm text-gray-700">${achievement}</li>
                                            `).join('')}
                                        </ul>
                                    </div>
                                    ` : ''}
                                    
                                    ${resume.matching_template || resume.vocation_template ? `
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Preferences & Requirements</h3>
                                        <div class="grid md:grid-cols-2 gap-4">
                                            ${(() => {
                                                const template = resume.matching_template ? 
                                                    (typeof resume.matching_template === 'string' ? 
                                                        JSON.parse(resume.matching_template) : resume.matching_template) :
                                                    (typeof resume.vocation_template === 'string' ? 
                                                        JSON.parse(resume.vocation_template) : resume.vocation_template);
                                                
                                                if (!template) return '';
                                                
                                                let html = '';
                                                
                                                // Desired role/title
                                                if (template.desired_role || template.title || template.current_role) {
                                                    html += `
                                                    <div class="border-l-2 border-gray-200 pl-3">
                                                        <div class="text-xs text-gray-500">Desired Role</div>
                                                        <div class="text-sm font-medium">${template.desired_role || template.title || template.current_role}</div>
                                                    </div>`;
                                                }
                                                
                                                // Salary expectations
                                                if (template.compensation || template.salary_expectation) {
                                                    const comp = template.compensation || {};
                                                    const min = comp.minimum_salary || template.salary_expectation?.min;
                                                    const max = comp.maximum_salary || template.salary_expectation?.max;
                                                    if (min || max) {
                                                        html += `
                                                        <div class="border-l-2 border-gray-200 pl-3">
                                                            <div class="text-xs text-gray-500">Salary Expectation</div>
                                                            <div class="text-sm font-medium">
                                                                ${min && max ? `$${(min/1000).toFixed(0)}k - $${(max/1000).toFixed(0)}k` : 
                                                                  min ? `$${(min/1000).toFixed(0)}k+` : 
                                                                  max ? `Up to $${(max/1000).toFixed(0)}k` : ''}
                                                            </div>
                                                        </div>`;
                                                    }
                                                }
                                                
                                                // Location preferences
                                                if (template.location || template.preferred_locations) {
                                                    const locations = template.location?.preferred_locations || 
                                                                    template.preferred_locations || [];
                                                    const arrangement = template.location?.work_arrangement || 
                                                                      template.work_arrangement;
                                                    if (locations.length > 0 || arrangement) {
                                                        html += `
                                                        <div class="border-l-2 border-gray-200 pl-3">
                                                            <div class="text-xs text-gray-500">Location Preference</div>
                                                            <div class="text-sm font-medium">
                                                                ${arrangement ? arrangement : ''}
                                                                ${locations.length > 0 ? locations.join(', ') : ''}
                                                            </div>
                                                        </div>`;
                                                    }
                                                }
                                                
                                                // Employment type
                                                if (template.employment_type || template.job_type) {
                                                    html += `
                                                    <div class="border-l-2 border-gray-200 pl-3">
                                                        <div class="text-xs text-gray-500">Employment Type</div>
                                                        <div class="text-sm font-medium">${template.employment_type || template.job_type}</div>
                                                    </div>`;
                                                }
                                                
                                                // Industry preferences
                                                if (template.culture_fit?.industry_preferences?.length > 0 || 
                                                    template.industry_preferences?.length > 0) {
                                                    const industries = template.culture_fit?.industry_preferences || 
                                                                     template.industry_preferences || [];
                                                    html += `
                                                    <div class="border-l-2 border-gray-200 pl-3">
                                                        <div class="text-xs text-gray-500">Industry Preferences</div>
                                                        <div class="text-sm font-medium">${industries.join(', ')}</div>
                                                    </div>`;
                                                }
                                                
                                                // Benefits required
                                                if (template.compensation?.benefits_required?.length > 0 || 
                                                    template.benefits_required?.length > 0) {
                                                    const benefits = template.compensation?.benefits_required || 
                                                                   template.benefits_required || [];
                                                    html += `
                                                    <div class="border-l-2 border-gray-200 pl-3">
                                                        <div class="text-xs text-gray-500">Benefits Required</div>
                                                        <div class="text-sm font-medium">${benefits.join(', ')}</div>
                                                    </div>`;
                                                }
                                                
                                                return html;
                                            })()}
                                        </div>
                                    </div>
                                    ` : ''}
                                    
                                    ${(!resume.experience || resume.experience.length === 0 || resume.experience[0] === null) && 
                                      (!resume.education || resume.education.length === 0 || resume.education[0] === null) ? `
                                    <div class="detail-section">
                                        <div class="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                                            <div class="flex items-start justify-between gap-2">
                                                <div class="flex items-start gap-2">
                                                    <span class="text-yellow-600">⚠️</span>
                                                    <div>
                                                        <div class="text-sm font-medium text-yellow-800">Incomplete Processing</div>
                                                        <div class="text-xs text-yellow-700 mt-1">
                                                            Experience and education sections were not extracted. 
                                                            The resume may need reprocessing for complete data extraction.
                                                        </div>
                                                    </div>
                                                </div>
                                                <button onclick="reprocessResume('${resume.id}', '${resume.filename}')" 
                                                        class="px-3 py-1 text-xs bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 flex-shrink-0">
                                                    Reprocess
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                    ` : ''}
                                    
                                    <!-- Original Resume Text -->
                                    <div class="detail-section">
                                        <h3 class="detail-section-title">Original Resume Text</h3>
                                        <div class="detail-text">
                                            ${resume.full_text.replace(/\n/g, '<br>')}
                                        </div>
                                    </div>
                                </div>
                                <div class="modal-footer">
                                    <button class="btn btn-outline-primary" onclick="closeModal('resumeDetailModal')">Close</button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                // Add modal to body
                const existingModal = document.getElementById('resumeDetailModal');
                if (existingModal) {
                    existingModal.remove();
                }
                document.body.insertAdjacentHTML('beforeend', modalContent);
            }
        })
        .catch(error => {
            console.error('Error loading resume details:', error);
            showNotification('Failed to load resume details', 'error');
        });
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.remove();
    }
}

function setupModalHandlers() {
    // Close modal when clicking outside
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('modal')) {
            e.target.remove();
        }
    });
    
    // Close modal with Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal');
            modals.forEach(modal => modal.remove());
        }
    });
}

// Preferences Functions
async function loadPreferences() {
    if (loadingStates.preferences) return;
    loadingStates.preferences = true;
    
    try {
        const response = await fetch('/api/get_preferences');
        const data = await response.json();
        
        if (data.success) {
            const prefs = data.preferences;
            
            // Update global preferences
            currentPreferences = prefs;
            // Also save to localStorage for persistence
            localStorage.setItem('preferences', JSON.stringify(prefs));
            
            // Fill form fields with safe defaults
            document.getElementById('prefTitles').value = (prefs.preferred_titles || []).join(', ');
            document.getElementById('prefLocations').value = (prefs.locations || []).join(', ');
            document.getElementById('prefMinSalary').value = prefs.min_salary || 0;
            document.getElementById('prefMaxSalary').value = prefs.max_salary || 0;
            document.getElementById('prefWorkType').value = prefs.work_type || '';
            document.getElementById('prefDomains').value = (prefs.domains || []).join(', ');
            
            // Set weight sliders with safe defaults
            if (prefs.weights) {
                document.getElementById('weightSkills').value = prefs.weights.skills || 50;
                document.getElementById('weightExperience').value = prefs.weights.experience || 50;
                document.getElementById('weightLocation').value = prefs.weights.location || 50;
                document.getElementById('weightDomain').value = prefs.weights.domain || 50;
                document.getElementById('weightCompensation').value = prefs.weights.compensation || 50;
            }
            
            // Update displayed values
            document.querySelectorAll('input[type="range"]').forEach(slider => {
                const valueSpan = slider.parentElement.querySelector('.weight-value');
                if (valueSpan) {
                    valueSpan.textContent = slider.value;
                }
            });
        }
    } catch (error) {
        console.error('Error loading preferences:', error);
    } finally {
        loadingStates.preferences = false;
    }
}

async function savePreferences() {
    const preferences = {
        preferred_titles: document.getElementById('prefTitles').value.split(',').map(s => s.trim()),
        locations: document.getElementById('prefLocations').value.split(',').map(s => s.trim()),
        min_salary: parseInt(document.getElementById('prefMinSalary').value) || 0,
        max_salary: parseInt(document.getElementById('prefMaxSalary').value) || 0,
        work_type: document.getElementById('prefWorkType').value,
        domains: document.getElementById('prefDomains').value.split(',').map(s => s.trim()),
        weights: {
            skills: parseInt(document.getElementById('weightSkills').value),
            experience: parseInt(document.getElementById('weightExperience').value),
            location: parseInt(document.getElementById('weightLocation').value),
            domain: parseInt(document.getElementById('weightDomain').value),
            compensation: parseInt(document.getElementById('weightCompensation').value)
        }
    };
    
    try {
        const response = await fetch('/api/save_preferences', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(preferences)
        });
        
        const data = await response.json();
        if (data.success) {
            // Update global preferences
            currentPreferences = preferences;
            // Also save to localStorage for persistence
            localStorage.setItem('preferences', JSON.stringify(preferences));
            showNotification('Preferences saved successfully', 'success');
        } else {
            showNotification('Failed to save preferences', 'error');
        }
    } catch (error) {
        console.error('Error saving preferences:', error);
        showNotification('Failed to save preferences', 'error');
    }
}

async function loadAnalytics() {
    if (loadingStates.analytics) return;
    loadingStates.analytics = true;
    
    // Show loading state for stats
    const statElements = ['statResumes', 'statJobs', 'statMatches', 'statAvgScore'];
    statElements.forEach(id => {
        const elem = document.getElementById(id);
        if (elem) elem.innerHTML = '<span class="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-gray-900"></span>';
    });
    
    try {
        const response = await fetch('/api/get_analytics');
        const data = await response.json();
        
        if (data.success) {
            const analytics = data.analytics;
            
            // Update stats
            document.getElementById('statResumes').textContent = analytics.resumes_parsed;
            document.getElementById('statJobs').textContent = analytics.jobs_crawled;
            document.getElementById('statMatches').textContent = analytics.matches_found;
            document.getElementById('statAvgScore').textContent = analytics.avg_match_score;
            
            // Display top skills
            const skillsContainer = document.getElementById('topSkills');
            if (skillsContainer && analytics.top_skills) {
                skillsContainer.innerHTML = analytics.top_skills.map(item => 
                    `<span class="badge badge-blue">${item.skill} (${item.count})</span>`
                ).join('');
            }
        }
        
        // Load existing analytics reports
        await loadAnalyticsReports();
        
    } catch (error) {
        console.error('Error loading analytics:', error);
        statElements.forEach(id => {
            const elem = document.getElementById(id);
            if (elem) elem.textContent = '0';
        });
    } finally {
        loadingStates.analytics = false;
    }
}

async function loadAnalyticsReports() {
    try {
        // Get list of existing reports
        const reportsResponse = await fetch('/api/list_analytics_reports');
        const reportsData = await reportsResponse.json();
        
        const reportsGrid = document.getElementById('reportsGrid');
        if (!reportsGrid) return;
        
        if (reportsData.success && reportsData.reports && reportsData.reports.length > 0) {
            // Display report cards
            reportsGrid.innerHTML = reportsData.reports.map(report => {
                const date = new Date(report.generation_date);
                const formattedDate = date.toLocaleDateString('en-US', { 
                    month: 'short', 
                    day: 'numeric', 
                    year: 'numeric' 
                });
                
                return `
                    <div class="report-card border-2 border-gray-200 rounded-xl p-4 hover:border-blue-400 hover:shadow-md transition-all cursor-pointer bg-white"
                         onclick="loadAnalyticsReport('${report.resume_id}', '${report.resume_name.replace(/'/g, "\\'")}')"
                         data-resume-id="${report.resume_id}">
                        <div class="flex flex-col h-full">
                            <div class="mb-3 text-center">
                                <svg class="w-10 h-10 mx-auto text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                                </svg>
                            </div>
                            <div class="text-sm font-semibold text-gray-900 truncate mb-1" title="${report.resume_name}">
                                ${report.resume_name}
                            </div>
                            <div class="text-xs text-gray-500 mt-auto">
                                ${formattedDate}
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        } else {
            // No existing reports
            reportsGrid.innerHTML = `
                <div class="text-center py-8 col-span-full">
                    <div class="text-gray-400 mb-2">
                        <svg class="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                  d="M9 17v1a3 3 0 003 3h0a3 3 0 003-3v-1m-6 0h6m-3-10V4m0 0a2 2 0 100-4 2 2 0 000 4zm0 0v3m-6 4h12a2 2 0 012 2v7a2 2 0 01-2 2H6a2 2 0 01-2-2v-7a2 2 0 012-2z"></path>
                        </svg>
                    </div>
                    <p class="text-sm text-gray-500">No reports generated yet</p>
                    <p class="text-xs text-gray-400 mt-1">Click "+ Generate New Report" to create your first analytics report</p>
                </div>
            `;
            // Hide the report section
            document.getElementById('analyticsReport').classList.add('hidden');
        }
    } catch (error) {
        console.error('Error loading analytics reports:', error);
        const reportsGrid = document.getElementById('reportsGrid');
        if (reportsGrid) {
            reportsGrid.innerHTML = '<div class="text-center py-8 col-span-full text-red-500 text-sm">Error loading reports</div>';
        }
    }
}

async function loadAnalyticsReport(resumeId, resumeName) {
    try {
        // Show loading state
        showNotification(`Loading report for ${resumeName}...`, 'info');
        
        // Get the full report
        const reportResponse = await fetch(`/api/get_analytics_report/${resumeId}`);
        const reportData = await reportResponse.json();
        
        // Check different possible response structures
        if (reportData) {
            let report = null;
            
            // Check if it's the new structure with nested fields
            if (reportData.success && (reportData.statistics || reportData.llm_analysis || reportData.skills_analysis)) {
                // Transform the new structure to match what displayAnalyticsReport expects
                report = {
                    match_distribution: reportData.statistics?.match_distribution,
                    skills_gap_analysis: reportData.skills_analysis?.skill_gaps ? {
                        missing_skills: reportData.skills_analysis.skill_gaps
                    } : reportData.skills_analysis?.message,
                    common_job_titles: reportData.market_analysis?.top_job_titles || reportData.market_analysis?.common_job_titles,
                    market_insights: reportData.llm_analysis?.market_insights,
                    recommendations: reportData.llm_analysis?.recommendations,
                    // Add the missing fields
                    top_demanded_skills: reportData.skills_analysis?.top_demanded_skills,
                    resume_strengths: reportData.skills_analysis?.skill_strengths,
                    skill_gaps: reportData.skills_analysis?.skill_gaps
                };
                
            }
            // Check if it's already the report data directly
            else if (reportData.match_distribution || reportData.skills_gap_analysis || reportData.market_insights) {
                report = reportData;
            }
            // Check if it's wrapped in a success response
            else if (reportData.success && reportData.report) {
                report = reportData.report;
            }
            // Check if it has exists flag
            else if (reportData.exists && reportData.report) {
                report = reportData.report;
            }
            
            if (report) {
                // Display the report
                displayAnalyticsReport(report);
                
                // Highlight the selected report card
                document.querySelectorAll('.report-card').forEach(card => {
                    if (card.dataset.resumeId === resumeId) {
                        card.classList.add('border-blue-500', 'bg-blue-50');
                    } else {
                        card.classList.remove('border-blue-500', 'bg-blue-50');
                    }
                });
                
                // Scroll to report section
                const reportSection = document.getElementById('analyticsReport');
                if (reportSection) {
                    reportSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            } else {
                console.error('Report data structure not recognized:', reportData);
                showNotification('Report format error - please regenerate the report', 'error');
            }
        } else {
            showNotification('No report data received', 'error');
        }
    } catch (error) {
        console.error('Error loading analytics report:', error);
        showNotification('Error loading report', 'error');
    }
}

function displayAnalyticsReport(report) {
    // Show the report section
    const reportSection = document.getElementById('analyticsReport');
    if (reportSection) {
        reportSection.classList.remove('hidden');
    }
    
    // Update match distribution
    if (report.match_distribution) {
        const dist = report.match_distribution;
        document.getElementById('excellentMatches').textContent = dist.excellent || 0;
        document.getElementById('goodMatches').textContent = dist.good || 0;
        document.getElementById('fairMatches').textContent = dist.fair || 0;
        document.getElementById('poorMatches').textContent = dist.poor || 0;
    }
    
    // Update skills analysis
    if (report.skills_gap_analysis) {
        const skillsContainer = document.getElementById('skillsGapAnalysis');
        if (skillsContainer) {
            const gapAnalysis = report.skills_gap_analysis;
            
            // Check if it's a message or actual data
            if (typeof gapAnalysis === 'string') {
                skillsContainer.innerHTML = `
                    <p class="text-sm text-gray-600">${gapAnalysis}</p>
                `;
            } else if (gapAnalysis.missing_skills && gapAnalysis.missing_skills.length > 0) {
                skillsContainer.innerHTML = gapAnalysis.missing_skills
                    .slice(0, 8)
                    .map(skill => `
                        <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 border border-yellow-200">
                            ${skill}
                        </span>
                    `).join('');
            } else {
                skillsContainer.innerHTML = '<p class="text-sm text-gray-600">No significant skill gaps identified</p>';
            }
        }
    }
    
    // Update job titles section
    if (report.common_job_titles && Array.isArray(report.common_job_titles)) {
        const titlesContainer = document.getElementById('topJobTitles');  // Fixed ID to match HTML
        if (titlesContainer) {
            titlesContainer.innerHTML = report.common_job_titles
                .slice(0, 5)
                .map(item => `
                    <div class="flex items-center justify-between bg-gray-50 px-4 py-2 rounded-lg">
                        <span class="text-sm font-medium text-gray-700">${item.title}</span>
                        <span class="text-xs bg-indigo-100 text-indigo-800 px-2 py-1 rounded-full">${item.count} openings</span>
                    </div>
                `).join('');
        }
    }
    
    // Update top demanded skills
    if (report.top_demanded_skills && Array.isArray(report.top_demanded_skills)) {
        const skillsContainer = document.getElementById('topDemandedSkills');
        if (skillsContainer) {
            skillsContainer.innerHTML = report.top_demanded_skills
                .slice(0, 10)
                .map(skill => {
                    const skillName = typeof skill === 'object' ? skill.skill || skill.name : skill;
                    const count = typeof skill === 'object' ? skill.count : null;
                    return `
                        <div class="flex items-center justify-between bg-gray-50 px-4 py-2 rounded-lg">
                            <span class="text-sm font-medium text-gray-700">${skillName}</span>
                            ${count ? `<span class="text-xs text-gray-500">${count} jobs</span>` : ''}
                        </div>
                    `;
                }).join('');
        }
    }
    
    // Update skill gaps
    if (report.skill_gaps && Array.isArray(report.skill_gaps)) {
        const gapsContainer = document.getElementById('skillGaps');
        if (gapsContainer) {
            gapsContainer.innerHTML = report.skill_gaps
                .slice(0, 8)
                .map(skill => `
                    <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800 border border-red-200">
                        ${skill}
                    </span>
                `).join('');
        }
    }
    
    // Update resume strengths
    if (report.resume_strengths && Array.isArray(report.resume_strengths)) {
        const strengthsContainer = document.getElementById('skillStrengths');
        if (strengthsContainer) {
            strengthsContainer.innerHTML = report.resume_strengths
                .slice(0, 8)
                .map(skill => `
                    <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 border border-green-200">
                        ${skill}
                    </span>
                `).join('');
        }
    }
    
    // Update LLM insights
    if (report.market_insights || report.recommendations) {
        const container = document.getElementById('llmInsights');
        if (container) {
            let insightsHTML = '';
            
            if (report.market_insights) {
                // Format the text for better display
                let marketText = report.market_insights
                    .replace(/\*\*/g, '') // Remove markdown bold
                    .replace(/\n\n/g, '</p><p class="text-sm text-gray-700 leading-relaxed mt-3">') // Convert double newlines to paragraphs
                    .replace(/\n/g, '<br>'); // Convert single newlines to line breaks
                
                insightsHTML += `
                    <div class="bg-white rounded-xl p-5 border border-purple-200 mb-4">
                        <h4 class="font-semibold text-purple-900 mb-3 text-base">Market Insights</h4>
                        <div class="prose prose-sm max-w-none">
                            <p class="text-sm text-gray-700 leading-relaxed">${marketText}</p>
                        </div>
                    </div>
                `;
            }
            
            if (report.recommendations) {
                // Format the text for better display
                let recText = report.recommendations
                    .replace(/\*\*/g, '') // Remove markdown bold
                    .replace(/\n\n/g, '</p><p class="text-sm text-gray-700 leading-relaxed mt-3">') // Convert double newlines to paragraphs
                    .replace(/\n/g, '<br>') // Convert single newlines to line breaks
                    .replace(/<br><br>/g, '<br>'); // Clean up double line breaks
                
                insightsHTML += `
                    <div class="bg-white rounded-xl p-5 border border-green-200">
                        <h4 class="font-semibold text-green-900 mb-3 text-base">Recommendations</h4>
                        <div class="prose prose-sm max-w-none">
                            <p class="text-sm text-gray-700 leading-relaxed">${recText}</p>
                        </div>
                    </div>
                `;
            }
            
            container.innerHTML = insightsHTML || '<p class="text-sm text-gray-600">No insights available</p>';
        }
    }
}

let selectedResumeId = null;

async function showResumeSelectionModal() {
    const modal = document.getElementById('resumeSelectionModal');
    const resumeList = document.getElementById('resumeList');
    
    // Load resumes
    resumeList.innerHTML = '<p class="text-gray-500">Loading resumes...</p>';
    modal.classList.remove('hidden');
    
    try {
        const response = await fetch('/api/get_resumes');
        const data = await response.json();
        
        if (data.success && data.resumes.length > 0) {
            console.log('Resumes data received:', data.resumes);
            
            // Check for existing reports
            const reportsResponse = await fetch('/api/list_analytics_reports');
            const reportsData = await reportsResponse.json();
            const existingReports = reportsData.success ? reportsData.reports : [];
            
            resumeList.innerHTML = data.resumes.map(resume => {
                console.log('Processing resume:', resume);
                const resumeId = resume.id || resume.resume_id;  // Handle both field names
                console.log('Resume ID extracted:', resumeId);
                const existingReport = existingReports.find(r => r.resume_id === resumeId);
                return `
                    <label class="flex items-start p-3 border rounded-lg hover:bg-gray-50 cursor-pointer">
                        <input type="radio" name="resume" value="${resumeId}" 
                               data-resume-id="${resumeId}"
                               onchange="selectResume(this)"
                               class="mt-1 mr-3">
                        <div class="flex-1">
                            <div class="font-medium text-gray-900">${resume.name || 'Unknown'}</div>
                            <div class="text-sm text-gray-600">${resume.filename || 'No filename'}</div>
                            ${existingReport ? `
                                <div class="text-xs text-green-600 mt-1">
                                    Report exists (${existingReport.generation_date})
                                </div>
                            ` : ''}
                        </div>
                    </label>
                `;
            }).join('');
            
            // Update existing reports info
            updateExistingReportsInfo(existingReports);
        } else {
            resumeList.innerHTML = '<p class="text-gray-500">No resumes found. Please upload a resume first.</p>';
        }
    } catch (error) {
        console.error('Error loading resumes:', error);
        resumeList.innerHTML = '<p class="text-red-500">Error loading resumes</p>';
    }
}

function selectResume(element) {
    console.log('selectResume called with element:', element);
    console.log('Element value:', element.value);
    console.log('Element data-resume-id:', element.getAttribute('data-resume-id'));
    
    selectedResumeId = element.getAttribute('data-resume-id') || element.value;
    console.log('Selected resume ID set to:', selectedResumeId);
    
    if (!selectedResumeId || selectedResumeId === 'undefined') {
        console.error('Invalid resume ID:', selectedResumeId);
        showNotification('Error: Invalid resume selected', 'error');
        return;
    }
    
    document.getElementById('confirmGenerateBtn').disabled = false;
}

function closeResumeSelectionModal() {
    document.getElementById('resumeSelectionModal').classList.add('hidden');
    selectedResumeId = null;
    document.getElementById('confirmGenerateBtn').disabled = true;
}

async function generateReportForSelectedResume() {
    if (!selectedResumeId) {
        showNotification('Please select a resume first', 'error');
        return;
    }
    
    console.log('Generating report for resume:', selectedResumeId);
    
    // Save the resume ID before closing modal (which resets it)
    const resumeIdToUse = selectedResumeId;
    
    // Close modal and generate report
    closeResumeSelectionModal();
    await generateAnalyticsReport(resumeIdToUse);
}

function updateExistingReportsInfo(reports) {
    // This function is no longer needed with the new card-based UI
    // Keeping it for backward compatibility
    return;
}

async function generateAnalyticsReport(resumeId) {
    if (!resumeId) {
        showNotification('Resume ID is required', 'error');
        return;
    }
    
    const btn = document.getElementById('generateReportBtn');
    const btnText = document.getElementById('reportBtnText');
    const btnLoader = document.getElementById('reportBtnLoader');
    const reportSection = document.getElementById('analyticsReport');
    
    // Show loading state
    btn.disabled = true;
    btnText.classList.add('hidden');
    btnLoader.classList.remove('hidden');
    
    console.log('Sending request with resume_id:', resumeId);
    
    try {
        const response = await fetch('/api/generate_analytics_report', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ resume_id: resumeId })
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const errorData = await response.json();
            console.error('Error response:', errorData);
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        
        const report = await response.json();
        console.log('Report received:', report);
        
        if (report.success) {
            // Show report section
            reportSection.classList.remove('hidden');
            
            // Update match distribution
            if (report.statistics && report.statistics.match_distribution) {
                const dist = report.statistics.match_distribution;
                document.getElementById('excellentMatches').textContent = dist.excellent || 0;
                document.getElementById('goodMatches').textContent = dist.good || 0;
                document.getElementById('fairMatches').textContent = dist.fair || 0;
                document.getElementById('poorMatches').textContent = dist.poor || 0;
            }
            
            // Update top demanded skills
            if (report.skills_analysis) {
                const skillsAnalysis = report.skills_analysis;
                
                // Check if there's a message about missing skills data
                if (skillsAnalysis.message) {
                    // Show message in all skill containers
                    const message = `<p class="text-sm text-gray-500 italic">${skillsAnalysis.message}</p>`;
                    document.getElementById('topDemandedSkills').innerHTML = message;
                    document.getElementById('topResumeSkills').innerHTML = message;
                    document.getElementById('skillGaps').innerHTML = `<span class="text-sm text-gray-500 italic">No data available</span>`;
                    document.getElementById('skillStrengths').innerHTML = `<span class="text-sm text-gray-500 italic">No data available</span>`;
                } else {
                    // Update top demanded skills
                    if (skillsAnalysis.top_demanded_skills && skillsAnalysis.top_demanded_skills.length > 0) {
                        const container = document.getElementById('topDemandedSkills');
                        container.innerHTML = skillsAnalysis.top_demanded_skills.slice(0, 10).map((item, index) => `
                            <div class="flex items-center justify-between p-2 rounded-lg ${index % 2 === 0 ? 'bg-gray-50' : ''}">
                                <span class="text-sm font-medium text-gray-700">${index + 1}. ${item.skill}</span>
                                <span class="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">${item.count} jobs</span>
                            </div>
                        `).join('');
                    } else {
                        document.getElementById('topDemandedSkills').innerHTML = '<p class="text-sm text-gray-500">No skills data found in job postings</p>';
                    }
                    
                    // Update skill gaps
                    if (skillsAnalysis.skill_gaps && skillsAnalysis.skill_gaps.length > 0) {
                        const container = document.getElementById('skillGaps');
                        container.innerHTML = skillsAnalysis.skill_gaps.map(skill => 
                            `<span class="px-3 py-1 text-sm bg-red-100 text-red-700 rounded-full border border-red-200">${skill}</span>`
                        ).join('');
                    } else {
                        document.getElementById('skillGaps').innerHTML = '<span class="text-sm text-gray-500">No skill gaps identified</span>';
                    }
                    
                    // Update skill strengths
                    if (skillsAnalysis.skill_strengths && skillsAnalysis.skill_strengths.length > 0) {
                        const container = document.getElementById('skillStrengths');
                        container.innerHTML = skillsAnalysis.skill_strengths.map(skill => 
                            `<span class="px-3 py-1 text-sm bg-green-100 text-green-700 rounded-full border border-green-200">${skill}</span>`
                        ).join('');
                    } else {
                        document.getElementById('skillStrengths').innerHTML = '<span class="text-sm text-gray-500">No aligned skills identified</span>';
                    }
                }
            }
            
            // Update top job titles
            if (report.market_analysis && report.market_analysis.top_job_titles) {
                const container = document.getElementById('topJobTitles');
                container.innerHTML = report.market_analysis.top_job_titles.slice(0, 8).map((item, index) => `
                    <div class="flex items-center justify-between p-2 rounded-lg ${index % 2 === 0 ? 'bg-gray-50' : ''}">
                        <span class="text-sm font-medium text-gray-700">${item.title}</span>
                        <span class="text-xs bg-indigo-100 text-indigo-800 px-2 py-1 rounded-full">${item.count} openings</span>
                    </div>
                `).join('');
            }
            
            // Update LLM insights - Only show Market Insights and Recommendations
            if (report.llm_analysis) {
                const container = document.getElementById('llmInsights');
                const analysis = report.llm_analysis;
                
                let insightsHTML = '';
                
                // Handle different response formats
                if (typeof analysis === 'object') {
                    // Show only Market Insights and Recommendations
                    if (analysis.market_insights) {
                        // Format the text for better display
                        let marketText = analysis.market_insights
                            .replace(/\*\*/g, '') // Remove markdown bold
                            .replace(/\n\n/g, '</p><p class="text-sm text-gray-700 leading-relaxed mt-3">') // Convert double newlines to paragraphs
                            .replace(/\n/g, '<br>'); // Convert single newlines to line breaks
                        
                        insightsHTML += `
                            <div class="bg-white rounded-xl p-5 border border-purple-200 mb-4">
                                <h4 class="font-semibold text-purple-900 mb-3 text-base">Market Insights</h4>
                                <div class="prose prose-sm max-w-none">
                                    <p class="text-sm text-gray-700 leading-relaxed">${marketText}</p>
                                </div>
                            </div>
                        `;
                    }
                    
                    if (analysis.recommendations) {
                        // Format the text for better display
                        let recText = analysis.recommendations
                            .replace(/\*\*/g, '') // Remove markdown bold
                            .replace(/\n\n/g, '</p><p class="text-sm text-gray-700 leading-relaxed mt-3">') // Convert double newlines to paragraphs
                            .replace(/\n/g, '<br>') // Convert single newlines to line breaks
                            .replace(/<br><br>/g, '<br>'); // Clean up double line breaks
                        
                        insightsHTML += `
                            <div class="bg-white rounded-xl p-5 border border-green-200">
                                <h4 class="font-semibold text-green-900 mb-3 text-base">Recommendations</h4>
                                <div class="prose prose-sm max-w-none">
                                    <p class="text-sm text-gray-700 leading-relaxed">${recText}</p>
                                </div>
                            </div>
                        `;
                    }
                } else {
                    // Plain text response
                    insightsHTML = `
                        <div class="bg-white rounded-xl p-4 border border-purple-200">
                            <p class="text-sm text-gray-700 whitespace-pre-wrap">${analysis}</p>
                        </div>
                    `;
                }
                
                container.innerHTML = insightsHTML || '<p class="text-sm text-gray-600">Analysis complete. Insights are being processed...</p>';
            }
            
            showNotification('Analytics report generated successfully!', 'success');
            
            // Refresh the reports grid to show the new report
            await loadAnalyticsReports();
            
            // Auto-select the newly generated report card
            const newReportCard = document.querySelector(`[data-resume-id="${resumeId}"]`);
            if (newReportCard) {
                newReportCard.classList.add('border-blue-500', 'bg-blue-50');
            }
        } else {
            showNotification('Failed to generate analytics report: ' + (report.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        console.error('Error generating analytics report:', error);
        showNotification('Error generating analytics report', 'error');
    } finally {
        // Reset button state
        btn.disabled = false;
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
    }
}

// Job Search
async function startJobSearch() {
    const query = document.getElementById('searchQuery').value;
    const location = document.getElementById('searchLocation').value || 'remote';
    const maxJobs = document.getElementById('searchMaxJobs').value || 20;
    
    if (!query) {
        showNotification('Please enter search keywords', 'error');
        return;
    }
    
    // Show initial status message
    updateSearchStatus(`Starting search for "${query}"...`);
    
    // Store current search process ID globally
    window.currentSearchProcessId = null;
    
    try {
        const response = await fetch('/api/search_jobs', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ query, location, max_jobs: maxJobs })
        });
        
        const data = await response.json();
        if (data.success) {
            window.currentSearchProcessId = data.process_id;
            activeProcesses.set(data.process_id, {
                type: 'search',
                query: query,
                startTime: new Date()
            });
            
            // Start polling as backup to WebSocket
            startProcessPolling(data.process_id, 'search');
            
            // Update status with process info using the correct function
            updateSearchStatus('Initializing job search...');
            
            // Show search details
            const searchDetails = document.getElementById('searchProgress');
            if (searchDetails) {
                searchDetails.innerHTML = `Query: ${query} | Location: ${location} | Max: ${maxJobs}`;
            }
        } else {
            showNotification(data.error || 'Search failed', 'error');
            updateSearchStatus('Search failed');
        }
    } catch (error) {
        console.error('Search error:', error);
        showNotification('Search failed', 'error');
        updateSearchStatus('Search failed');
    }
}

// Toggle match details expansion
function toggleMatchDetails(matchId) {
    const detailsDiv = document.getElementById(matchId);
    if (detailsDiv) {
        detailsDiv.classList.toggle('hidden');
        
        // Update chevron icon if present
        const chevron = detailsDiv.previousElementSibling.querySelector('.chevron-icon');
        if (chevron) {
            chevron.classList.toggle('rotate-180');
        }
    }
}

// Toggle job/match visibility
function toggleJobVisibility(jobId) {
    if (hiddenJobIds.has(jobId)) {
        hiddenJobIds.delete(jobId);
    } else {
        hiddenJobIds.add(jobId);
    }
    
    // Save to localStorage
    localStorage.setItem('hiddenJobIds', JSON.stringify(Array.from(hiddenJobIds)));
    
    // Refresh both job and match displays
    displayJobs(currentJobs);
    displayMatches(currentMatches);
}

// Toggle showing hidden items
function toggleShowHidden() {
    showHiddenItems = !showHiddenItems;
    
    // Save state to localStorage
    localStorage.setItem('showHiddenItems', showHiddenItems.toString());
    
    // Update button text on both tabs
    const jobToggle = document.getElementById('toggleHiddenJobs');
    const matchToggle = document.getElementById('toggleHiddenMatches');
    
    if (jobToggle) {
        jobToggle.textContent = showHiddenItems ? 'Hide Archived' : 'Show Archived';
        jobToggle.classList.toggle('bg-gray-500', !showHiddenItems);
        jobToggle.classList.toggle('bg-blue-500', showHiddenItems);
    }
    
    if (matchToggle) {
        matchToggle.textContent = showHiddenItems ? 'Hide Archived' : 'Show Archived';
        matchToggle.classList.toggle('bg-gray-500', !showHiddenItems);
        matchToggle.classList.toggle('bg-blue-500', showHiddenItems);
    }
    
    // Refresh displays
    displayJobs(currentJobs);
    displayMatches(currentMatches);
}

// Load hidden job IDs from localStorage on startup
function loadHiddenJobIds() {
    const saved = localStorage.getItem('hiddenJobIds');
    if (saved) {
        try {
            hiddenJobIds = new Set(JSON.parse(saved));
        } catch (e) {
            hiddenJobIds = new Set();
        }
    }
    
    // Also load the show/hide state
    const showHiddenSaved = localStorage.getItem('showHiddenItems');
    if (showHiddenSaved !== null) {
        showHiddenItems = showHiddenSaved === 'true';
    }
}

// Matching
async function runMatching() {
    // Set matching in progress flag
    matchingInProgress = true;
    matchingStatusMessage = 'Initializing matching engine...';
    
    // Show spinner and status in the matches section
    const matchesContent = document.getElementById('matchesContent');
    const originalContent = matchesContent ? matchesContent.innerHTML : '';
    
    if (matchesContent) {
        matchesContent.innerHTML = `
            <div class="border rounded-2xl overflow-hidden bg-white">
                <div class="p-6">
                    <div class="flex flex-col items-center justify-center">
                        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
                        <p class="text-gray-700 font-semibold text-lg">Running Matching Algorithm</p>
                        <p class="text-gray-500 text-sm mt-1">Analyzing compatibility between resumes and jobs...</p>
                        <div id="matchingStatus" class="mt-6 w-full max-w-md space-y-3 text-sm">
                            <div class="flex items-center gap-3 p-3 bg-blue-50 rounded-lg animate-fadeIn">
                                <div class="animate-pulse w-2 h-2 bg-blue-600 rounded-full"></div>
                                <span class="text-gray-700">${matchingStatusMessage}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    try {
        const response = await fetch('/api/run_matching', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({})
        });
        
        const data = await response.json();
        if (data.success) {
            showNotification('Matching process started', 'success');
            activeProcesses.set(data.process_id, {
                type: 'matching',
                startTime: new Date()
            });
            
            // Update status message
            updateMatchingStatus('Loading resumes and jobs from database...');
            
            // Don't show process modal - just keep the spinner
            // Polling will update when complete
            startProcessPolling(data.process_id, 'matching');
        } else {
            showNotification('Matching failed', 'error');
            if (matchesContent) {
                matchesContent.innerHTML = originalContent;
            }
        }
    } catch (error) {
        console.error('Matching error:', error);
        showNotification('Matching failed', 'error');
        if (matchesContent) {
            matchesContent.innerHTML = originalContent;
        }
    }
}

// Update matching status display
function updateMatchingStatus(message) {
    // Update global status
    matchingStatusMessage = message;
    
    const statusDiv = document.getElementById('matchingStatus');
    if (statusDiv) {
        // Choose icon based on message content
        let iconColor = 'bg-blue-600';
        if (message.toLowerCase().includes('complete') || message.toLowerCase().includes('success')) {
            matchingInProgress = false;
            iconColor = 'bg-green-600';
        } else if (message.toLowerCase().includes('error') || message.toLowerCase().includes('fail')) {
            iconColor = 'bg-red-600';
        }
        
        // Replace entire content with single status message
        statusDiv.innerHTML = `
            <div class="flex items-center gap-3 p-3 bg-blue-50 rounded-lg animate-fadeIn">
                <div class="animate-pulse w-2 h-2 ${iconColor} rounded-full"></div>
                <span class="text-gray-700">${message}</span>
            </div>
        `;
    }
}

// Export Matches
async function exportMatches() {
    window.location.href = '/api/export_matches';
}

// Socket.IO Event Listeners
function setupSocketListeners() {
    socket.on('connect', () => {
        console.log('Connected to server');
        isConnected = true;
        startHeartbeat();
        // Clear any connection error notifications
        clearConnectionError();
    });
    
    socket.on('disconnect', (reason) => {
        console.log(`Disconnected from server: ${reason}`);
        isConnected = false;
        stopHeartbeat();
        // Don't show error for normal disconnections
        if (reason === 'io server disconnect') {
            // Server disconnected us, will not automatically reconnect
            socket.connect();
        }
        // For other reasons, socket.io will handle reconnection automatically
    });
    
    socket.on('connect_error', (error) => {
        // Silent fail - don't spam console
        if (error.message !== 'xhr poll error') {
            console.log('Connection error:', error.message);
        }
        isConnected = false;
    });
    
    socket.on('reconnect', (attemptNumber) => {
        console.log(`Reconnected after ${attemptNumber} attempts`);
        isConnected = true;
        clearConnectionError();
    });
    
    socket.on('reconnect_failed', () => {
        console.log('Failed to reconnect after maximum attempts');
        isConnected = false;
        // Optionally show a user notification here
        startHeartbeat();
        // Refresh data when reconnected
        if (activeTab === 'upload') {
            loadResumes();
        } else if (activeTab === 'search') {
            loadJobs();
        } else if (activeTab === 'matches') {
            loadMatches();
        }
    });
    
    socket.on('pong', (data) => {
        lastPongTime = Date.now();
        console.log('Received pong:', data.timestamp);
    });
    
    socket.on('process_status', (data) => {
        // Update any status displays
        if (data.process_id.startsWith('resume_')) {
            updateUploadStatus(data.message);
        } else if (data.process_id.startsWith('search_')) {
            // Update search status with the new styled function
            updateSearchStatus(data.message);
        } else if (data.process_id.startsWith('match_')) {
            // Update matching status inline
            updateMatchingStatus(data.message);
        }
    });
    
    socket.on('process_output', (data) => {
        appendProcessOutput(data.process_id, data.output);
        
        // Check for structured job data
        if (data.output && data.output.startsWith('[JOB_SAVED]')) {
            try {
                const jobData = JSON.parse(data.output.substring('[JOB_SAVED]'.length).trim());
                // Add job to the jobs list immediately
                addJobToList(jobData);
                // Also trigger a refresh of matches if we have resumes
                if (window.resumeCount > 0) {
                    setTimeout(() => loadMatches(), 1000); // Give time for matches to be calculated
                }
            } catch (e) {
                console.error('Failed to parse job data:', e);
            }
        }
        
        // Check for structured match data
        if (data.output && data.output.startsWith('[MATCH_FOUND]')) {
            try {
                const matchData = JSON.parse(data.output.substring('[MATCH_FOUND]'.length).trim());
                // Add match to the matches list immediately
                addMatchToList(matchData);
            } catch (e) {
                console.error('Failed to parse match data:', e);
            }
        }
        
        // Auto-refresh matches when new matches are found or saved
        // Simplified detection - just look for "Saved X matches"
        if (data.output && data.output.includes('Saved') && data.output.includes('matches')) {
            console.log('Matches saved detected:', data.output);
            // Only refresh if we're not in the middle of matching
            if (!matchingInProgress) {
                // Refresh matches tab if it's active
                if (activeTab === 'matches') {
                    // Debounce rapid updates to prevent too many refreshes
                    clearTimeout(window.matchRefreshTimeout);
                    window.matchRefreshTimeout = setTimeout(() => {
                        console.log('Auto-refreshing matches tab');
                        loadMatches();
                    }, 1000); // Wait 1 second to batch multiple saves
                } else {
                    // Store flag to refresh when user switches to matches tab
                    console.log('Storing pending match refresh for later');
                    window.pendingMatchRefresh = true;
                }
            }
        }
        
        // Parse output for resume processing
        if (data.process_id.startsWith('resume_')) {
            const output = data.output.toLowerCase();
            
            if (output.includes('processing') && output.includes('resume')) {
                updateUploadStatus('Processing resume structure and content...');
            } else if (output.includes('extracting') && output.includes('skills')) {
                updateUploadStatus('Extracting skills and experience...');
            } else if (output.includes('analyzing')) {
                updateUploadStatus('Analyzing career history and education...');
            } else if (output.includes('generating') && output.includes('embeddings')) {
                updateUploadStatus('Generating semantic embeddings...');
            } else if (output.includes('saved') || output.includes('stored')) {
                updateUploadStatus('Resume saved to database');
            } else if (output.includes('complete')) {
                updateUploadStatus('Resume processing complete!');
            }
        }
        
        // Parse output for job search
        else if (data.process_id.startsWith('search_')) {
            const output = data.output.toLowerCase();
            
            if (output.includes('launching') || output.includes('starting browser')) {
                updateSearchStatus('Launching browser for Google Jobs...');
            } else if (output.includes('navigating') || output.includes('searching')) {
                updateSearchStatus('Searching Google Jobs...');
            } else if (output.includes('found') && output.includes('job')) {
                const jobMatch = output.match(/found (\d+) jobs?/);
                if (jobMatch) {
                    updateSearchStatus(`Found ${jobMatch[1]} job listing(s)`);
                }
            } else if (output.includes('crawling') || output.includes('fetching')) {
                const crawlMatch = output.match(/job (\d+)\/(\d+)/);
                if (crawlMatch) {
                    updateSearchStatus(`Fetching job ${crawlMatch[1]} of ${crawlMatch[2]}...`);
                } else {
                    updateSearchStatus('Fetching job details...');
                }
            } else if (output.includes('processing') && output.includes('job')) {
                const processMatch = output.match(/processing job (\d+)/);
                if (processMatch) {
                    updateSearchStatus(`Processing job ${processMatch[1]}...`);
                } else {
                    updateSearchStatus('Processing job information...');
                }
            } else if (output.includes('extracting')) {
                updateSearchStatus('Extracting job requirements and details...');
            } else if (output.includes('saved') && output.includes('database')) {
                const savedMatch = output.match(/saved (\d+) jobs?/);
                if (savedMatch) {
                    updateSearchStatus(`Saved ${savedMatch[1]} job(s) to database`);
                }
            } else if (output.includes('complete')) {
                updateSearchStatus('Job search complete!');
            }
        }
        
        // Also update matching status for certain output patterns
        else if (data.process_id.startsWith('match_')) {
            const output = data.output.toLowerCase();
            
            // Parse specific messages for better status updates
            if (output.includes('loaded') && output.includes('resumes')) {
                const resumeMatch = output.match(/loaded (\d+) resumes?/);
                if (resumeMatch) {
                    updateMatchingStatus(`Loaded ${resumeMatch[1]} resume(s) from database`);
                }
            } else if (output.includes('loaded') && output.includes('jobs')) {
                const jobMatch = output.match(/loaded (\d+) jobs?/);
                if (jobMatch) {
                    updateMatchingStatus(`Loaded ${jobMatch[1]} job(s) from database`);
                }
            } else if (output.includes('calculating compatibility')) {
                const calcMatch = output.match(/(\d+) resumes? × (\d+) jobs?/);
                if (calcMatch) {
                    updateMatchingStatus(`Analyzing ${calcMatch[1]} resume(s) against ${calcMatch[2]} job(s)...`);
                }
            } else if (output.includes('processing resume')) {
                const progressMatch = output.match(/processing resume (\d+)\/(\d+)/);
                if (progressMatch) {
                    updateMatchingStatus(`Processing resume ${progressMatch[1]} of ${progressMatch[2]}...`);
                }
            } else if (output.includes('found') && output.includes('matches')) {
                const matchesFound = output.match(/found (\d+) .*matches/);
                if (matchesFound) {
                    updateMatchingStatus(`Found ${matchesFound[1]} potential matches`);
                }
            } else if (output.includes('saved') && output.includes('matches')) {
                const savedMatch = output.match(/saved (\d+) matches/);
                if (savedMatch) {
                    updateMatchingStatus(`Saved ${savedMatch[1]} matches to database`);
                }
            } else if (output.includes('starting enhanced matching')) {
                updateMatchingStatus('Starting advanced matching analysis...');
            } else if (output.includes('initializing matching engine')) {
                updateMatchingStatus('Initializing AI matching engine...');
            } else if (output.includes('performing matching analysis')) {
                updateMatchingStatus('Performing deep compatibility analysis...');
            }
        }
    });
    
    socket.on('process_complete', (data) => {
        console.log('Process complete event received:', data);
        const process = activeProcesses.get(data.process_id);
        
        // If we don't have the process in our map, try to determine type from the data
        const processType = process ? process.type : data.type;
        
        if (process || data.type) {
            const successMsg = data.success ? 'completed successfully' : 'failed';
            showNotification(`${processType} process ${successMsg}`, data.success ? 'success' : 'error');
            
            // Refresh the page for job search completion
            if (processType === 'Job search' && data.success) {
                // Instead of full page reload, just refresh the job data
                setTimeout(() => {
                    loadJobs();  // Refresh job list
                    showNotification('Job search completed! Jobs have been loaded.', 'success');
                }, 1500);
            }
            
            if (process) {
                activeProcesses.delete(data.process_id);
            }
            
            // Refresh relevant data based on process type
            if (processType && processType.toLowerCase().includes('resume')) {
                if (data.success) {
                    console.log('Refreshing resume list after processing...');
                    updateUploadStatus(data.message || 'Processing complete');
                    // Refresh resume list instead of full page reload
                    setTimeout(() => {
                        loadResumes();  // Refresh resume list
                        showNotification('Resume processed successfully!', 'success');
                    }, 1500);
                }
            } else if (processType === 'Job search') {
                if (data.success) {
                    // Already handled above with page refresh
                    document.getElementById('searchStatus').innerHTML = `
                        <div class="flex items-center gap-2">
                            <span class="text-green-600">✅</span>
                            <span class="text-sm text-green-600">Search completed successfully</span>
                        </div>
                    `;
                } else {
                    document.getElementById('searchStatus').innerHTML = `
                        <div class="flex items-center gap-2">
                            <span class="text-red-600">❌</span>
                            <span class="text-sm text-red-600">Search failed</span>
                        </div>
                    `;
                }
            } else if (processType && processType.toLowerCase().includes('match')) {
                if (data.success) {
                    // Update status to show completion
                    updateMatchingStatus('Matching completed successfully!');
                    matchingInProgress = false;
                    
                    // Refresh match list after a brief delay to show the success message
                    setTimeout(() => {
                        loadMatches();  // Refresh match list
                        showNotification('Matching completed successfully!', 'success');
                        // Close the process modal if it's open
                        closeProcessModal();
                    }, 2000);
                } else {
                    // Update status to show failure
                    updateMatchingStatus('✗ Matching failed - please check the logs');
                    matchingInProgress = false;
                    
                    // On failure, also refresh to remove spinner after delay
                    setTimeout(() => {
                        loadMatches();
                        showNotification('Matching failed', 'error');
                    }, 3000);
                }
            }
        }
    });
    
    socket.on('process_error', (data) => {
        showNotification(`Process error: ${data.error}`, 'error');
        activeProcesses.delete(data.process_id);
    });
}

// Start polling for process completion as backup to WebSocket
function startProcessPolling(processId, processType) {
    // Clear any existing polling for this process
    if (pollingIntervals.has(processId)) {
        clearInterval(pollingIntervals.get(processId));
    }
    
    // Poll every 5 seconds
    const interval = setInterval(() => {
        // Check if process is still active
        if (!activeProcesses.has(processId)) {
            clearInterval(interval);
            pollingIntervals.delete(processId);
            
            // Refresh appropriate data based on process type
            if (processType === 'resume') {
                loadResumes();
            } else if (processType === 'search' || processType === 'Job search') {
                loadJobs();
            } else if (processType === 'match' || processType === 'Matching') {
                loadMatches();
            }
        }
    }, 5000);
    
    pollingIntervals.set(processId, interval);
}

// Process Modal
function showProcessModal(processId, title) {
    const modal = document.getElementById('processModal');
    const titleElement = document.getElementById('processTitle');
    const outputElement = document.getElementById('processOutput');
    const stopBtn = document.getElementById('stopProcessBtn');
    
    if (modal && titleElement && outputElement) {
        titleElement.textContent = title;
        outputElement.textContent = 'Starting process...\n';
        modal.classList.remove('hidden');
        
        // Set up stop button
        if (stopBtn) {
            stopBtn.onclick = () => stopProcess(processId);
        }
    }
}

function closeProcessModal() {
    const modal = document.getElementById('processModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

function appendProcessOutput(processId, output) {
    const outputElement = document.getElementById('processOutput');
    if (outputElement) {
        outputElement.textContent += output + '\n';
        outputElement.scrollTop = outputElement.scrollHeight;
    }
}

async function stopProcess(processId) {
    try {
        const response = await fetch(`/api/stop_process/${processId}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        if (data.success) {
            showNotification('Process stopped', 'info');
            activeProcesses.delete(processId);
            closeProcessModal();
        }
    } catch (error) {
        console.error('Error stopping process:', error);
    }
}

// Utility Functions
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `fixed top-20 right-4 px-4 py-3 rounded-lg shadow-lg z-50 ${
        type === 'success' ? 'bg-green-500 text-white' :
        type === 'error' ? 'bg-red-500 text-white' :
        'bg-blue-500 text-white'
    }`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

function updateUploadStatus(message) {
    const statusDiv = document.getElementById('uploadStatus');
    if (statusDiv) {
        // Check if this is the initial "No files processing" message
        if (statusDiv.innerHTML.includes('No files processing')) {
            statusDiv.innerHTML = ''; // Clear the initial message
        }
        
        // Add new status message with animation
        const newStatus = document.createElement('div');
        newStatus.className = 'flex items-center gap-3 p-3 bg-blue-50 rounded-lg animate-fadeIn';
        
        // Choose icon color based on message content
        let iconColor = 'bg-blue-600';
        if (message.toLowerCase().includes('success') || message.toLowerCase().includes('complete')) {
            iconColor = 'bg-green-600';
        } else if (message.toLowerCase().includes('error') || message.toLowerCase().includes('fail')) {
            iconColor = 'bg-red-600';
        }
        
        newStatus.innerHTML = `
            <div class="animate-pulse w-2 h-2 ${iconColor} rounded-full"></div>
            <span class="text-gray-700">${message}</span>
        `;
        statusDiv.appendChild(newStatus);
        
        // Keep only last 5 status messages
        while (statusDiv.children.length > 5) {
            statusDiv.removeChild(statusDiv.firstChild);
        }
        
        // Scroll to the latest message
        statusDiv.scrollTop = statusDiv.scrollHeight;
    }
}

// Update search status display
function updateSearchStatus(message) {
    const statusDiv = document.getElementById('searchStatus');
    if (statusDiv) {
        // Choose icon color based on message content
        let iconColor = 'bg-blue-600';
        if (message.toLowerCase().includes('success') || message.toLowerCase().includes('complete') || message.toLowerCase().includes('found')) {
            iconColor = 'bg-green-600';
        } else if (message.toLowerCase().includes('error') || message.toLowerCase().includes('fail')) {
            iconColor = 'bg-red-600';
        }
        
        // Replace entire content with single status message
        statusDiv.innerHTML = `
            <div class="flex items-center gap-3 p-3 bg-blue-50 rounded-lg animate-fadeIn">
                <div class="animate-pulse w-2 h-2 ${iconColor} rounded-full"></div>
                <span class="text-gray-700">${message}</span>
            </div>
        `;
    }
}

function formatDate(dateString) {
    if (!dateString) return 'Unknown';
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;
    
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 30) return `${days}d ago`;
    
    return date.toLocaleDateString();
}

function getSalaryMatchClass(salaryScore, salaryMin, salaryMax) {
    // If no salary info provided, treat as poor match
    if (!salaryMin && !salaryMax) {
        return 'bg-red-50 border-red-200';
    }
    
    // Use the salary score if provided (it's already a percentage)
    if (salaryScore !== null && salaryScore !== undefined) {
        if (salaryScore >= 70) {
            // Good match
            return 'bg-green-50 border-green-200';
        } else if (salaryScore >= 40) {
            // Moderate match
            return 'bg-yellow-50 border-yellow-200';
        } else {
            // Poor match
            return 'bg-red-50 border-red-200';
        }
    }
    
    // Fallback: if no score provided, treat as unknown/poor
    return 'bg-red-50 border-red-200';
}

function getPostedDateInfo(dateString, crawledAt) {
    if (!dateString) return { text: 'Unknown Post Date', days: -1, className: 'text-gray-700', bgClass: 'bg-red-50' };
    
    const now = new Date();
    let actualDaysAgo = -1;
    let text = '';
    let className = 'text-gray-700';
    let bgClass = 'bg-red-50';
    
    // If we have a crawl timestamp, calculate the actual age
    if (crawledAt) {
        const crawlDate = new Date(crawledAt);
        // Calculate days more accurately by resetting time to midnight
        const crawlMidnight = new Date(crawlDate.getFullYear(), crawlDate.getMonth(), crawlDate.getDate());
        const nowMidnight = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const daysSinceCrawl = Math.floor((nowMidnight - crawlMidnight) / 86400000);
        
        // Parse the original posted date to get days ago at crawl time
        const lowerDateString = dateString.toLowerCase();
        
        if (lowerDateString.includes('today')) {
            // "Posted today" at crawl time
            actualDaysAgo = daysSinceCrawl;
        } else if (lowerDateString.includes('yesterday')) {
            // "Posted yesterday" at crawl time
            actualDaysAgo = 1 + daysSinceCrawl;
        } else if (lowerDateString.includes('ago')) {
            // Parse "X days/weeks/months ago"
            const match = dateString.match(/(\d+)\s*(day|week|month|hour|minute)/i);
            if (match) {
                const num = parseInt(match[1]);
                const unit = match[2].toLowerCase();
                
                let daysAtCrawl = 0;
                if (unit.includes('minute') || unit.includes('hour')) {
                    daysAtCrawl = 0;
                } else if (unit.includes('day')) {
                    daysAtCrawl = num;
                } else if (unit.includes('week')) {
                    daysAtCrawl = num * 7;
                } else if (unit.includes('month')) {
                    daysAtCrawl = num * 30;
                }
                
                // Calculate actual days ago
                actualDaysAgo = daysAtCrawl + daysSinceCrawl;
            }
        } else {
            // Try to parse as an absolute date
            const postDate = new Date(dateString);
            if (!isNaN(postDate.getTime())) {
                actualDaysAgo = Math.floor((now - postDate) / 86400000);
            }
        }
    } else {
        // No crawl date, fall back to parsing the string as-is
        const lowerDateString = dateString.toLowerCase();
        
        if (lowerDateString.includes('today')) {
            actualDaysAgo = 0;
        } else if (lowerDateString.includes('yesterday')) {
            actualDaysAgo = 1;
        } else if (dateString.includes('ago')) {
            const match = dateString.match(/(\d+)\s*(day|week|month|hour|minute)/i);
            if (match) {
                const num = parseInt(match[1]);
                const unit = match[2].toLowerCase();
                
                if (unit.includes('minute') || unit.includes('hour')) {
                    actualDaysAgo = 0;
                } else if (unit.includes('day')) {
                    actualDaysAgo = num;
                } else if (unit.includes('week')) {
                    actualDaysAgo = num * 7;
                } else if (unit.includes('month')) {
                    actualDaysAgo = num * 30;
                }
            }
        } else {
            // Try to parse as an absolute date
            const postDate = new Date(dateString);
            if (!isNaN(postDate.getTime())) {
                actualDaysAgo = Math.floor((now - postDate) / 86400000);
            }
        }
    }
    
    // Format the text based on actual days ago
    if (actualDaysAgo < 0) {
        text = 'Unknown Post Date';
        bgClass = 'bg-red-50';
    } else if (actualDaysAgo === 0) {
        text = 'Posted Today';
        className = 'text-gray-700 font-semibold';
        bgClass = 'bg-green-100';
    } else if (actualDaysAgo === 1) {
        text = 'Posted Yesterday';
        bgClass = 'bg-green-50';
    } else if (actualDaysAgo <= 7) {
        text = `Posted ${actualDaysAgo} days ago`;
        bgClass = 'bg-green-50';
    } else if (actualDaysAgo <= 14) {
        text = `Posted ${actualDaysAgo} days ago`;
        bgClass = 'bg-yellow-50';
    } else if (actualDaysAgo <= 30) {
        text = `Posted ${actualDaysAgo} days ago`;
        bgClass = 'bg-red-50';
    } else {
        const weeks = Math.floor(actualDaysAgo / 7);
        if (weeks <= 8) {
            text = `Posted ${weeks} week${weeks > 1 ? 's' : ''} ago`;
        } else {
            const months = Math.floor(actualDaysAgo / 30);
            text = `Posted ${months} month${months > 1 ? 's' : ''} ago`;
        }
        bgClass = 'bg-red-50';
    }
    
    return { text, days: actualDaysAgo, className, bgClass };
}

function formatSalary(min, max) {
    if (!min && !max) return null;
    
    const formatNum = (num) => {
        if (num >= 1000) {
            return '$' + (num / 1000).toFixed(0) + 'k';
        }
        return '$' + num;
    };
    
    if (min && max) {
        return `${formatNum(min)} - ${formatNum(max)}`;
    } else if (min) {
        return `${formatNum(min)}+`;
    } else if (max) {
        return `Up to ${formatNum(max)}`;
    }
}

// Load initial data
function loadInitialData() {
    // Only load the initially active tab
    loadTabData(activeTab);
}

// Delete functions
async function deleteResume(resumeId) {
    if (!confirm('Are you sure you want to delete this resume? This will also remove all associated matches.')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/delete_resume/${resumeId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        if (data.success) {
            showNotification('Resume deleted successfully', 'success');
            loadedTabs.matches = false; // Force refresh next time
            loadResumes();
            // Only reload matches if currently viewing matches tab
            if (activeTab === 'matches') {
                loadMatches();
            }
        } else {
            showNotification(data.error || 'Failed to delete resume', 'error');
        }
    } catch (error) {
        console.error('Error deleting resume:', error);
        showNotification('Failed to delete resume', 'error');
    }
}

async function deleteJob(jobId) {
    if (!confirm('Are you sure you want to delete this job? This will also remove all associated matches.')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/delete_job/${jobId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        if (data.success) {
            showNotification('Job deleted successfully', 'success');
            loadedTabs.search = false;  // Force refresh next time
            loadedTabs.matches = false; // Force refresh next time
            loadJobs();
            // Only reload matches if currently viewing matches tab
            if (activeTab === 'matches') {
                loadMatches();
            }
        } else {
            showNotification(data.error || 'Failed to delete job', 'error');
        }
    } catch (error) {
        console.error('Error deleting job:', error);
        showNotification('Failed to delete job', 'error');
    }
}

// Reprocess Resume Function
async function reprocessResume(resumeId, filename) {
    if (!confirm(`Reprocess ${filename}? This will re-run the extraction process.`)) {
        return;
    }
    
    try {
        // Close the modal first
        closeModal('resumeDetailModal');
        
        // Find the file in uploads directory
        const filepath = `data/uploads/${filename}`;
        const process_id = `resume_reprocess_${filename.replace('.', '_')}`;
        const command = `python -u -m graph.pipeline --mode process_resumes --resume-path "${filepath}" --no-auto-server`;
        
        showNotification('Starting resume reprocessing...', 'info');
        
        // Start the reprocessing (would need backend endpoint)
        // For now, just notify the user
        showNotification('Please re-upload the resume to reprocess it', 'info');
        
    } catch (error) {
        console.error('Error reprocessing resume:', error);
        showNotification('Failed to reprocess resume', 'error');
    }
}

// Skill Management Functions
function openSkillManager() {
    const modalContent = `
        <div class="modal show" id="skillManagerModal">
            <div class="modal-dialog" style="max-width: 700px;">
                <div class="modal-content">
                    <div class="modal-header">
                        <h3 class="modal-title">Manage Skills</h3>
                        <button class="btn-close" onclick="closeModal('skillManagerModal')">×</button>
                    </div>
                    <div class="modal-body">
                        <div class="mb-4">
                            <button onclick="addNewSkill()" class="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600">
                                + Add New Skill
                            </button>
                        </div>
                        <div id="skillsList" class="space-y-2">
                            ${window.currentResumeSkills.map((skill, index) => renderSkillRow(skill, index)).join('')}
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button onclick="saveSkills()" class="btn btn-primary">Save Changes</button>
                        <button onclick="closeModal('skillManagerModal')" class="btn btn-outline-secondary">Cancel</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Remove existing modal if present
    const existingModal = document.getElementById('skillManagerModal');
    if (existingModal) {
        existingModal.remove();
    }
    document.body.insertAdjacentHTML('beforeend', modalContent);
}

function renderSkillRow(skill, index) {
    const skillName = typeof skill === 'string' ? skill : (skill.skill_name || skill.skill || skill.name || '');
    const proficiency = typeof skill === 'object' ? (skill.required_proficiency || skill.proficiency || '') : '';
    const years = typeof skill === 'object' ? (skill.years_required || skill.years || '') : '';
    
    return `
        <div class="skill-row flex items-center gap-2 p-3 border rounded-lg bg-white" data-index="${index}">
            <input type="text" placeholder="Skill name" value="${skillName}" 
                   class="flex-1 px-2 py-1 border rounded text-sm skill-name-input">
            <select class="px-2 py-1 border rounded text-sm skill-proficiency-input">
                <option value="">Proficiency</option>
                <option value="beginner" ${proficiency === 'beginner' ? 'selected' : ''}>Beginner</option>
                <option value="intermediate" ${proficiency === 'intermediate' ? 'selected' : ''}>Intermediate</option>
                <option value="advanced" ${proficiency === 'advanced' ? 'selected' : ''}>Advanced</option>
                <option value="expert" ${proficiency === 'expert' ? 'selected' : ''}>Expert</option>
            </select>
            <input type="number" placeholder="Years" value="${years || ''}" 
                   class="w-20 px-2 py-1 border rounded text-sm skill-years-input" min="0" max="50">
            <button onclick="removeSkill(${index})" class="text-red-500 hover:text-red-700">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
            </button>
        </div>
    `;
}

function addNewSkill() {
    const skillsList = document.getElementById('skillsList');
    const newIndex = window.currentResumeSkills.length;
    window.currentResumeSkills.push({skill_name: '', proficiency: '', years: ''});
    
    const newSkillHtml = renderSkillRow({skill_name: '', proficiency: '', years: ''}, newIndex);
    skillsList.insertAdjacentHTML('beforeend', newSkillHtml);
}

function removeSkill(index) {
    window.currentResumeSkills.splice(index, 1);
    refreshSkillsList();
}

function refreshSkillsList() {
    const skillsList = document.getElementById('skillsList');
    skillsList.innerHTML = window.currentResumeSkills.map((skill, index) => renderSkillRow(skill, index)).join('');
}

async function saveSkills() {
    // Collect all skills from the form
    const skillRows = document.querySelectorAll('.skill-row');
    const updatedSkills = [];
    
    skillRows.forEach(row => {
        const name = row.querySelector('.skill-name-input').value.trim();
        if (name) {
            const proficiency = row.querySelector('.skill-proficiency-input').value;
            const years = parseInt(row.querySelector('.skill-years-input').value) || null;
            
            updatedSkills.push({
                skill_name: name,
                required_proficiency: proficiency || null,
                years_required: years
            });
        }
    });
    
    try {
        const response = await fetch(`/api/update_resume_skills/${window.currentResumeId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ skills: updatedSkills })
        });
        
        const data = await response.json();
        if (data.success) {
            showNotification('Skills updated successfully', 'success');
            window.currentResumeSkills = updatedSkills;
            
            // Update the skills display in the resume modal
            const skillsContainer = document.getElementById('resumeSkillsContainer');
            if (skillsContainer) {
                skillsContainer.innerHTML = updatedSkills.map(skill => {
                    let skillHtml = `<span class="skill-tag">`;
                    skillHtml += `<span class="font-medium">${skill.skill_name}</span>`;
                    
                    const metadata = [];
                    if (skill.required_proficiency) metadata.push(skill.required_proficiency);
                    if (skill.years_required) metadata.push(`${skill.years_required} yrs`);
                    
                    if (metadata.length > 0) {
                        skillHtml += ` <span class="text-xs text-gray-500">(${metadata.join(', ')})</span>`;
                    }
                    
                    skillHtml += `</span>`;
                    return skillHtml;
                }).join('');
            }
            
            closeModal('skillManagerModal');
            
            // Resume tab always refreshes automatically
        } else {
            showNotification(data.error || 'Failed to update skills', 'error');
        }
    } catch (error) {
        console.error('Error updating skills:', error);
        showNotification('Failed to update skills', 'error');
    }
}

// Comprehensive Resume Editor
async function openResumeEditor(resumeId) {
    try {
        const response = await fetch(`/api/get_resume/${resumeId}`);
        const data = await response.json();
        
        if (!data.success) {
            showNotification('Failed to load resume', 'error');
            return;
        }
        
        const resume = data.resume;
        
        const modalContent = `
            <div class="modal show" id="resumeEditorModal">
                <div class="modal-dialog" style="max-width: 1000px; max-height: 90vh;">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3 class="modal-title">Edit Resume Details</h3>
                            <button class="btn-close" onclick="closeModal('resumeEditorModal')">×</button>
                        </div>
                        <div class="modal-body" style="max-height: 70vh; overflow-y: auto;">
                            <!-- Contact Information -->
                            <div class="detail-section">
                                <h4 class="detail-section-title">Contact Information</h4>
                                <div class="grid md:grid-cols-2 gap-4">
                                    <div>
                                        <label class="text-sm text-gray-600">Name</label>
                                        <input type="text" id="edit-name" class="w-full px-3 py-2 border rounded-lg" 
                                               value="${resume.name || ''}" placeholder="Full Name">
                                    </div>
                                    <div>
                                        <label class="text-sm text-gray-600">Email</label>
                                        <input type="email" id="edit-email" class="w-full px-3 py-2 border rounded-lg" 
                                               value="${resume.email || ''}" placeholder="email@example.com">
                                    </div>
                                    <div>
                                        <label class="text-sm text-gray-600">Phone</label>
                                        <input type="tel" id="edit-phone" class="w-full px-3 py-2 border rounded-lg" 
                                               value="${resume.phone || ''}" placeholder="+1 234 567 8900">
                                    </div>
                                    <div>
                                        <label class="text-sm text-gray-600">Location</label>
                                        <input type="text" id="edit-location" class="w-full px-3 py-2 border rounded-lg" 
                                               value="${resume.location || ''}" placeholder="City, State">
                                    </div>
                                    <div>
                                        <label class="text-sm text-gray-600">LinkedIn</label>
                                        <input type="url" id="edit-linkedin" class="w-full px-3 py-2 border rounded-lg" 
                                               value="${resume.linkedin || ''}" placeholder="https://linkedin.com/in/...">
                                    </div>
                                    <div>
                                        <label class="text-sm text-gray-600">GitHub</label>
                                        <input type="url" id="edit-github" class="w-full px-3 py-2 border rounded-lg" 
                                               value="${resume.github || ''}" placeholder="https://github.com/...">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Professional Summary -->
                            <div class="detail-section">
                                <h4 class="detail-section-title">Professional Summary</h4>
                                <textarea id="edit-summary" class="w-full px-3 py-2 border rounded-lg" rows="4"
                                          placeholder="Brief professional summary...">${resume.summary || ''}</textarea>
                            </div>
                            
                            <!-- Work Preferences -->
                            <div class="detail-section">
                                <h4 class="detail-section-title">Work Preferences</h4>
                                <div class="grid md:grid-cols-2 gap-4">
                                    <div>
                                        <label class="text-sm text-gray-600">Years of Experience</label>
                                        <input type="number" id="edit-years" class="w-full px-3 py-2 border rounded-lg" 
                                               value="${resume.years_experience || 0}" min="0" max="50">
                                    </div>
                                    <div>
                                        <label class="text-sm text-gray-600">Salary Expectation (Min)</label>
                                        <input type="number" id="edit-salary-min" class="w-full px-3 py-2 border rounded-lg" 
                                               value="${resume.salary_expectations?.min || ''}" placeholder="50000">
                                    </div>
                                    <div>
                                        <label class="text-sm text-gray-600">Salary Expectation (Max)</label>
                                        <input type="number" id="edit-salary-max" class="w-full px-3 py-2 border rounded-lg" 
                                               value="${resume.salary_expectations?.max || ''}" placeholder="100000">
                                    </div>
                                    <div>
                                        <label class="text-sm text-gray-600">Preferred Work Arrangement</label>
                                        <select id="edit-work-arrangement" class="w-full px-3 py-2 border rounded-lg">
                                            <option value="">Select...</option>
                                            <option value="remote" ${resume.work_preferences?.arrangement === 'remote' ? 'selected' : ''}>Remote</option>
                                            <option value="hybrid" ${resume.work_preferences?.arrangement === 'hybrid' ? 'selected' : ''}>Hybrid</option>
                                            <option value="onsite" ${resume.work_preferences?.arrangement === 'onsite' ? 'selected' : ''}>Onsite</option>
                                            <option value="flexible" ${resume.work_preferences?.arrangement === 'flexible' ? 'selected' : ''}>Flexible</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Certifications -->
                            <div class="detail-section">
                                <h4 class="detail-section-title">Certifications</h4>
                                <div id="certifications-list" class="space-y-2">
                                    ${(resume.certifications || []).map((cert, idx) => `
                                        <div class="flex gap-2 cert-item">
                                            <input type="text" class="flex-1 px-3 py-2 border rounded-lg cert-input" 
                                                   value="${typeof cert === 'object' ? cert.name || '' : cert}" placeholder="Certification name">
                                            <button onclick="this.parentElement.remove()" class="px-3 py-2 text-red-600 hover:bg-red-50 rounded-lg">Remove</button>
                                        </div>
                                    `).join('')}
                                </div>
                                <button onclick="addCertification()" class="mt-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600">
                                    + Add Certification
                                </button>
                            </div>
                            
                            <!-- Achievements -->
                            <div class="detail-section">
                                <h4 class="detail-section-title">Key Achievements</h4>
                                <div id="achievements-list" class="space-y-2">
                                    ${(resume.achievements || []).map((achievement, idx) => `
                                        <div class="flex gap-2 achievement-item">
                                            <textarea class="flex-1 px-3 py-2 border rounded-lg achievement-input" rows="2"
                                                      placeholder="Describe achievement...">${achievement}</textarea>
                                            <button onclick="this.parentElement.remove()" class="px-3 py-2 text-red-600 hover:bg-red-50 rounded-lg">Remove</button>
                                        </div>
                                    `).join('')}
                                </div>
                                <button onclick="addAchievement()" class="mt-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600">
                                    + Add Achievement
                                </button>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button onclick="saveResumeEdits('${resumeId}')" class="btn btn-primary">Save Changes</button>
                            <button onclick="closeModal('resumeEditorModal')" class="btn btn-outline-secondary">Cancel</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Remove existing modal if present
        const existingModal = document.getElementById('resumeEditorModal');
        if (existingModal) {
            existingModal.remove();
        }
        
        // Close the detail modal first
        closeModal('resumeDetailModal');
        
        // Add new modal
        document.body.insertAdjacentHTML('beforeend', modalContent);
        
    } catch (error) {
        console.error('Error opening resume editor:', error);
        showNotification('Failed to open editor', 'error');
    }
}

function addCertification() {
    const list = document.getElementById('certifications-list');
    const newItem = document.createElement('div');
    newItem.className = 'flex gap-2 cert-item';
    newItem.innerHTML = `
        <input type="text" class="flex-1 px-3 py-2 border rounded-lg cert-input" placeholder="Certification name">
        <button onclick="this.parentElement.remove()" class="px-3 py-2 text-red-600 hover:bg-red-50 rounded-lg">Remove</button>
    `;
    list.appendChild(newItem);
}

function addAchievement() {
    const list = document.getElementById('achievements-list');
    const newItem = document.createElement('div');
    newItem.className = 'flex gap-2 achievement-item';
    newItem.innerHTML = `
        <textarea class="flex-1 px-3 py-2 border rounded-lg achievement-input" rows="2" placeholder="Describe achievement..."></textarea>
        <button onclick="this.parentElement.remove()" class="px-3 py-2 text-red-600 hover:bg-red-50 rounded-lg">Remove</button>
    `;
    list.appendChild(newItem);
}

async function saveResumeEdits(resumeId) {
    try {
        // Gather all the edited data
        const updatedData = {
            name: document.getElementById('edit-name').value,
            email: document.getElementById('edit-email').value,
            phone: document.getElementById('edit-phone').value,
            location: document.getElementById('edit-location').value,
            linkedin: document.getElementById('edit-linkedin').value,
            github: document.getElementById('edit-github').value,
            summary: document.getElementById('edit-summary').value,
            years_experience: parseInt(document.getElementById('edit-years').value) || 0,
            salary_expectations: {
                min: parseInt(document.getElementById('edit-salary-min').value) || null,
                max: parseInt(document.getElementById('edit-salary-max').value) || null
            },
            work_preferences: {
                arrangement: document.getElementById('edit-work-arrangement').value
            },
            certifications: Array.from(document.querySelectorAll('.cert-input')).map(input => input.value).filter(v => v),
            achievements: Array.from(document.querySelectorAll('.achievement-input')).map(input => input.value).filter(v => v)
        };
        
        // Send update to backend
        const response = await fetch(`/api/update_resume/${resumeId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(updatedData)
        });
        
        const data = await response.json();
        if (data.success) {
            showNotification('Resume updated successfully', 'success');
            closeModal('resumeEditorModal');
            
            // Refresh the resume display
            showResumeDetail(resumeId);
            
            // Resume tab always refreshes automatically
        } else {
            showNotification(data.error || 'Failed to update resume', 'error');
        }
    } catch (error) {
        console.error('Error saving resume edits:', error);
        showNotification('Failed to save changes', 'error');
    }
}

// WebSocket Connection Management Functions
function startHeartbeat() {
    stopHeartbeat(); // Clear any existing interval
    pingInterval = setInterval(() => {
        if (isConnected) {
            socket.emit('ping');
            // Don't log every ping - too verbose
            
            // Check if we haven't received a pong in too long
            if (Date.now() - lastPongTime > PONG_TIMEOUT) {
                console.warn('Connection timeout - attempting to reconnect');
                // Force reconnection
                socket.disconnect();
                setTimeout(() => socket.connect(), 1000);
            }
        }
    }, PING_INTERVAL);
}

function stopHeartbeat() {
    if (pingInterval) {
        clearInterval(pingInterval);
        pingInterval = null;
    }
}

function addConnectionStatusIndicator() {
    // Add a connection status indicator to the page
    const indicator = document.createElement('div');
    indicator.id = 'connection-status';
    indicator.className = 'fixed bottom-4 right-4 px-4 py-2 rounded-lg shadow-lg hidden z-50';
    indicator.innerHTML = `
        <div class="flex items-center gap-2">
            <div class="w-2 h-2 rounded-full bg-current animate-pulse"></div>
            <span id="connection-status-text">Connecting...</span>
        </div>
    `;
    document.body.appendChild(indicator);
}

function showConnectionError(message) {
    const indicator = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-status-text');
    
    if (indicator && statusText) {
        indicator.className = 'fixed bottom-4 right-4 px-4 py-2 rounded-lg shadow-lg bg-red-100 text-red-700 z-50';
        statusText.textContent = message;
        indicator.style.display = 'block';
    }
}

function clearConnectionError() {
    const indicator = document.getElementById('connection-status');
    if (indicator) {
        indicator.style.display = 'none';
    }
}

// Enhanced process output handling with fallback polling
function appendProcessOutput(processId, output) {
    console.log(`[${processId}] Output: ${output}`);
    
    // Store output for later display
    if (!window.processOutputs) {
        window.processOutputs = {};
    }
    if (!window.processOutputs[processId]) {
        window.processOutputs[processId] = [];
    }
    window.processOutputs[processId].push({
        text: output,
        timestamp: new Date().toISOString()
    });
    
    // Update UI based on process type
    if (processId.startsWith('resume_')) {
        // Parse for specific resume processing stages
        const lowerOutput = output.toLowerCase();
        if (lowerOutput.includes('complete') || lowerOutput.includes('success')) {
            updateUploadStatus('Resume processing completed!');
            setTimeout(() => loadResumes(), 1000); // Refresh resume list
        } else if (lowerOutput.includes('error') || lowerOutput.includes('fail')) {
            updateUploadStatus('Error processing resume');
        }
    } else if (processId.startsWith('search_')) {
        // Parse for job search stages
        const lowerOutput = output.toLowerCase();
        if (lowerOutput.includes('complete') || lowerOutput.includes('finished')) {
            updateSearchStatus('Job search completed!');
            setTimeout(() => loadJobs(), 1000); // Refresh job list
        } else if (lowerOutput.includes('error') || lowerOutput.includes('fail')) {
            updateSearchStatus('Error during job search');
        }
    } else if (processId.startsWith('match_')) {
        // Parse for matching stages
        const lowerOutput = output.toLowerCase();
        if (lowerOutput.includes('complete') || lowerOutput.includes('finished')) {
            updateMatchingStatus('Matching completed!');
            setTimeout(() => loadMatches(), 1000); // Refresh matches
        } else if (lowerOutput.includes('error') || lowerOutput.includes('fail')) {
            updateMatchingStatus('Error during matching');
        }
    }
}

// ===== MATCH FILTERING FUNCTIONS =====

async function loadResumeFilters() {
    // Only load if the container exists
    const container = document.getElementById('resumeFilters');
    if (!container) return;
    
    try {
        const response = await fetch('/api/get_resumes');
        const data = await response.json();
        
        if (data.success && data.resumes) {
            
            container.innerHTML = data.resumes.map(resume => `
                <label class="inline-flex items-center">
                    <input type="checkbox" 
                           class="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                           id="resume-filter-${resume.id}"
                           value="${resume.id}"
                           onchange="toggleResumeFilter('${resume.id}')"
                           ${!matchFilters.hiddenResumes.includes(resume.id) ? 'checked' : ''}>
                    <span class="ml-2 text-sm text-gray-700">
                        ${resume.filename || resume.name || 'Unknown'}
                    </span>
                </label>
            `).join('');
        }
    } catch (error) {
        console.error('Error loading resume filters:', error);
    }
}

function toggleResumeFilter(resumeId) {
    const checkbox = document.getElementById(`resume-filter-${resumeId}`);
    if (checkbox.checked) {
        // Remove from hidden list
        matchFilters.hiddenResumes = matchFilters.hiddenResumes.filter(id => id !== resumeId);
    } else {
        // Add to hidden list
        if (!matchFilters.hiddenResumes.includes(resumeId)) {
            matchFilters.hiddenResumes.push(resumeId);
        }
    }
    
    // Save filters to localStorage
    saveMatchFiltersToStorage();
    
    // Reload matches with new filters
    loadMatches();
}

function updateMinScore(value) {
    matchFilters.minScore = parseInt(value);
    const scoreValueEl = document.getElementById('minScoreValue');
    if (scoreValueEl) {
        scoreValueEl.textContent = `${value}%`;
    }
    
    // Save filters to localStorage
    saveMatchFiltersToStorage();
    
    // Debounce the reload
    clearTimeout(window.scoreFilterTimeout);
    window.scoreFilterTimeout = setTimeout(() => {
        loadMatches();
    }, 300);
}

function resetFilters() {
    // Reset filter state
    matchFilters = {
        minScore: 30,
        hiddenResumes: []
    };
    
    // Clear from localStorage
    localStorage.removeItem('matchFilters');
    
    // Reset UI elements if they exist
    const slider = document.getElementById('minScoreSlider');
    if (slider) {
        slider.value = 30;
    }
    
    const scoreValue = document.getElementById('minScoreValue');
    if (scoreValue) {
        scoreValue.textContent = '30%';
    }
    
    // Check all resume checkboxes
    const checkboxes = document.querySelectorAll('#resumeFilters input[type="checkbox"]');
    checkboxes.forEach(cb => cb.checked = true);
    
    // Reload matches
    loadMatches();
}

// Initialize slider event listener when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    const slider = document.getElementById('minScoreSlider');
    if (slider) {
        slider.addEventListener('input', function() {
            updateMinScore(this.value);
        });
    }
});

// ===== DATA MANAGEMENT FUNCTIONS =====

function showClearDataModal() {
    const modal = document.getElementById('clearDataModal');
    if (modal) {
        modal.classList.remove('hidden');
    }
}

function closeClearDataModal() {
    const modal = document.getElementById('clearDataModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

async function confirmClearData() {
    // Get selected options
    const clearJobs = document.getElementById('clearJobs').checked;
    const clearMatches = document.getElementById('clearMatches').checked;
    const clearResumes = document.getElementById('clearResumes').checked;
    
    if (!clearJobs && !clearMatches && !clearResumes) {
        alert('Please select at least one type of data to clear.');
        return;
    }
    
    // Final confirmation
    let confirmMsg = 'Are you sure you want to permanently delete:\n';
    if (clearJobs) confirmMsg += '- All jobs\n';
    if (clearMatches) confirmMsg += '- All matches\n';
    if (clearResumes) confirmMsg += '- All resumes (INCLUDING UPLOADED FILES)\n';
    confirmMsg += '\nThis action cannot be undone!';
    
    if (!confirm(confirmMsg)) {
        return;
    }
    
    try {
        // Show loading state
        const modal = document.getElementById('clearDataModal');
        const originalContent = modal.querySelector('.bg-white').innerHTML;
        modal.querySelector('.bg-white').innerHTML = `
            <div class="text-center py-8">
                <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-red-600"></div>
                <p class="text-sm text-gray-600 mt-2">Clearing data...</p>
            </div>
        `;
        
        // Make API call
        const response = await fetch('/api/clear_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                clear_jobs: clearJobs,
                clear_matches: clearMatches,
                clear_resumes: clearResumes
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Show success message
            modal.querySelector('.bg-white').innerHTML = `
                <div class="text-center py-8">
                    <div class="text-green-600 mb-4">
                        <svg class="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                    <h3 class="text-lg font-semibold text-gray-900 mb-2">Data Cleared Successfully</h3>
                    <p class="text-sm text-gray-600">${data.message}</p>
                    <button onclick="closeClearDataModal(); location.reload();" 
                            class="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                        Refresh Page
                    </button>
                </div>
            `;
            
            // Refresh the current tab's data
            setTimeout(() => {
                if (clearJobs && activeTab === 'search') {
                    loadJobs();
                }
                if (clearMatches && activeTab === 'matches') {
                    loadMatches();
                }
                if (clearResumes && activeTab === 'upload') {
                    loadResumes();
                }
                if (activeTab === 'analytics') {
                    loadAnalytics();
                }
            }, 1000);
            
        } else {
            // Show error
            modal.querySelector('.bg-white').innerHTML = `
                <div class="text-center py-8">
                    <div class="text-red-600 mb-4">
                        <svg class="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                    <h3 class="text-lg font-semibold text-gray-900 mb-2">Error Clearing Data</h3>
                    <p class="text-sm text-gray-600">${data.error || 'An error occurred'}</p>
                    <button onclick="closeClearDataModal(); location.reload();" 
                            class="mt-4 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700">
                        Close
                    </button>
                </div>
            `;
        }
        
    } catch (error) {
        console.error('Error clearing data:', error);
        alert('An error occurred while clearing data. Please check the console for details.');
        closeClearDataModal();
    }
}
