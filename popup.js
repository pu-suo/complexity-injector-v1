/**
 * Popup Script - Complexity Injector
 *
 * Handles:
 * - UI interactions
 * - Communication with background script
 * - CSV upload and parsing
 */

// ============================================================================
// DOM ELEMENTS
// ============================================================================

const elements = {
  statusIndicator: document.getElementById('status-indicator'),
  statusText: document.getElementById('status-text'),
  processPageBtn: document.getElementById('process-page-btn'),
  densitySlider: document.getElementById('density-slider'),
  densityValue: document.getElementById('density-value'),
  uploadBtn: document.getElementById('upload-btn'),
  csvInput: document.getElementById('csv-input'),
  fileName: document.getElementById('file-name'),
  defaultVocabCount: document.getElementById('default-vocab-count'),
  customVocabCount: document.getElementById('custom-vocab-count'),
  clearCustomBtn: document.getElementById('clear-custom-btn'),
  instructionsToggle: document.getElementById('instructions-toggle'),
  instructionsContent: document.getElementById('instructions-content'),
  loadingOverlay: document.getElementById('loading-overlay'),
  loadingMessage: document.getElementById('loading-message'),
};

// ============================================================================
// STATE
// ============================================================================

const State = {
  workerReady: false,
  workerLoading: false,
  defaultVocabCount: 0,
  customVocabCount: 0,
};

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize the popup
 */
async function initialize() {
  // Set up event listeners
  setupEventListeners();

  // Check worker status
  await checkWorkerStatus();

  // Load vocabulary stats
  await loadVocabularyStats();
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
  // Process page button
  elements.processPageBtn.addEventListener('click', processCurrentPage);

  // Density slider
  elements.densitySlider.addEventListener('input', (e) => {
    elements.densityValue.textContent = `${e.target.value}%`;
  });

  // Upload button
  elements.uploadBtn.addEventListener('click', () => {
    elements.csvInput.click();
  });

  // CSV file input
  elements.csvInput.addEventListener('change', handleCsvUpload);

  // Clear custom vocabulary
  elements.clearCustomBtn.addEventListener('click', clearCustomVocabulary);

  // Instructions toggle
  elements.instructionsToggle.addEventListener('click', toggleInstructions);

  // Listen for worker status updates
  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === 'workerStatus') {
      updateStatus(message.status, message.message);
    }
  });
}

/**
 * Check worker status
 */
async function checkWorkerStatus() {
  try {
    const response = await chrome.runtime.sendMessage({ type: 'getWorkerStatus' });

    if (response.ready) {
      updateStatus('ready', 'Model loaded');
    } else if (response.loading) {
      updateStatus('loading', 'Loading model...');
      // Initialize the worker
      showLoading('Loading AI model...');
      await chrome.runtime.sendMessage({ type: 'initWorker' });
      hideLoading();
    } else {
      updateStatus('idle', 'Click to initialize');
      // Start initialization
      initializeWorker();
    }
  } catch (error) {
    console.error('Error checking worker status:', error);
    updateStatus('error', 'Error');
  }
}

/**
 * Initialize the worker
 */
async function initializeWorker() {
  updateStatus('loading', 'Initializing...');
  showLoading('Loading AI model (this may take a moment)...');

  try {
    const response = await chrome.runtime.sendMessage({ type: 'initWorker' });

    if (response.ready) {
      updateStatus('ready', 'Model loaded');
    } else {
      updateStatus('error', 'Failed to load');
    }
  } catch (error) {
    console.error('Error initializing worker:', error);
    updateStatus('error', 'Error');
  } finally {
    hideLoading();
  }
}

// ============================================================================
// STATUS MANAGEMENT
// ============================================================================

/**
 * Update status indicator
 */
function updateStatus(status, message) {
  elements.statusIndicator.className = `status-indicator status-${status}`;
  elements.statusText.textContent = message;

  State.workerReady = status === 'ready';
  State.workerLoading = status === 'loading' || status === 'caching';

  // Enable/disable process button
  elements.processPageBtn.disabled = !State.workerReady;
}

/**
 * Show loading overlay
 */
function showLoading(message) {
  elements.loadingMessage.textContent = message;
  elements.loadingOverlay.classList.remove('hidden');
}

/**
 * Hide loading overlay
 */
function hideLoading() {
  elements.loadingOverlay.classList.add('hidden');
}

// ============================================================================
// PAGE PROCESSING
// ============================================================================

/**
 * Process the current page
 */
async function processCurrentPage() {
  if (!State.workerReady) {
    return;
  }

  showLoading('Processing page...');
  elements.processPageBtn.disabled = true;

  try {
    const density = parseInt(elements.densitySlider.value) / 100;

    // Get the active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab) {
      throw new Error('No active tab found');
    }

    // Send message to background to process the page
    const result = await chrome.runtime.sendMessage({
      type: 'processWholePage',
      data: { maxDensity: density, tabId: tab.id },
    });

    if (result.error) {
      throw new Error(result.error);
    }

    // Show success notification
    showNotification(`Made ${result.substitutionsMade} substitution(s)`, 'success');

  } catch (error) {
    console.error('Error processing page:', error);
    showNotification(error.message || 'Error processing page', 'error');
  } finally {
    hideLoading();
    elements.processPageBtn.disabled = false;
  }
}

// ============================================================================
// VOCABULARY MANAGEMENT
// ============================================================================

/**
 * Load vocabulary statistics
 */
async function loadVocabularyStats() {
  try {
    const response = await chrome.runtime.sendMessage({ type: 'getVocabulary' });

    if (response) {
      State.defaultVocabCount = response.default?.length || 0;
      State.customVocabCount = response.custom?.length || 0;

      elements.defaultVocabCount.textContent = State.defaultVocabCount;
      elements.customVocabCount.textContent = State.customVocabCount;

      elements.clearCustomBtn.disabled = State.customVocabCount === 0;
    }
  } catch (error) {
    console.error('Error loading vocabulary stats:', error);
  }
}

/**
 * Handle CSV file upload
 */
async function handleCsvUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  elements.fileName.textContent = file.name;

  try {
    showLoading('Parsing CSV...');

    const text = await file.text();
    const vocabulary = parseCSV(text);

    if (vocabulary.length === 0) {
      throw new Error('No valid entries found in CSV');
    }

    showLoading(`Processing ${vocabulary.length} words...`);

    // Send to background
    const response = await chrome.runtime.sendMessage({
      type: 'addCustomVocabulary',
      data: { vocabulary },
    });

    if (response.error) {
      throw new Error(response.error);
    }

    // Save to storage
    const existing = await chrome.storage.local.get('customVocabulary');
    const combined = [...(existing.customVocabulary || []), ...vocabulary];
    await chrome.storage.local.set({ customVocabulary: combined });

    // Update stats
    await loadVocabularyStats();

    showNotification(`Added ${vocabulary.length} custom word(s)`, 'success');

  } catch (error) {
    console.error('Error uploading CSV:', error);
    showNotification(error.message || 'Error parsing CSV', 'error');
  } finally {
    hideLoading();
    // Reset file input
    elements.csvInput.value = '';
  }
}

/**
 * Parse CSV content
 * Expected columns: Word, Synonym (optional: Definition, Examples)
 */
function parseCSV(text) {
  const lines = text.trim().split('\n');
  if (lines.length < 2) {
    return [];
  }

  // Parse header
  const header = lines[0].toLowerCase().split(',').map(h => h.trim());
  const wordIndex = header.findIndex(h => h === 'word' || h === 'simple');
  const synonymIndex = header.findIndex(h => h === 'synonym' || h === 'complex' || h === 'replacement');
  const defIndex = header.findIndex(h => h === 'definition' || h === 'def');

  if (wordIndex === -1 || synonymIndex === -1) {
    throw new Error('CSV must have "Word" and "Synonym" columns');
  }

  const vocabulary = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    // Handle quoted values
    const values = parseCSVLine(line);

    const word = values[wordIndex]?.trim();
    const synonym = values[synonymIndex]?.trim();
    const definition = defIndex !== -1 ? values[defIndex]?.trim() : '';

    if (word && synonym) {
      vocabulary.push({
        word: word.toLowerCase(),
        synonym,
        definition,
        examples: [],
      });
    }
  }

  return vocabulary;
}

/**
 * Parse a single CSV line (handling quoted values)
 */
function parseCSVLine(line) {
  const values = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      values.push(current);
      current = '';
    } else {
      current += char;
    }
  }

  values.push(current);
  return values;
}

/**
 * Clear custom vocabulary
 */
async function clearCustomVocabulary() {
  if (!confirm('Are you sure you want to clear all custom vocabulary?')) {
    return;
  }

  try {
    showLoading('Clearing vocabulary...');

    await chrome.runtime.sendMessage({ type: 'clearCustomVocabulary' });
    await chrome.storage.local.remove('customVocabulary');

    State.customVocabCount = 0;
    elements.customVocabCount.textContent = '0';
    elements.clearCustomBtn.disabled = true;
    elements.fileName.textContent = 'No file selected';

    showNotification('Custom vocabulary cleared', 'success');

  } catch (error) {
    console.error('Error clearing vocabulary:', error);
    showNotification('Error clearing vocabulary', 'error');
  } finally {
    hideLoading();
  }
}

// ============================================================================
// UI HELPERS
// ============================================================================

/**
 * Toggle instructions section
 */
function toggleInstructions() {
  const section = elements.instructionsToggle.closest('.section');
  section.classList.toggle('section-collapsed');
}

/**
 * Show temporary notification
 */
function showNotification(message, type = 'info') {
  // Create notification element
  const notification = document.createElement('div');
  notification.className = `popup-notification popup-notification--${type}`;
  notification.textContent = message;

  document.body.appendChild(notification);

  // Animate in
  requestAnimationFrame(() => {
    notification.classList.add('popup-notification--visible');
  });

  // Remove after delay
  setTimeout(() => {
    notification.classList.remove('popup-notification--visible');
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', initialize);
