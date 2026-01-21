/**
 * Background Service Worker - Complexity Injector
 *
 * Handles:
 * - Context menu creation and events
 * - Communication with offscreen document (which hosts the AI worker)
 * - Extension state management
 *
 * Note: Service Workers cannot spawn Web Workers directly, so we use
 * the Offscreen API to create a hidden document that can host the worker.
 */

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

const ExtensionState = {
  workerReady: false,
  workerLoading: false,
  pendingRequests: new Map(),
  requestId: 0,
  offscreenCreated: false,
};

// ============================================================================
// OFFSCREEN DOCUMENT MANAGEMENT
// ============================================================================

const OFFSCREEN_DOCUMENT_PATH = 'offscreen.html';

/**
 * Check if an offscreen document already exists
 */
async function hasOffscreenDocument() {
  // Check if the offscreen document is already open
  // Use getContexts() for Chrome 116+ or fallback for older versions
  if ('getContexts' in chrome.runtime) {
    const contexts = await chrome.runtime.getContexts({
      contextTypes: ['OFFSCREEN_DOCUMENT'],
      documentUrls: [chrome.runtime.getURL(OFFSCREEN_DOCUMENT_PATH)],
    });
    return contexts.length > 0;
  }

  // Fallback for older Chrome versions - try to send a ping
  try {
    const response = await chrome.runtime.sendMessage({
      target: 'offscreen',
      type: 'ping',
    });
    return response?.alive === true;
  } catch {
    return false;
  }
}

/**
 * Setup the offscreen document if it doesn't exist
 */
async function setupOffscreenDocument() {
  if (await hasOffscreenDocument()) {
    console.log('[Background] Offscreen document already exists');
    ExtensionState.offscreenCreated = true;
    return;
  }

  console.log('[Background] Creating offscreen document...');

  try {
    await chrome.offscreen.createDocument({
      url: OFFSCREEN_DOCUMENT_PATH,
      reasons: [chrome.offscreen.Reason.WORKERS],
      justification: 'Run AI model in Web Worker for text complexification',
    });

    ExtensionState.offscreenCreated = true;
    console.log('[Background] Offscreen document created successfully');
  } catch (error) {
    if (error.message?.includes('Only a single offscreen')) {
      // Document already exists (race condition)
      console.log('[Background] Offscreen document already exists (race condition)');
      ExtensionState.offscreenCreated = true;
    } else {
      console.error('[Background] Failed to create offscreen document:', error);
      throw error;
    }
  }
}

/**
 * Close the offscreen document (if needed for cleanup)
 */
async function closeOffscreenDocument() {
  if (!(await hasOffscreenDocument())) {
    return;
  }

  try {
    await chrome.offscreen.closeDocument();
    ExtensionState.offscreenCreated = false;
    console.log('[Background] Offscreen document closed');
  } catch (error) {
    console.error('[Background] Failed to close offscreen document:', error);
  }
}

// ============================================================================
// WORKER COMMUNICATION VIA OFFSCREEN
// ============================================================================

/**
 * Send message to worker via offscreen document and wait for response
 */
function sendToWorker(type, data = {}) {
  return new Promise(async (resolve, reject) => {
    // Ensure offscreen document exists
    try {
      await setupOffscreenDocument();
    } catch (error) {
      reject(new Error('Failed to setup offscreen document: ' + error.message));
      return;
    }

    const id = ++ExtensionState.requestId;
    ExtensionState.pendingRequests.set(id, { resolve, reject });

    console.log('[Background] Sending to worker via offscreen:', type, id);

    try {
      // Send message to offscreen document
      const response = await chrome.runtime.sendMessage({
        target: 'offscreen',
        type: 'workerMessage',
        workerType: type,
        id,
        data,
      });

      if (response?.error) {
        ExtensionState.pendingRequests.delete(id);
        reject(new Error(response.error));
        return;
      }

      // Response acknowledged - actual result will come via workerResponse message
    } catch (error) {
      ExtensionState.pendingRequests.delete(id);
      reject(new Error('Failed to send message to offscreen: ' + error.message));
    }

    // Timeout after 120 seconds (model loading can take time)
    setTimeout(() => {
      if (ExtensionState.pendingRequests.has(id)) {
        ExtensionState.pendingRequests.delete(id);
        reject(new Error('Request timeout'));
      }
    }, 120000);
  });
}

/**
 * Handle messages from offscreen document (worker responses)
 */
function handleOffscreenMessage(message) {
  console.log('[Background] Received from offscreen:', message.type);

  switch (message.type) {
    case 'workerStatus':
      ExtensionState.workerLoading = message.status === 'loading' || message.status === 'caching';
      ExtensionState.workerReady = message.status === 'ready';

      // Broadcast status to popup and other listeners
      chrome.runtime.sendMessage({
        type: 'workerStatus',
        status: message.status,
        message: message.message,
      }).catch(() => {}); // Ignore errors if no listeners
      break;

    case 'workerResponse':
      const { id, originalType, result, error, message: errorMessage } = message;
      console.log('[Background] workerResponse id:', id, 'result:', JSON.stringify(result).substring(0, 500));

      if (ExtensionState.pendingRequests.has(id)) {
        const { resolve, reject } = ExtensionState.pendingRequests.get(id);
        ExtensionState.pendingRequests.delete(id);

        if (originalType === 'error') {
          reject(new Error(error || errorMessage || 'Worker error'));
        } else {
          resolve(result);
        }
      }
      break;

    case 'workerError':
      console.error('[Background] Worker error:', message.error);
      ExtensionState.workerReady = false;
      ExtensionState.workerLoading = false;

      // Reject all pending requests
      for (const [id, { reject }] of ExtensionState.pendingRequests) {
        reject(new Error(message.error || 'Worker error'));
      }
      ExtensionState.pendingRequests.clear();
      break;
  }
}

/**
 * Initialize the worker (via offscreen document)
 */
async function initializeWorker() {
  if (ExtensionState.workerReady || ExtensionState.workerLoading) {
    return;
  }

  ExtensionState.workerLoading = true;

  try {
    // Ensure offscreen document exists
    await setupOffscreenDocument();

    // Send init message to worker
    await sendToWorker('init');
    ExtensionState.workerReady = true;
  } catch (error) {
    console.error('[Background] Failed to initialize worker:', error);
    ExtensionState.workerLoading = false;
    throw error;
  }
}

// ============================================================================
// CONTEXT MENU
// ============================================================================

/**
 * Create context menu on install
 */
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'complexify-selection',
    title: 'Complexify Selection',
    contexts: ['selection'],
  });
});

/**
 * Handle context menu click
 */
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === 'complexify-selection' && info.selectionText) {
    // Ensure content script is injected first
    await ensureContentScript(tab.id);

    // Ensure worker is initialized
    if (!ExtensionState.workerReady) {
      // Notify content script about loading state
      chrome.tabs.sendMessage(tab.id, {
        type: 'showLoading',
        message: 'Model is loading, please wait...',
      }).catch(() => {});

      try {
        await initializeWorker();
      } catch (error) {
        chrome.tabs.sendMessage(tab.id, {
          type: 'showError',
          message: 'Failed to load AI model: ' + error.message,
        }).catch(() => {});
        return;
      }
    }

    if (!ExtensionState.workerReady) {
      chrome.tabs.sendMessage(tab.id, {
        type: 'showLoading',
        message: 'Model is still loading, please wait...',
      }).catch(() => {});
      return;
    }

    // Show loading state
    chrome.tabs.sendMessage(tab.id, {
      type: 'showLoading',
      message: 'Processing selection...',
    }).catch(() => {});

    try {
      // Process the selected text
      const result = await sendToWorker('processText', {
        text: info.selectionText,
        maxDensity: 0.2, // Higher density for selections
      });

      console.log('[Background] processText result:', result.substitutionsMade, 'substitutions');

      // Send result to content script
      await chrome.tabs.sendMessage(tab.id, {
        type: 'replaceSelection',
        result,
      });
    } catch (error) {
      console.error('[Background] Error processing selection:', error);
      chrome.tabs.sendMessage(tab.id, {
        type: 'showError',
        message: error.message,
      }).catch(() => {});
    }
  }
});

// ============================================================================
// MESSAGE HANDLING
// ============================================================================

/**
 * Handle messages from content script, popup, and offscreen document
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Handle messages from offscreen document
  if (message.target === 'background') {
    handleOffscreenMessage(message);
    return false; // No response needed
  }

  // Handle messages from content script and popup
  handleMessage(message, sender)
    .then(sendResponse)
    .catch(error => sendResponse({ error: error.message }));

  return true; // Keep message channel open for async response
});

async function handleMessage(message, sender) {
  switch (message.type) {
    case 'getWorkerStatus':
      return {
        ready: ExtensionState.workerReady,
        loading: ExtensionState.workerLoading,
      };

    case 'initWorker':
      await initializeWorker();
      return {
        ready: ExtensionState.workerReady,
      };

    case 'processText':
      if (!ExtensionState.workerReady) {
        await initializeWorker();
      }
      return await sendToWorker('processText', message.data);

    case 'processSubstitution':
      if (!ExtensionState.workerReady) {
        await initializeWorker();
      }
      return await sendToWorker('processSubstitution', message.data);

    case 'findBestSubstitution':
      if (!ExtensionState.workerReady) {
        await initializeWorker();
      }
      return await sendToWorker('findBestSubstitution', message.data);

    case 'findVocabularyWords':
      if (!ExtensionState.workerReady) {
        await initializeWorker();
      }
      return await sendToWorker('findVocabularyWords', message.data);

    case 'addCustomVocabulary':
      if (!ExtensionState.workerReady) {
        await initializeWorker();
      }
      return await sendToWorker('addCustomVocabulary', message.data);

    case 'clearCustomVocabulary':
      if (!ExtensionState.workerReady) {
        await initializeWorker();
      }
      return await sendToWorker('clearCustomVocabulary');

    case 'getVocabulary':
      if (!ExtensionState.workerReady) {
        await initializeWorker();
      }
      return await sendToWorker('getVocabulary');

    case 'processWholePage':
      // Get tabId from message.data (popup) or sender.tab (content script)
      const tabId = message.data?.tabId || sender.tab?.id;
      if (!tabId) {
        throw new Error('No tab ID provided');
      }
      return await processWholePage(tabId, message.data);

    default:
      throw new Error(`Unknown message type: ${message.type}`);
  }
}

/**
 * Ensure content script is injected in the tab
 */
async function ensureContentScript(tabId) {
  try {
    // Try to ping the content script
    await chrome.tabs.sendMessage(tabId, { type: 'ping' });
  } catch (error) {
    // Content script not loaded, inject it
    console.log('[Background] Injecting content script into tab', tabId);
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ['content_script.js'],
    });
    await chrome.scripting.insertCSS({
      target: { tabId },
      files: ['styles.css'],
    });
  }
}

/**
 * Process entire visible page
 */
async function processWholePage(tabId, options = {}) {
  // Ensure worker is ready
  if (!ExtensionState.workerReady) {
    await initializeWorker();
  }

  if (!ExtensionState.workerReady) {
    throw new Error('Model not loaded');
  }

  // Ensure content script is loaded
  await ensureContentScript(tabId);

  // Get page text
  const [{ result: pageData }] = await chrome.scripting.executeScript({
    target: { tabId },
    func: () => {
      // Get visible text from page
      const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        {
          acceptNode: (node) => {
            // Skip hidden elements
            const style = window.getComputedStyle(node.parentElement);
            if (style.display === 'none' || style.visibility === 'hidden') {
              return NodeFilter.FILTER_REJECT;
            }
            // Skip script and style content
            if (['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(node.parentElement.tagName)) {
              return NodeFilter.FILTER_REJECT;
            }
            // Only accept non-empty text
            if (node.textContent.trim().length > 0) {
              return NodeFilter.FILTER_ACCEPT;
            }
            return NodeFilter.FILTER_REJECT;
          }
        }
      );

      const textNodes = [];
      let wordCount = 0;
      const maxWords = 1000;

      while (walker.nextNode() && wordCount < maxWords) {
        const text = walker.currentNode.textContent.trim();
        const words = text.split(/\s+/).length;
        textNodes.push({
          text,
          wordCount: words,
        });
        wordCount += words;
      }

      return {
        textNodes,
        totalWords: wordCount,
      };
    },
  });

  // Combine text for processing
  const fullText = pageData.textNodes.map(n => n.text).join(' ');

  // Process through AI worker
  const result = await sendToWorker('processText', {
    text: fullText,
    maxDensity: options.maxDensity || 0.08,
  });

  // Send replacements to content script
  // Use catch to handle pages where content script can't run (chrome://, etc.)
  try {
    await chrome.tabs.sendMessage(tabId, {
      type: 'applyReplacements',
      substitutions: result.substitutions,
    });
  } catch (error) {
    console.warn('[Background] Could not send to content script:', error.message);
    // Still return the result - the processing worked, just couldn't apply to page
  }

  return result;
}

// ============================================================================
// STORAGE MANAGEMENT
// ============================================================================

/**
 * Save custom vocabulary to storage
 */
async function saveCustomVocabulary(vocabulary) {
  await chrome.storage.local.set({ customVocabulary: vocabulary });
}

/**
 * Load custom vocabulary from storage
 */
async function loadCustomVocabulary() {
  const { customVocabulary } = await chrome.storage.local.get('customVocabulary');
  return customVocabulary || [];
}

/**
 * Initialize custom vocabulary on startup
 */
chrome.runtime.onStartup.addListener(async () => {
  const vocabulary = await loadCustomVocabulary();
  if (vocabulary.length > 0) {
    try {
      await initializeWorker();
      await sendToWorker('addCustomVocabulary', { vocabulary });
    } catch (error) {
      console.error('[Background] Failed to load custom vocabulary on startup:', error);
    }
  }
});

// ============================================================================
// INITIALIZATION
// ============================================================================

// Setup offscreen document when the service worker starts
// (but don't block on worker initialization)
setupOffscreenDocument()
  .then(() => {
    console.log('[Background] Offscreen document setup complete');
    // Optionally start worker initialization in background
    // initializeWorker().catch(console.error);
  })
  .catch(error => {
    console.error('[Background] Failed to setup offscreen document:', error);
  });
