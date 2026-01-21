/**
 * Content Script - Complexity Injector
 *
 * Handles:
 * - DOM manipulation (highlighting/replacing text)
 * - User interaction (hover tooltips)
 * - Selection replacement
 */

(() => {
  // Prevent double initialization
  if (window.complexityInjectorInitialized) return;
  window.complexityInjectorInitialized = true;

  // ============================================================================
  // CONSTANTS
  // ============================================================================

  const COMPLEXIFIED_CLASS = 'complexity-injector-word';
  const TOOLTIP_CLASS = 'complexity-injector-tooltip';
  const LOADING_CLASS = 'complexity-injector-loading';
  const NOTIFICATION_CLASS = 'complexity-injector-notification';

  // ============================================================================
  // STATE
  // ============================================================================

  const State = {
    activeTooltip: null,
    replacements: new Map(), // Map of replacement elements to original words
  };

  // ============================================================================
  // TOOLTIP MANAGEMENT
  // ============================================================================

  /**
   * Create and show tooltip
   */
  function showTooltip(element, originalWord) {
    hideTooltip();

    const tooltip = document.createElement('div');
    tooltip.className = TOOLTIP_CLASS;
    tooltip.innerHTML = `
      <div class="tooltip-header">Original Word</div>
      <div class="tooltip-content">${escapeHtml(originalWord)}</div>
      <div class="tooltip-hint">Click to revert</div>
    `;

    // Position tooltip above the element
    const rect = element.getBoundingClientRect();
    tooltip.style.left = `${rect.left + window.scrollX}px`;
    tooltip.style.top = `${rect.top + window.scrollY - 10}px`;

    document.body.appendChild(tooltip);

    // Adjust position after adding to DOM
    const tooltipRect = tooltip.getBoundingClientRect();
    tooltip.style.top = `${rect.top + window.scrollY - tooltipRect.height - 8}px`;

    // Center horizontally
    const centerOffset = (rect.width - tooltipRect.width) / 2;
    tooltip.style.left = `${rect.left + window.scrollX + centerOffset}px`;

    // Keep in viewport
    const finalRect = tooltip.getBoundingClientRect();
    if (finalRect.left < 10) {
      tooltip.style.left = '10px';
    }
    if (finalRect.right > window.innerWidth - 10) {
      tooltip.style.left = `${window.innerWidth - finalRect.width - 10}px`;
    }
    if (finalRect.top < 10) {
      // Show below instead
      tooltip.style.top = `${rect.bottom + window.scrollY + 8}px`;
    }

    State.activeTooltip = tooltip;
  }

  /**
   * Hide active tooltip
   */
  function hideTooltip() {
    if (State.activeTooltip) {
      State.activeTooltip.remove();
      State.activeTooltip = null;
    }
  }

  // ============================================================================
  // NOTIFICATION MANAGEMENT
  // ============================================================================

  /**
   * Show notification
   */
  function showNotification(message, type = 'info', duration = 3000) {
    // Remove existing notifications
    document.querySelectorAll(`.${NOTIFICATION_CLASS}`).forEach(n => n.remove());

    const notification = document.createElement('div');
    notification.className = `${NOTIFICATION_CLASS} ${NOTIFICATION_CLASS}--${type}`;
    notification.textContent = message;

    document.body.appendChild(notification);

    // Animate in
    requestAnimationFrame(() => {
      notification.classList.add(`${NOTIFICATION_CLASS}--visible`);
    });

    // Auto-remove
    if (duration > 0) {
      setTimeout(() => {
        notification.classList.remove(`${NOTIFICATION_CLASS}--visible`);
        setTimeout(() => notification.remove(), 300);
      }, duration);
    }

    return notification;
  }

  /**
   * Show loading indicator
   */
  function showLoading(message = 'Processing...') {
    // Remove existing loading indicators
    document.querySelectorAll(`.${LOADING_CLASS}`).forEach(l => l.remove());

    const loading = document.createElement('div');
    loading.className = LOADING_CLASS;
    loading.innerHTML = `
      <div class="${LOADING_CLASS}-spinner"></div>
      <div class="${LOADING_CLASS}-message">${escapeHtml(message)}</div>
    `;

    document.body.appendChild(loading);
    return loading;
  }

  /**
   * Hide loading indicator
   */
  function hideLoading() {
    document.querySelectorAll(`.${LOADING_CLASS}`).forEach(l => l.remove());
  }

  // ============================================================================
  // TEXT REPLACEMENT
  // ============================================================================

  /**
   * Replace selected text with complexified version
   */
  function replaceSelection(result) {
    const selection = window.getSelection();
    if (!selection.rangeCount) return;

    const { modifiedText, substitutions, substitutionsMade } = result;

    if (substitutionsMade === 0) {
      showNotification('No substitutions found for selected text', 'info');
      return;
    }

    const range = selection.getRangeAt(0);

    // Create a document fragment with the modified text
    const fragment = createReplacementFragment(modifiedText, substitutions);

    // Replace selection
    range.deleteContents();
    range.insertNode(fragment);

    // Clear selection
    selection.removeAllRanges();

    showNotification(`Made ${substitutionsMade} substitution(s)`, 'success');
  }

  /**
   * Create a document fragment with replaced words wrapped in spans
   */
  function createReplacementFragment(text, substitutions) {
    const fragment = document.createDocumentFragment();

    // Build a map of replacements for quick lookup
    const replacementMap = new Map();
    for (const sub of substitutions) {
      replacementMap.set(sub.replacement.toLowerCase(), sub.original);
    }

    // Split text and process each word
    const parts = text.split(/(\s+)/);

    for (const part of parts) {
      if (/^\s+$/.test(part)) {
        // Whitespace
        fragment.appendChild(document.createTextNode(part));
      } else {
        // Check if this word is a replacement
        const cleanWord = part.replace(/^[.,!?;:'\"()-]+|[.,!?;:'\"()-]+$/g, '');
        const leading = part.match(/^[.,!?;:'\"()-]*/)[0];
        const trailing = part.match(/[.,!?;:'\"()-]*$/)[0];

        if (replacementMap.has(cleanWord.toLowerCase())) {
          const original = replacementMap.get(cleanWord.toLowerCase());

          // Add leading punctuation
          if (leading) {
            fragment.appendChild(document.createTextNode(leading));
          }

          // Create span for replaced word
          const span = document.createElement('span');
          span.className = COMPLEXIFIED_CLASS;
          span.textContent = cleanWord;
          span.dataset.original = original;

          // Add event listeners
          span.addEventListener('mouseenter', () => showTooltip(span, original));
          span.addEventListener('mouseleave', hideTooltip);
          span.addEventListener('click', () => revertWord(span, original));

          fragment.appendChild(span);
          State.replacements.set(span, original);

          // Add trailing punctuation
          if (trailing) {
            fragment.appendChild(document.createTextNode(trailing));
          }

          // Remove from map so we don't match again
          replacementMap.delete(cleanWord.toLowerCase());
        } else {
          // Regular word
          fragment.appendChild(document.createTextNode(part));
        }
      }
    }

    return fragment;
  }

  /**
   * Apply replacements to the whole page
   */
  function applyReplacements(substitutions) {
    if (!substitutions || substitutions.length === 0) {
      showNotification('No substitutions found on this page', 'info');
      return;
    }

    let count = 0;

    for (const sub of substitutions) {
      const replaced = replaceWordInDocument(sub.original, sub.replacement);
      if (replaced) {
        count++;
      }
    }

    showNotification(`Made ${count} substitution(s) on the page`, 'success');
  }

  /**
   * Replace a word throughout the document
   */
  function replaceWordInDocument(originalWord, replacementWord) {
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
          // Skip elements we've already processed
          if (node.parentElement.classList.contains(COMPLEXIFIED_CLASS)) {
            return NodeFilter.FILTER_REJECT;
          }
          // Check if text contains the word
          const regex = new RegExp(`\\b${escapeRegExp(originalWord)}\\b`, 'i');
          if (regex.test(node.textContent)) {
            return NodeFilter.FILTER_ACCEPT;
          }
          return NodeFilter.FILTER_REJECT;
        }
      }
    );

    const nodesToReplace = [];
    while (walker.nextNode()) {
      nodesToReplace.push(walker.currentNode);
    }

    // Only replace first occurrence
    if (nodesToReplace.length === 0) {
      return false;
    }

    const node = nodesToReplace[0];
    const regex = new RegExp(`(\\b)(${escapeRegExp(originalWord)})(\\b)`, 'i');
    const match = node.textContent.match(regex);

    if (!match) return false;

    const matchIndex = match.index + match[1].length;
    const matchedWord = match[2];

    // Split the text node
    const before = node.textContent.substring(0, matchIndex);
    const after = node.textContent.substring(matchIndex + matchedWord.length);

    // Create replacement span
    const span = document.createElement('span');
    span.className = COMPLEXIFIED_CLASS;
    span.textContent = preserveCase(matchedWord, replacementWord);
    span.dataset.original = matchedWord;

    span.addEventListener('mouseenter', () => showTooltip(span, matchedWord));
    span.addEventListener('mouseleave', hideTooltip);
    span.addEventListener('click', () => revertWord(span, matchedWord));

    // Replace the text node
    const parent = node.parentNode;
    const fragment = document.createDocumentFragment();

    if (before) {
      fragment.appendChild(document.createTextNode(before));
    }
    fragment.appendChild(span);
    if (after) {
      fragment.appendChild(document.createTextNode(after));
    }

    parent.replaceChild(fragment, node);
    State.replacements.set(span, matchedWord);

    return true;
  }

  /**
   * Revert a replaced word to original
   */
  function revertWord(span, originalWord) {
    hideTooltip();

    const textNode = document.createTextNode(originalWord);
    span.parentNode.replaceChild(textNode, span);
    State.replacements.delete(span);

    showNotification(`Reverted to "${originalWord}"`, 'info', 2000);
  }

  // ============================================================================
  // UTILITY FUNCTIONS
  // ============================================================================

  /**
   * Escape HTML to prevent XSS
   */
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Escape string for use in regex
   */
  function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  /**
   * Preserve case when replacing
   */
  function preserveCase(original, replacement) {
    if (original === original.toUpperCase()) {
      return replacement.toUpperCase();
    }
    if (original[0] === original[0].toUpperCase()) {
      return replacement.charAt(0).toUpperCase() + replacement.slice(1);
    }
    return replacement;
  }

  // ============================================================================
  // MESSAGE HANDLING
  // ============================================================================

  /**
   * Listen for messages from background script
   */
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    switch (message.type) {
      case 'ping':
        // Respond to ping to confirm content script is loaded
        sendResponse({ alive: true });
        return true; // Async response

      case 'getPageText':
        // Return visible text from the page
        const text = getVisibleText();
        sendResponse({ text });
        return true; // Async response

      case 'replaceSelection':
        hideLoading();
        if (message.result) {
          replaceSelection(message.result);
        }
        break;

      case 'applyReplacements':
        hideLoading();
        if (message.substitutions) {
          applyReplacements(message.substitutions);
        }
        break;

      case 'showLoading':
        showLoading(message.message);
        break;

      case 'hideLoading':
        hideLoading();
        break;

      case 'showError':
        hideLoading();
        showNotification(message.message || 'An error occurred', 'error');
        break;

      case 'showNotification':
        showNotification(message.message, message.notificationType || 'info', message.duration);
        break;
    }
    // No return true for synchronous cases - channel closes immediately
  });

  /**
   * Get visible text from the page
   */
  function getVisibleText(maxWords = 1000) {
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode: (node) => {
          const style = window.getComputedStyle(node.parentElement);
          if (style.display === 'none' || style.visibility === 'hidden') {
            return NodeFilter.FILTER_REJECT;
          }
          if (['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(node.parentElement.tagName)) {
            return NodeFilter.FILTER_REJECT;
          }
          if (node.textContent.trim().length > 0) {
            return NodeFilter.FILTER_ACCEPT;
          }
          return NodeFilter.FILTER_REJECT;
        }
      }
    );

    const texts = [];
    let wordCount = 0;

    while (walker.nextNode() && wordCount < maxWords) {
      const text = walker.currentNode.textContent.trim();
      const words = text.split(/\s+/).length;
      texts.push(text);
      wordCount += words;
    }

    return texts.join(' ');
  }

  // ============================================================================
  // INITIALIZATION
  // ============================================================================

  // Notify that content script is ready
  chrome.runtime.sendMessage({ type: 'contentScriptReady' }).catch(() => {});

})();
