/**
 * Offscreen Document - Complexity Injector (ES Module)
 *
 * Runs AI inference directly in the offscreen document context.
 * This is necessary because ONNX Runtime Web creates internal workers,
 * and Chrome extensions block blob URL workers created from within workers.
 * Running directly here (a regular page context) allows ONNX to work normally.
 */

// Import Transformers.js
import { pipeline, env } from './transformers.min.js';

// Import data directly as ES module exports
import {
  CONFIG,
  BlockReason,
  GREVocabularyDatabase,
  AntonymPairs,
  IdiomDatabase,
  ProperNounPatterns,
  TitlePatterns,
  NegatorWords,
  DiminisherWords,
  IntensifierWords,
} from './data.js';

// Configure ONNX for Chrome extension environment
env.allowLocalModels = false;
env.useBrowserCache = true;

// Disable ONNX WASM proxy if the property exists (for some versions)
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.proxy = false;
  env.backends.onnx.wasm.numThreads = 1;
}

console.log('[Offscreen] Module loaded, ONNX configured');
console.log('[Offscreen] GREVocabularyDatabase loaded:', !!GREVocabularyDatabase);
console.log('[Offscreen] GREVocabularyDatabase keys:', GREVocabularyDatabase ? Object.keys(GREVocabularyDatabase).length : 0);

// ============================================================================
// STATE
// ============================================================================

const State = {
  modelLoaded: false,
  modelLoading: false,
  extractor: null,
  fillMask: null,
  embeddingCache: new Map(),
  contextCache: new Map(),
  customVocabulary: new Map(),
};

// ============================================================================
// INITIALIZATION
// ============================================================================

async function initialize() {
  console.log('[Offscreen] Initializing...');
  console.log('[Offscreen] CONFIG:', !!CONFIG);
  console.log('[Offscreen] GREVocabularyDatabase keys:', Object.keys(GREVocabularyDatabase).length);
  await loadModel();
  return { success: true, message: 'Initialized successfully' };
}

async function loadModel() {
  if (State.modelLoaded || State.modelLoading) return;

  State.modelLoading = true;
  sendStatus('loading', 'Loading feature extraction model...');

  try {
    State.extractor = await pipeline('feature-extraction', 'Xenova/distilbert-base-uncased', {
      quantized: true,
    });

    sendStatus('loading', 'Loading fill-mask model...');

    State.fillMask = await pipeline('fill-mask', 'Xenova/distilbert-base-uncased', {
      quantized: true,
    });

    State.modelLoaded = true;
    State.modelLoading = false;
    sendStatus('ready', 'Models loaded successfully');
    console.log('[Offscreen] Models loaded successfully');

  } catch (error) {
    State.modelLoading = false;
    sendStatus('error', `Failed to load model: ${error.message}`);
    throw error;
  }
}

// ============================================================================
// EMBEDDING & SCORING FUNCTIONS
// ============================================================================

async function getEmbedding(word) {
  const lower = word.toLowerCase();

  if (State.embeddingCache.has(lower)) {
    return State.embeddingCache.get(lower);
  }

  if (!State.modelLoaded || !State.extractor) return null;

  try {
    const output = await State.extractor(word, { pooling: 'mean', normalize: true });
    const embedding = new Float32Array(output.data);
    State.embeddingCache.set(lower, embedding);
    return embedding;
  } catch (error) {
    console.error(`Error getting embedding for "${word}":`, error);
    return null;
  }
}

async function getContextVector(sentence) {
  if (!State.modelLoaded || !State.extractor) return null;

  try {
    const output = await State.extractor(sentence, { pooling: 'mean', normalize: true });
    return new Float32Array(output.data);
  } catch (error) {
    console.error('Error getting context vector:', error);
    return null;
  }
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

async function scoreSyntax(maskedSentence, candidate) {
  if (!State.modelLoaded || !State.fillMask) return -10;

  try {
    const predictions = await State.fillMask(maskedSentence, { topk: 100 });
    const candidateLower = candidate.toLowerCase();

    for (const pred of predictions) {
      if (pred.token_str && pred.token_str.toLowerCase().trim() === candidateLower) {
        return Math.log(pred.score + 1e-10);
      }
    }
    return -10;
  } catch (error) {
    console.error('Error scoring syntax:', error);
    return -10;
  }
}

// ============================================================================
// FILTERING FUNCTIONS
// ============================================================================

function isAntonym(word1, word2) {
  const w1 = word1.toLowerCase();
  const w2 = word2.toLowerCase();
  for (const pair of AntonymPairs) {
    if ((pair[0] === w1 && pair[1] === w2) || (pair[0] === w2 && pair[1] === w1)) {
      return true;
    }
  }
  return false;
}

function checkIdiom(sentence, targetWord) {
  const lower = targetWord.toLowerCase();
  const sentenceLower = sentence.toLowerCase();
  if (!IdiomDatabase[lower]) return { isIdiom: false };
  for (const idiom of IdiomDatabase[lower]) {
    if (sentenceLower.includes(idiom.phrase)) {
      return { isIdiom: true, meaning: idiom.meaning };
    }
  }
  return { isIdiom: false };
}

function checkProperNoun(sentence, targetWord) {
  const lower = targetWord.toLowerCase();
  const sentenceLower = sentence.toLowerCase();
  for (const pattern of ProperNounPatterns) {
    if (sentenceLower.includes(pattern) && pattern.includes(lower)) {
      return { isProperNoun: true };
    }
  }
  return { isProperNoun: false };
}

function checkNegation(sentence, targetWord) {
  const lower = targetWord.toLowerCase();
  const sentenceLower = sentence.toLowerCase();
  const targetIdx = sentenceLower.indexOf(lower);
  if (targetIdx === -1) return { isNegated: false, expanded: targetWord };

  const precedingText = sentenceLower.substring(0, targetIdx).trim();
  const precedingWords = precedingText.split(/\s+/).slice(-4);

  for (const word of precedingWords) {
    const cleaned = word.replace(/[.,!?;:'\"()-]/g, '');
    if (NegatorWords.has(cleaned) || DiminisherWords.has(cleaned)) {
      return { isNegated: true, expanded: targetWord };
    }
  }
  return { isNegated: false, expanded: targetWord };
}

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// ============================================================================
// MAIN PROCESSING PIPELINE
// ============================================================================

async function processSubstitution(sentence, original, candidate) {
  if (!State.modelLoaded) {
    return { original, candidate, passed: false, reason: 'MODEL_NOT_READY' };
  }

  if (isAntonym(original, candidate)) {
    return { original, candidate, passed: false, reason: 'ANTONYM_DETECTED' };
  }

  const embOriginal = await getEmbedding(original);
  const embCandidate = await getEmbedding(candidate);
  const similarity = cosineSimilarity(embOriginal, embCandidate);

  console.log(`[Offscreen] ${original} -> ${candidate}: similarity=${similarity?.toFixed(3)} (min=${CONFIG.EMBEDDING_MIN}, max=${CONFIG.EMBEDDING_MAX})`);

  if (similarity < CONFIG.EMBEDDING_MIN) {
    return { original, candidate, passed: false, reason: 'NOT_SIMILAR_ENOUGH', similarity };
  }
  if (similarity > CONFIG.EMBEDDING_MAX) {
    return { original, candidate, passed: false, reason: 'TOO_SIMILAR', similarity };
  }

  if (checkProperNoun(sentence, original).isProperNoun) {
    return { original, candidate, passed: false, reason: 'PROPER_NOUN', similarity };
  }
  if (checkIdiom(sentence, original).isIdiom) {
    return { original, candidate, passed: false, reason: 'IDIOM_DETECTED', similarity };
  }

  const negationResult = checkNegation(sentence, original);
  if (negationResult.isNegated) {
    return { original, candidate, passed: false, reason: 'NEGATION_CONTEXT', similarity };
  }

  // Create masked sentence for syntax scoring (case-insensitive replacement)
  const maskedSentence = sentence.replace(
    new RegExp(`\\b${escapeRegExp(negationResult.expanded)}\\b`, 'i'),
    '[MASK]'
  );
  console.log(`[Offscreen] Masked sentence: "${maskedSentence.substring(0, 100)}..."`);

  const syntaxScore = await scoreSyntax(maskedSentence, candidate);
  // Use similarity as semantic score if it's above trust threshold, otherwise use lower value
  let semanticScore = similarity >= CONFIG.EMBEDDING_TRUST_THRESHOLD ? similarity : similarity * 0.5;

  console.log(`[Offscreen] ${original} -> ${candidate}: syntax=${syntaxScore?.toFixed(2)} (floor=${CONFIG.SYNTAX_FLOOR}), semantic=${semanticScore?.toFixed(3)} (floor=${CONFIG.SEMANTIC_FLOOR})`);

  let passed = false;
  let passReason = '';
  if (syntaxScore > CONFIG.SYNTAX_FLOOR) {
    passed = semanticScore > CONFIG.SEMANTIC_FLOOR;
    passReason = passed ? 'syntax+semantic OK' : `semantic ${semanticScore?.toFixed(3)} < floor ${CONFIG.SEMANTIC_FLOOR}`;
  } else if (semanticScore > CONFIG.SEMANTIC_OVERRIDE) {
    passed = true;
    passReason = 'semantic override';
  } else {
    passReason = `syntax ${syntaxScore?.toFixed(2)} < floor ${CONFIG.SYNTAX_FLOOR}`;
  }

  console.log(`[Offscreen] ${original} -> ${candidate}: PASSED=${passed} (${passReason})`);

  return { original, candidate, passed, similarity, syntaxScore, semanticScore };
}

async function findBestSubstitution(sentence, word) {
  const lower = word.toLowerCase();
  let candidates = GREVocabularyDatabase[lower] || [];

  if (State.customVocabulary.has(lower)) {
    candidates = [...candidates, ...State.customVocabulary.get(lower)];
  }

  if (candidates.length === 0) return null;

  console.log(`[Offscreen] Finding substitution for "${word}", ${candidates.length} candidates`);

  const results = [];
  for (const candidate of candidates) {
    const result = await processSubstitution(sentence, word, candidate.word);
    console.log(`[Offscreen]   ${word} -> ${candidate.word}: passed=${result.passed}, reason=${result.reason || 'PASSED'}, similarity=${result.similarity?.toFixed(3)}`);
    if (result.passed) {
      results.push({ ...result, candidateInfo: candidate });
    }
  }

  if (results.length === 0) {
    console.log(`[Offscreen] No valid substitutions found for "${word}"`);
    return null;
  }
  results.sort((a, b) => b.syntaxScore - a.syntaxScore);
  return results[0];
}

async function processText(text, maxDensity = CONFIG.MAX_DENSITY) {
  console.log('[Offscreen] processText called, text length:', text.length);
  console.log('[Offscreen] GREVocabularyDatabase loaded:', !!GREVocabularyDatabase);
  console.log('[Offscreen] GREVocabularyDatabase keys:', GREVocabularyDatabase ? Object.keys(GREVocabularyDatabase).length : 0);
  console.log('[Offscreen] State.modelLoaded:', State.modelLoaded);

  const words = text.split(/\s+/);
  const maxSubs = Math.max(1, Math.ceil(words.length * maxDensity));
  const wordsToProcess = words.slice(0, CONFIG.MAX_WORDS_PER_BATCH);

  const allResults = [];
  const processedWords = new Set();
  const vocabMatches = [];

  for (const word of wordsToProcess) {
    const clean = word.toLowerCase().replace(/[.,!?;:'\"()-]/g, '');
    if (processedWords.has(clean) || clean.length < 2) continue;

    if (GREVocabularyDatabase && GREVocabularyDatabase[clean]) {
      vocabMatches.push(clean);
      const result = await findBestSubstitution(text, clean);
      if (result) {
        allResults.push(result);
        processedWords.add(clean);
      }
    } else if (State.customVocabulary.has(clean)) {
      vocabMatches.push(clean + ' (custom)');
      const result = await findBestSubstitution(text, clean);
      if (result) {
        allResults.push(result);
        processedWords.add(clean);
      }
    }
  }

  console.log('[Offscreen] Vocabulary matches found:', vocabMatches);
  console.log('[Offscreen] Substitutions that passed filters:', allResults.length);

  allResults.sort((a, b) => b.syntaxScore - a.syntaxScore);
  const selectedSubs = allResults.slice(0, maxSubs);

  let modifiedText = text;
  const substitutions = [];

  for (const sub of selectedSubs) {
    const pattern = new RegExp(`\\b${escapeRegExp(sub.original)}\\b`, 'i');
    const newText = modifiedText.replace(pattern, sub.candidate);
    if (newText !== modifiedText) {
      substitutions.push({
        original: sub.original,
        replacement: sub.candidate,
        similarity: sub.similarity,
        syntaxScore: sub.syntaxScore,
      });
      modifiedText = newText;
    }
  }

  return { originalText: text, modifiedText, substitutions, substitutionsMade: substitutions.length };
}

function findVocabularyWordsInText(text) {
  const words = text.toLowerCase().split(/\s+/);
  const found = [];
  for (const word of words) {
    const clean = word.replace(/[.,!?;:'\"()-]/g, '');
    if (GREVocabularyDatabase[clean] || State.customVocabulary.has(clean)) {
      found.push(clean);
    }
  }
  return [...new Set(found)];
}

// ============================================================================
// MESSAGE HANDLING
// ============================================================================

function sendStatus(status, message) {
  chrome.runtime.sendMessage({
    target: 'background',
    type: 'workerStatus',
    status,
    message,
  }).catch(() => {});
}

function sendResponse(id, result) {
  chrome.runtime.sendMessage({
    target: 'background',
    type: 'workerResponse',
    originalType: 'response',
    id,
    result,
  }).catch(err => console.error('[Offscreen] Failed to send response:', err));
}

function sendError(id, error) {
  chrome.runtime.sendMessage({
    target: 'background',
    type: 'workerResponse',
    originalType: 'error',
    id,
    error,
  }).catch(err => console.error('[Offscreen] Failed to send error:', err));
}

// ============================================================================
// CUSTOM VOCABULARY MANAGEMENT
// ============================================================================

async function addCustomVocabulary(data) {
  // data is { vocabulary: [...] } where each entry has { word, synonym, definition?, examples? }
  const vocabulary = data.vocabulary || data;

  if (!Array.isArray(vocabulary)) {
    return { error: 'Invalid vocabulary data: expected array' };
  }

  for (const entry of vocabulary) {
    const lower = entry.word.toLowerCase();

    if (!State.customVocabulary.has(lower)) {
      State.customVocabulary.set(lower, []);
    }

    State.customVocabulary.get(lower).push({
      word: entry.synonym,
      pos: entry.pos || 'unknown',
      domain: entry.domain || 'general',
      definition: entry.definition || '',
      examples: entry.examples || [],
    });

    // Pre-compute embedding for new word
    await getEmbedding(entry.synonym);
  }

  return { success: true, count: vocabulary.length };
}

function clearCustomVocabulary() {
  State.customVocabulary.clear();
  return { success: true };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponseSync) => {
  if (message.target !== 'offscreen') return false;

  console.log('[Offscreen] Received:', message.type || message.workerType, message.id);

  if (message.type === 'workerMessage') {
    handleWorkerMessage(message.workerType, message.id, message.data);
    sendResponseSync({ acknowledged: true });
  } else if (message.type === 'getWorkerStatus') {
    sendResponseSync({ ready: State.modelLoaded, initialized: true });
  } else if (message.type === 'ping') {
    sendResponseSync({ alive: true, workerReady: State.modelLoaded });
  } else {
    sendResponseSync({ error: `Unknown message type: ${message.type}` });
  }

  return true;
});

async function handleWorkerMessage(type, id, data) {
  try {
    let result;

    switch (type) {
      case 'init':
        result = await initialize();
        break;
      case 'loadModel':
        await loadModel();
        result = { success: true };
        break;
      case 'getStatus':
        result = {
          modelLoaded: State.modelLoaded,
          modelLoading: State.modelLoading,
          embeddingsCached: State.embeddingCache.size,
        };
        break;
      case 'processSubstitution':
        result = await processSubstitution(data.sentence, data.original, data.candidate);
        break;
      case 'findBestSubstitution':
        result = await findBestSubstitution(data.sentence, data.word);
        break;
      case 'processText':
        result = await processText(data.text, data.maxDensity);
        break;
      case 'findVocabularyWords':
        result = findVocabularyWordsInText(data.text);
        break;
      case 'getVocabulary':
        result = {
          default: Object.keys(GREVocabularyDatabase || {}),
          custom: Array.from(State.customVocabulary.keys()),
        };
        break;
      case 'addCustomVocabulary':
        result = await addCustomVocabulary(data);
        break;
      case 'clearCustomVocabulary':
        result = clearCustomVocabulary();
        break;
      default:
        result = { error: `Unknown type: ${type}` };
    }

    sendResponse(id, result);
  } catch (error) {
    console.error('[Offscreen] Error handling message:', error);
    sendError(id, error.message);
  }
}

// ============================================================================
// STARTUP
// ============================================================================

console.log('[Offscreen] Document loaded');
console.log('[Offscreen] Vocabulary database has', Object.keys(GREVocabularyDatabase).length, 'entries');
sendStatus('loaded', 'Offscreen document ready, awaiting initialization');
