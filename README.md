# Complexity Injector - Chrome Extension

A Manifest V3 Chrome Extension that intelligently complexifies text using AI-powered synonym substitution with DistilBERT.

## Features

- **Context Menu Integration**: Right-click on selected text to "Complexify Selection"
- **Whole Page Mode**: Process entire visible page text with configurable density
- **Hover Tooltips**: See original words by hovering over replaced text
- **Click to Revert**: Click on any replaced word to restore the original
- **Custom Vocabulary**: Upload CSV files with custom word pairs
- **Local Processing**: All AI processing happens locally in the browser

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Chrome Extension                          │
├─────────────────────────────────────────────────────────────┤
│  popup.html/js     │  background.js    │  content_script.js │
│  (User Interface)  │  (Service Worker) │  (DOM Manipulation)│
├─────────────────────────────────────────────────────────────┤
│                      ai_worker.js                            │
│              (Web Worker with Transformers.js)               │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                V8 Pipeline Logic                      │  │
│  │                                                       │  │
│  │  Layer 0: Antonym Check                              │  │
│  │  Layer 1: Embedding Similarity (0.55 - 0.96)         │  │
│  │  Layer 2: Proper Noun Detection                      │  │
│  │  Layer 3: Idiom Detection                            │  │
│  │  Layer 4: Negation Context Detection                 │  │
│  │  Layer 5: Context Vector Extraction                  │  │
│  │  Layer 6: Syntax Scoring (DistilBERT MLM)           │  │
│  │  Layer 7: Semantic Scoring                           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Installation

1. Clone or download this repository
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" (toggle in top right)
4. Click "Load unpacked"
5. Select the `extension` folder

## Usage

### Right-Click Menu (Context Menu)
1. Select any text on a webpage
2. Right-click and choose "Complexify Selection"
3. The selected text will be processed and simple words will be replaced

### Whole Page Mode
1. Click the extension icon in the toolbar
2. Adjust the "Substitution Density" slider (1-20%)
3. Click "Complexify Page"
4. Wait for processing (model loads on first use)

### Custom Vocabulary
1. Prepare a CSV file with columns: `Word,Synonym` (optionally: `Definition`)
2. Click "Upload CSV" in the popup
3. Custom words will be added to the vocabulary

### Interacting with Replaced Words
- **Hover**: See the original word in a tooltip
- **Click**: Revert to the original word

## Thresholds (from Python V8 Pipeline)

| Threshold | Value | Description |
|-----------|-------|-------------|
| EMBEDDING_MIN | 0.55 | Minimum similarity to be considered |
| EMBEDDING_MAX | 0.96 | Maximum similarity (too similar = no change) |
| EMBEDDING_TRUST_THRESHOLD | 0.75 | Use embedding as semantic fallback |
| SYNTAX_FLOOR | -3.5 | Minimum syntax score to pass |
| SEMANTIC_FLOOR | 0.45 | Minimum semantic score |
| SEMANTIC_OVERRIDE | 0.80 | Override poor syntax if semantic > 0.80 |
| MAX_DENSITY | 0.08 | Default max percentage of words to substitute |

## File Structure

```
extension/
├── manifest.json       # Extension manifest (V3)
├── background.js       # Service worker for context menu
├── ai_worker.js        # Web Worker with Transformers.js & pipeline
├── content_script.js   # DOM manipulation & hover UI
├── data.js             # Vocabulary, antonyms, idioms databases
├── popup.html          # Extension popup UI
├── popup.js            # Popup interaction logic
├── popup.css           # Popup styles
├── styles.css          # Content script styles
└── icons/              # Extension icons
    ├── icon16.png
    ├── icon48.png
    └── icon128.png
```

## Model Details

The extension uses:
- **Model**: `Xenova/distilbert-base-uncased` (quantized)
- **Library**: [Transformers.js](https://huggingface.co/docs/transformers.js)
- **Loading**: Fetched from HuggingFace CDN on first use, then cached

## Privacy

- **All processing happens locally** in the browser
- No data is sent to external servers (except initial model download from HuggingFace)
- Custom vocabulary is stored locally in `chrome.storage.local`

## Credits

Based on the unified_complexifier_v2.py Python script implementing the V8 Pipeline logic.

## License

MIT License
