/**
 * AI Worker - Complexity Injector (ES Module)
 *
 * Web Worker that handles heavy AI processing using Transformers.js
 * Implements the V8 Pipeline logic from unified_complexifier_v2.py
 *
 * Features:
 * - DistilBERT for embeddings and masked language modeling
 * - Layered filtering: Antonym → Idiom → Proper Noun → Negation → AI Scoring
 * - Embedding and context caching for performance
 *
 * Note: This worker uses ES modules (type: 'module') to support Transformers.js
 */

// Import Transformers.js as ES module (bundled locally for Manifest V3 compliance)
import { pipeline, env } from './transformers.min.js';

// CRITICAL: Configure ONNX to disable internal workers IMMEDIATELY after import
// Chrome extensions block blob URL workers that ONNX Runtime creates for multi-threading
// This MUST be set before any pipeline() calls
env.allowLocalModels = false;
env.useBrowserCache = true;

// Disable ONNX WASM proxy (which spawns blob URL workers)
// This is the key setting that prevents the "importScripts blob:" error
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.proxy = false;
  env.backends.onnx.wasm.numThreads = 1;
}
console.log('[AI Worker] ONNX proxy disabled, single-threaded mode');

// Import data module
import './data.js';

// Data variables (will be populated from data.js which sets them on globalThis/self)
let DATA_LOADED = false;
let GREVocabularyDatabase = null;
let AntonymPairs = null;
let IdiomDatabase = null;
let ProperNounPatterns = null;
let TitlePatterns = null;
let NegatorWords = null;
let DiminisherWords = null;
let IntensifierWords = null;
let CONFIG = null;
let BlockReason = null;

// ============================================================================
// CONFIGURATION & STATE
// ============================================================================

const WorkerState = {
  modelLoaded: false,
  modelLoading: false,
  extractor: null,               // Feature extraction pipeline for embeddings
  fillMask: null,                // Fill-mask pipeline for syntax scoring
  embeddingCache: new Map(),     // word -> embedding array
  contextCache: new Map(),       // word -> [context vectors from examples]
  customVocabulary: new Map(),   // User-uploaded vocabulary
};

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize the worker with data and model
 */
async function initialize() {
  // Load data from imported data.js module (attached to self)
  loadDataFromSelf();

  // Load the model
  await loadModel();

  // Pre-compute embeddings for vocabulary
  await precomputeVocabulary();

  return { success: true, message: 'Worker initialized successfully' };
}

/**
 * Load data from self (populated by data.js ES module import)
 */
function loadDataFromSelf() {
  if (DATA_LOADED) return;

  // Read from self (populated by data.js)
  CONFIG = self.CONFIG;
  BlockReason = self.BlockReason;
  GREVocabularyDatabase = self.GREVocabularyDatabase;
  AntonymPairs = self.AntonymPairs;
  IdiomDatabase = self.IdiomDatabase;
  ProperNounPatterns = self.ProperNounPatterns;
  TitlePatterns = self.TitlePatterns;
  NegatorWords = self.NegatorWords;
  DiminisherWords = self.DiminisherWords;
  IntensifierWords = self.IntensifierWords;

  DATA_LOADED = true;
  console.log('[AI Worker] Data loaded from data.js module');
}

/**
 * Load data inline as fallback (copy from data.js for compatibility)
 */
function loadDataInlineFallback() {
  if (DATA_LOADED) return;

  console.log('[AI Worker] Loading inline data fallback...');

  // Configuration
  CONFIG = {
    EMBEDDING_MIN: 0.55,
    EMBEDDING_MAX: 0.96,
    EMBEDDING_TRUST_THRESHOLD: 0.75,
    SYNTAX_FLOOR: -3.5,
    SEMANTIC_FLOOR: 0.45,
    SEMANTIC_OVERRIDE: 0.80,
    MAX_DENSITY: 0.08,
    MAX_WORDS_PER_BATCH: 1000,
  };

  BlockReason = {
    PASSED: 'PASSED',
    ANTONYM_DETECTED: 'ANTONYM_DETECTED',
    NOT_SIMILAR_ENOUGH: 'NOT_SIMILAR_ENOUGH',
    TOO_SIMILAR: 'TOO_SIMILAR',
    PROPER_NOUN: 'PROPER_NOUN',
    IDIOM_DETECTED: 'IDIOM_DETECTED',
    NEGATION_CONTEXT: 'NEGATION_CONTEXT',
    SYNTAX_FAILED: 'SYNTAX_FAILED',
    SEMANTIC_FAILED: 'SEMANTIC_FAILED',
    NOT_IN_VOCAB: 'NOT_IN_VOCAB',
    MODEL_NOT_READY: 'MODEL_NOT_READY',
  };

  // GRE Vocabulary Database (abbreviated - full version in data.js)
  GREVocabularyDatabase = {
    "hot": [
      { word: "scalding", pos: "adj", domain: "physical", definition: "extremely hot, burning", examples: ["The scalding water burned his hand.", "She dropped the scalding coffee.", "The scalding sun beat down on them."] },
      { word: "sweltering", pos: "adj", domain: "physical", definition: "uncomfortably hot", examples: ["The sweltering heat made it hard to breathe.", "We escaped the sweltering afternoon.", "The sweltering room needed air conditioning."] },
      { word: "torrid", pos: "adj", domain: "physical", definition: "very hot and dry", examples: ["The torrid climate dried the land.", "They survived the torrid summer.", "The torrid zone receives intense sunlight."] },
      { word: "scorching", pos: "adj", domain: "physical", definition: "intensely hot", examples: ["The scorching pavement burned her feet.", "We endured the scorching temperatures.", "The scorching wind carried desert sand."] },
    ],
    "cold": [
      { word: "frigid", pos: "adj", domain: "physical", definition: "extremely cold", examples: ["The frigid air froze their breath.", "She shivered in the frigid water.", "The frigid temperatures broke records."] },
      { word: "glacial", pos: "adj", domain: "physical", definition: "icy cold", examples: ["The glacial wind cut through their coats.", "His glacial stare silenced the room.", "The glacial pace frustrated everyone."] },
      { word: "arctic", pos: "adj", domain: "physical", definition: "extremely cold", examples: ["The arctic conditions halted construction.", "They braved the arctic temperatures.", "The arctic blast swept the city."] },
      { word: "frosty", pos: "adj", domain: "physical", definition: "cold with frost", examples: ["The frosty morning covered the grass.", "Her frosty reception was unexpected.", "The frosty windowpanes obscured the view."] },
    ],
    "big": [
      { word: "enormous", pos: "adj", domain: "size", definition: "extremely large", examples: ["The enormous building dominated the skyline.", "An enormous crowd gathered.", "The task required enormous effort."] },
      { word: "colossal", pos: "adj", domain: "size", definition: "extraordinarily large", examples: ["The colossal statue amazed visitors.", "A colossal mistake cost them dearly.", "The colossal wave destroyed everything."] },
      { word: "immense", pos: "adj", domain: "size", definition: "extremely large or great", examples: ["The immense forest stretched for miles.", "She felt immense gratitude.", "The immense pressure was overwhelming."] },
      { word: "mammoth", pos: "adj", domain: "size", definition: "huge, enormous", examples: ["The mammoth project took years.", "A mammoth effort was required.", "The mammoth structure impressed everyone."] },
      { word: "gargantuan", pos: "adj", domain: "size", definition: "enormous, gigantic", examples: ["The gargantuan feast lasted hours.", "His gargantuan appetite was legendary.", "The gargantuan task seemed impossible."] },
    ],
    "small": [
      { word: "minuscule", pos: "adj", domain: "size", definition: "extremely small", examples: ["The minuscule text was hard to read.", "A minuscule amount remained.", "The minuscule insect escaped notice."] },
      { word: "diminutive", pos: "adj", domain: "size", definition: "extremely small", examples: ["The diminutive figure stood in the corner.", "Her diminutive stature belied her strength.", "The diminutive plant bloomed beautifully."] },
      { word: "microscopic", pos: "adj", domain: "size", definition: "extremely small, invisible to naked eye", examples: ["The microscopic organisms multiplied.", "A microscopic crack caused the failure.", "The microscopic details revealed the truth."] },
      { word: "infinitesimal", pos: "adj", domain: "size", definition: "extremely small", examples: ["The infinitesimal chance still existed.", "An infinitesimal difference separated them.", "The infinitesimal particles floated."] },
    ],
    "fast": [
      { word: "rapid", pos: "adj", domain: "temporal", definition: "happening quickly", examples: ["The rapid growth surprised analysts.", "Rapid changes transformed the industry.", "The rapid heartbeat signaled anxiety."] },
      { word: "swift", pos: "adj", domain: "temporal", definition: "moving quickly", examples: ["The swift response saved lives.", "A swift current carried the boat.", "Her swift action prevented disaster."] },
      { word: "expeditious", pos: "adj", domain: "temporal", definition: "quick and efficient", examples: ["The expeditious handling impressed clients.", "An expeditious solution was needed.", "The expeditious process saved time."] },
      { word: "brisk", pos: "adj", domain: "temporal", definition: "quick and energetic", examples: ["The brisk pace tired the group.", "A brisk wind refreshed them.", "The brisk business kept them busy."] },
    ],
    "slow": [
      { word: "sluggish", pos: "adj", domain: "temporal", definition: "slow-moving, lacking energy", examples: ["The sluggish economy worried investors.", "He felt sluggish after lunch.", "The sluggish response frustrated users."] },
      { word: "lethargic", pos: "adj", domain: "temporal", definition: "sluggish and apathetic", examples: ["The lethargic patient needed rest.", "A lethargic attitude pervaded the office.", "The lethargic market showed little activity."] },
      { word: "leisurely", pos: "adj", domain: "temporal", definition: "unhurried, relaxed", examples: ["They enjoyed a leisurely breakfast.", "The leisurely pace suited them.", "A leisurely stroll calmed her nerves."] },
      { word: "gradual", pos: "adj", domain: "temporal", definition: "happening slowly over time", examples: ["The gradual change went unnoticed.", "A gradual improvement was evident.", "The gradual decline worried experts."] },
    ],
    "good": [
      { word: "excellent", pos: "adj", domain: "quality", definition: "extremely good", examples: ["The excellent performance earned applause.", "She has excellent taste.", "The excellent results exceeded expectations."] },
      { word: "superb", pos: "adj", domain: "quality", definition: "of the highest quality", examples: ["The superb craftsmanship was evident.", "A superb meal awaited them.", "The superb view took their breath away."] },
      { word: "exemplary", pos: "adj", domain: "quality", definition: "serving as a desirable model", examples: ["His exemplary behavior inspired others.", "The exemplary work earned recognition.", "She set an exemplary standard."] },
      { word: "outstanding", pos: "adj", domain: "quality", definition: "exceptionally good", examples: ["The outstanding achievement was celebrated.", "An outstanding performance impressed critics.", "Her outstanding dedication was recognized."] },
      { word: "impeccable", pos: "adj", domain: "quality", definition: "flawless, perfect", examples: ["His impeccable manners charmed everyone.", "The impeccable timing was crucial.", "Her impeccable record spoke for itself."] },
    ],
    "bad": [
      { word: "atrocious", pos: "adj", domain: "quality", definition: "horrifyingly bad", examples: ["The atrocious conditions shocked inspectors.", "His atrocious behavior was unacceptable.", "The atrocious weather ruined the event."] },
      { word: "deplorable", pos: "adj", domain: "quality", definition: "deserving strong condemnation", examples: ["The deplorable state of affairs demanded action.", "Deplorable conditions persisted.", "The deplorable treatment was condemned."] },
      { word: "abysmal", pos: "adj", domain: "quality", definition: "extremely bad", examples: ["The abysmal performance disappointed fans.", "Abysmal grades threatened his future.", "The abysmal service drove customers away."] },
      { word: "dreadful", pos: "adj", domain: "quality", definition: "causing great suffering or fear", examples: ["The dreadful news devastated the family.", "A dreadful mistake was made.", "The dreadful storm caused havoc."] },
      { word: "appalling", pos: "adj", domain: "quality", definition: "causing shock or dismay", examples: ["The appalling lack of care was evident.", "Appalling statistics emerged.", "The appalling decision backfired."] },
    ],
    "happy": [
      { word: "elated", pos: "adj", domain: "emotional", definition: "ecstatically happy", examples: ["She was elated by the news.", "The elated crowd cheered loudly.", "He felt elated after winning."] },
      { word: "jubilant", pos: "adj", domain: "emotional", definition: "feeling or expressing great happiness", examples: ["The jubilant fans celebrated.", "A jubilant atmosphere filled the room.", "They were jubilant over the victory."] },
      { word: "ecstatic", pos: "adj", domain: "emotional", definition: "feeling overwhelming happiness", examples: ["She was ecstatic about the promotion.", "The ecstatic response surprised everyone.", "He felt ecstatic seeing her again."] },
      { word: "euphoric", pos: "adj", domain: "emotional", definition: "intensely happy and confident", examples: ["The euphoric feeling lasted for days.", "A euphoric crowd filled the streets.", "She felt euphoric after the achievement."] },
      { word: "exuberant", pos: "adj", domain: "emotional", definition: "filled with lively energy and excitement", examples: ["The exuberant children played happily.", "His exuberant personality attracted friends.", "The exuberant celebration continued."] },
    ],
    "sad": [
      { word: "melancholy", pos: "adj", domain: "emotional", definition: "a feeling of pensive sadness", examples: ["A melancholy mood settled over her.", "The melancholy music touched their hearts.", "He felt melancholy on rainy days."] },
      { word: "despondent", pos: "adj", domain: "emotional", definition: "in low spirits from loss of hope", examples: ["The despondent athlete gave up.", "She grew despondent after the rejection.", "The despondent patient needed support."] },
      { word: "morose", pos: "adj", domain: "emotional", definition: "sullen and ill-tempered", examples: ["His morose attitude worried his friends.", "The morose expression never left his face.", "She became morose after the loss."] },
      { word: "forlorn", pos: "adj", domain: "emotional", definition: "pitifully sad and lonely", examples: ["The forlorn puppy waited by the door.", "A forlorn hope remained.", "She looked forlorn sitting alone."] },
      { word: "woeful", pos: "adj", domain: "emotional", definition: "characterized by or full of sorrow", examples: ["The woeful tale brought tears.", "A woeful expression crossed her face.", "The woeful situation demanded attention."] },
    ],
    "difficult": [
      { word: "arduous", pos: "adj", domain: "mental", definition: "involving great effort", examples: ["The arduous journey tested their limits.", "An arduous task lay ahead.", "The arduous climb was worth it."] },
      { word: "formidable", pos: "adj", domain: "mental", definition: "inspiring fear or respect through difficulty", examples: ["The formidable challenge awaited.", "A formidable opponent emerged.", "The formidable task required expertise."] },
      { word: "onerous", pos: "adj", domain: "mental", definition: "involving heavy burden", examples: ["The onerous duties exhausted him.", "An onerous responsibility was assigned.", "The onerous regulations frustrated businesses."] },
      { word: "strenuous", pos: "adj", domain: "physical", definition: "requiring great effort", examples: ["The strenuous workout left her tired.", "A strenuous effort was made.", "The strenuous activity built strength."] },
      { word: "laborious", pos: "adj", domain: "mental", definition: "requiring much time and effort", examples: ["The laborious process took months.", "A laborious task was completed.", "The laborious research paid off."] },
    ],
    "hard": [
      { word: "challenging", pos: "adj", domain: "mental", definition: "testing one's abilities", examples: ["The challenging exam stumped many.", "A challenging problem emerged.", "The challenging situation required creativity."] },
      { word: "demanding", pos: "adj", domain: "mental", definition: "requiring much skill or effort", examples: ["The demanding boss expected perfection.", "A demanding schedule left no time.", "The demanding role tested her skills."] },
      { word: "rigorous", pos: "adj", domain: "mental", definition: "extremely thorough and demanding", examples: ["The rigorous training prepared them.", "A rigorous analysis was conducted.", "The rigorous standards ensured quality."] },
      { word: "grueling", pos: "adj", domain: "physical", definition: "extremely tiring and demanding", examples: ["The grueling marathon tested endurance.", "A grueling schedule wore them down.", "The grueling work finally ended."] },
    ],
    "easy": [
      { word: "effortless", pos: "adj", domain: "mental", definition: "requiring no effort", examples: ["The effortless grace impressed judges.", "An effortless victory was achieved.", "She made it look effortless."] },
      { word: "straightforward", pos: "adj", domain: "mental", definition: "uncomplicated and easy to understand", examples: ["The straightforward instructions helped.", "A straightforward approach worked best.", "The straightforward solution was obvious."] },
      { word: "facile", pos: "adj", domain: "mental", definition: "easily achieved", examples: ["The facile explanation oversimplified.", "A facile solution emerged.", "The facile assumption proved wrong."] },
      { word: "elementary", pos: "adj", domain: "mental", definition: "basic and simple", examples: ["The elementary concepts were clear.", "An elementary mistake was made.", "The elementary level suited beginners."] },
    ],
    "bright": [
      { word: "luminous", pos: "adj", domain: "light", definition: "full of or shedding light", examples: ["The luminous stars filled the sky.", "A luminous glow surrounded her.", "The luminous display attracted attention."] },
      { word: "radiant", pos: "adj", domain: "light", definition: "sending out light, shining", examples: ["The radiant sun warmed the earth.", "Her radiant smile lit up the room.", "The radiant colors dazzled visitors."] },
      { word: "brilliant", pos: "adj", domain: "light", definition: "very bright and intense", examples: ["The brilliant light blinded them.", "A brilliant flash illuminated the sky.", "The brilliant diamond sparkled."] },
      { word: "resplendent", pos: "adj", domain: "light", definition: "impressive through brightness", examples: ["The resplendent palace amazed tourists.", "She looked resplendent in her gown.", "The resplendent sunset painted the sky."] },
    ],
    "dark": [
      { word: "murky", pos: "adj", domain: "light", definition: "dark and gloomy", examples: ["The murky water hid dangers.", "A murky past haunted him.", "The murky depths were unexplored."] },
      { word: "shadowy", pos: "adj", domain: "light", definition: "full of shadows", examples: ["The shadowy figure disappeared.", "A shadowy alley led nowhere.", "The shadowy room felt oppressive."] },
      { word: "somber", pos: "adj", domain: "light", definition: "dark or dull in color", examples: ["The somber clouds gathered.", "A somber mood pervaded.", "The somber occasion called for silence."] },
      { word: "tenebrous", pos: "adj", domain: "light", definition: "dark, shadowy", examples: ["The tenebrous forest was forbidding.", "A tenebrous atmosphere surrounded the castle.", "The tenebrous corners hid secrets."] },
    ],
    "walk": [
      { word: "saunter", pos: "verb", domain: "movement", definition: "walk in a slow, relaxed manner", examples: ["He sauntered into the room.", "She sauntered along the beach.", "They sauntered through the park."] },
      { word: "stroll", pos: "verb", domain: "movement", definition: "walk in a leisurely way", examples: ["They strolled through the garden.", "She strolled down the street.", "We strolled along the riverbank."] },
      { word: "amble", pos: "verb", domain: "movement", definition: "walk at a slow, relaxed pace", examples: ["The horse ambled along the trail.", "He ambled through the market.", "She ambled towards the exit."] },
      { word: "trudge", pos: "verb", domain: "movement", definition: "walk slowly with heavy steps", examples: ["They trudged through the snow.", "He trudged up the hill.", "She trudged home after work."] },
    ],
    "run": [
      { word: "sprint", pos: "verb", domain: "movement", definition: "run at full speed", examples: ["He sprinted to catch the bus.", "She sprinted across the finish line.", "They sprinted to escape the rain."] },
      { word: "dash", pos: "verb", domain: "movement", definition: "run or move quickly", examples: ["She dashed to the store.", "He dashed across the street.", "They dashed to meet the deadline."] },
      { word: "bolt", pos: "verb", domain: "movement", definition: "move or run away suddenly", examples: ["The horse bolted from the stable.", "He bolted from the room.", "She bolted when she saw danger."] },
      { word: "scurry", pos: "verb", domain: "movement", definition: "move hurriedly with short quick steps", examples: ["The mice scurried across the floor.", "People scurried for shelter.", "She scurried to finish on time."] },
    ],
    "think": [
      { word: "contemplate", pos: "verb", domain: "mental", definition: "look thoughtfully at for a long time", examples: ["She contemplated her next move.", "He contemplated the meaning of life.", "They contemplated the offer carefully."] },
      { word: "ponder", pos: "verb", domain: "mental", definition: "think about something carefully", examples: ["He pondered the question.", "She pondered her options.", "They pondered the implications."] },
      { word: "deliberate", pos: "verb", domain: "mental", definition: "engage in careful thought", examples: ["The jury deliberated for hours.", "She deliberated before deciding.", "They deliberated on the matter."] },
      { word: "ruminate", pos: "verb", domain: "mental", definition: "think deeply about something", examples: ["He ruminated on past mistakes.", "She ruminated over the decision.", "They ruminated about the future."] },
      { word: "cogitate", pos: "verb", domain: "mental", definition: "think deeply about something", examples: ["He cogitated on the problem.", "She cogitated before answering.", "They cogitated for days."] },
    ],
    "understand": [
      { word: "comprehend", pos: "verb", domain: "mental", definition: "grasp mentally", examples: ["She couldn't comprehend the concept.", "He finally comprehended the instructions.", "They struggled to comprehend the situation."] },
      { word: "fathom", pos: "verb", domain: "mental", definition: "understand after much thought", examples: ["He couldn't fathom her motives.", "She fathomed the mystery.", "They couldn't fathom the depth of the problem."] },
      { word: "discern", pos: "verb", domain: "mental", definition: "perceive or recognize", examples: ["She discerned a pattern.", "He discerned the truth.", "They discerned the hidden meaning."] },
      { word: "grasp", pos: "verb", domain: "mental", definition: "seize and hold firmly; understand", examples: ["He grasped the concept quickly.", "She grasped the opportunity.", "They grasped the significance."] },
    ],
    "strong": [
      { word: "robust", pos: "adj", domain: "physical", definition: "strong and healthy", examples: ["The robust economy grew steadily.", "A robust constitution helped him recover.", "The robust flavor pleased everyone."] },
      { word: "formidable", pos: "adj", domain: "physical", definition: "inspiring fear or respect through strength", examples: ["The formidable warrior defeated all.", "A formidable presence commanded attention.", "The formidable defense held firm."] },
      { word: "stalwart", pos: "adj", domain: "physical", definition: "loyal, reliable, and hardworking", examples: ["The stalwart supporter never wavered.", "A stalwart defender protected them.", "The stalwart team member contributed greatly."] },
      { word: "sturdy", pos: "adj", domain: "physical", definition: "strongly built", examples: ["The sturdy bridge held the weight.", "A sturdy frame supported the structure.", "The sturdy table lasted years."] },
    ],
    "weak": [
      { word: "frail", pos: "adj", domain: "physical", definition: "weak and delicate", examples: ["The frail patient needed care.", "A frail voice called out.", "The frail structure collapsed."] },
      { word: "feeble", pos: "adj", domain: "physical", definition: "lacking physical strength", examples: ["The feeble attempt failed.", "A feeble excuse was given.", "The feeble light flickered."] },
      { word: "fragile", pos: "adj", domain: "physical", definition: "easily broken or damaged", examples: ["The fragile vase was handled carefully.", "A fragile ecosystem needed protection.", "The fragile truce held briefly."] },
      { word: "debilitated", pos: "adj", domain: "physical", definition: "made weak or infirm", examples: ["The debilitated patient rested.", "A debilitated economy struggled.", "The debilitated army retreated."] },
    ],
    "true": [
      { word: "authentic", pos: "adj", domain: "quality", definition: "genuine, not false", examples: ["The authentic document was verified.", "An authentic experience awaited.", "The authentic flavor was unmistakable."] },
      { word: "genuine", pos: "adj", domain: "quality", definition: "truly what it is said to be", examples: ["The genuine concern was appreciated.", "A genuine smile crossed her face.", "The genuine article was rare."] },
      { word: "veritable", pos: "adj", domain: "quality", definition: "being truly so called", examples: ["A veritable feast awaited them.", "The veritable treasure was found.", "She was a veritable expert."] },
      { word: "bona fide", pos: "adj", domain: "quality", definition: "genuine, real", examples: ["A bona fide offer was made.", "The bona fide credentials were checked.", "The bona fide hero was honored."] },
    ],
    "false": [
      { word: "spurious", pos: "adj", domain: "quality", definition: "not genuine, false", examples: ["The spurious claim was rejected.", "A spurious argument was made.", "The spurious evidence was dismissed."] },
      { word: "fraudulent", pos: "adj", domain: "quality", definition: "obtained by deception", examples: ["The fraudulent scheme was exposed.", "A fraudulent transaction occurred.", "The fraudulent documents were confiscated."] },
      { word: "counterfeit", pos: "adj", domain: "quality", definition: "made in exact imitation with intent to deceive", examples: ["The counterfeit bills were detected.", "A counterfeit product was sold.", "The counterfeit signature was obvious."] },
      { word: "bogus", pos: "adj", domain: "quality", definition: "not genuine, fake", examples: ["The bogus story was disproven.", "A bogus claim was filed.", "The bogus credentials were discovered."] },
    ],
    "old": [
      { word: "ancient", pos: "adj", domain: "temporal", definition: "belonging to the very distant past", examples: ["The ancient ruins attracted tourists.", "An ancient tradition continued.", "The ancient civilization left artifacts."] },
      { word: "antiquated", pos: "adj", domain: "temporal", definition: "old-fashioned or outdated", examples: ["The antiquated system needed updating.", "An antiquated law was reformed.", "The antiquated equipment was replaced."] },
      { word: "archaic", pos: "adj", domain: "temporal", definition: "very old or old-fashioned", examples: ["The archaic language was difficult.", "An archaic custom persisted.", "The archaic methods were abandoned."] },
      { word: "venerable", pos: "adj", domain: "temporal", definition: "accorded great respect because of age", examples: ["The venerable institution celebrated.", "A venerable leader was honored.", "The venerable tradition continued."] },
    ],
    "new": [
      { word: "novel", pos: "adj", domain: "temporal", definition: "new and original", examples: ["The novel approach surprised everyone.", "A novel idea emerged.", "The novel technique was patented."] },
      { word: "innovative", pos: "adj", domain: "temporal", definition: "introducing new ideas", examples: ["The innovative design won awards.", "An innovative solution was found.", "The innovative company led the industry."] },
      { word: "nascent", pos: "adj", domain: "temporal", definition: "just beginning to develop", examples: ["The nascent movement gained followers.", "A nascent industry emerged.", "The nascent idea needed development."] },
      { word: "unprecedented", pos: "adj", domain: "temporal", definition: "never done or known before", examples: ["The unprecedented event shocked everyone.", "An unprecedented opportunity arose.", "The unprecedented growth continued."] },
    ],
    "beautiful": [
      { word: "exquisite", pos: "adj", domain: "quality", definition: "extremely beautiful and delicate", examples: ["The exquisite jewelry sparkled.", "An exquisite meal was served.", "The exquisite detail impressed everyone."] },
      { word: "stunning", pos: "adj", domain: "quality", definition: "extremely impressive or attractive", examples: ["The stunning view amazed visitors.", "A stunning performance captivated audiences.", "The stunning revelation shocked them."] },
      { word: "gorgeous", pos: "adj", domain: "quality", definition: "beautiful, very attractive", examples: ["The gorgeous sunset painted the sky.", "A gorgeous dress caught her eye.", "The gorgeous scenery inspired artists."] },
      { word: "ravishing", pos: "adj", domain: "quality", definition: "delightful, entrancing", examples: ["The ravishing beauty enchanted all.", "A ravishing display amazed viewers.", "The ravishing melody soothed them."] },
    ],
    "ugly": [
      { word: "grotesque", pos: "adj", domain: "quality", definition: "comically or repulsively ugly", examples: ["The grotesque statue stood alone.", "A grotesque figure emerged.", "The grotesque scene disturbed viewers."] },
      { word: "hideous", pos: "adj", domain: "quality", definition: "ugly or disgusting to look at", examples: ["The hideous creature appeared.", "A hideous mistake was made.", "The hideous wallpaper was removed."] },
      { word: "repulsive", pos: "adj", domain: "quality", definition: "arousing intense distaste", examples: ["The repulsive smell drove them away.", "A repulsive act was committed.", "The repulsive behavior was condemned."] },
      { word: "unsightly", pos: "adj", domain: "quality", definition: "unpleasant to look at", examples: ["The unsightly building was demolished.", "An unsightly stain remained.", "The unsightly mess was cleaned."] },
    ],
    "strange": [
      { word: "peculiar", pos: "adj", domain: "quality", definition: "strange or odd", examples: ["A peculiar smell filled the room.", "The peculiar behavior raised questions.", "She had a peculiar way of speaking."] },
      { word: "bizarre", pos: "adj", domain: "quality", definition: "very strange or unusual", examples: ["The bizarre incident puzzled everyone.", "A bizarre coincidence occurred.", "The bizarre story was hard to believe."] },
      { word: "anomalous", pos: "adj", domain: "quality", definition: "deviating from what is standard", examples: ["The anomalous data was investigated.", "An anomalous result emerged.", "The anomalous situation required attention."] },
      { word: "eccentric", pos: "adj", domain: "quality", definition: "unconventional and slightly strange", examples: ["The eccentric artist lived alone.", "An eccentric habit annoyed others.", "The eccentric millionaire donated generously."] },
    ],
    "important": [
      { word: "crucial", pos: "adj", domain: "quality", definition: "of great importance", examples: ["The crucial decision was made.", "A crucial moment arrived.", "The crucial evidence was presented."] },
      { word: "pivotal", pos: "adj", domain: "quality", definition: "of crucial importance", examples: ["The pivotal role was assigned.", "A pivotal moment changed everything.", "The pivotal meeting decided the outcome."] },
      { word: "paramount", pos: "adj", domain: "quality", definition: "more important than anything else", examples: ["Safety is paramount.", "A paramount concern was addressed.", "The paramount issue was resolved."] },
      { word: "vital", pos: "adj", domain: "quality", definition: "absolutely necessary", examples: ["Vital information was shared.", "A vital component was missing.", "The vital signs were stable."] },
    ],
    "clear": [
      { word: "lucid", pos: "adj", domain: "mental", definition: "expressed clearly, easy to understand", examples: ["The lucid explanation helped.", "A lucid moment came to him.", "The lucid writing impressed readers."] },
      { word: "transparent", pos: "adj", domain: "quality", definition: "easy to perceive or detect", examples: ["The transparent motives were obvious.", "A transparent process was established.", "The transparent material allowed light through."] },
      { word: "explicit", pos: "adj", domain: "quality", definition: "stated clearly and in detail", examples: ["The explicit instructions were helpful.", "An explicit warning was given.", "The explicit content was restricted."] },
      { word: "unambiguous", pos: "adj", domain: "quality", definition: "not open to more than one interpretation", examples: ["The unambiguous message was received.", "An unambiguous statement was made.", "The unambiguous results confirmed the theory."] },
    ],
    "secret": [
      { word: "clandestine", pos: "adj", domain: "quality", definition: "kept secret or done secretively", examples: ["The clandestine meeting was held.", "A clandestine operation was conducted.", "The clandestine affair was discovered."] },
      { word: "covert", pos: "adj", domain: "quality", definition: "not openly acknowledged", examples: ["The covert mission succeeded.", "A covert glance was exchanged.", "The covert actions were revealed."] },
      { word: "surreptitious", pos: "adj", domain: "quality", definition: "kept secret because disapproved of", examples: ["The surreptitious entry was detected.", "A surreptitious look was cast.", "The surreptitious recording was illegal."] },
      { word: "furtive", pos: "adj", domain: "quality", definition: "attempting to avoid notice", examples: ["The furtive glance betrayed him.", "A furtive movement was spotted.", "The furtive behavior aroused suspicion."] },
    ],
  };

  // Antonym pairs
  AntonymPairs = [
    ["happy", "sad"], ["happy", "miserable"], ["happy", "unhappy"],
    ["elated", "despondent"], ["jubilant", "morose"],
    ["ecstatic", "melancholy"], ["euphoric", "forlorn"],
    ["hot", "cold"], ["scalding", "frigid"], ["sweltering", "glacial"],
    ["torrid", "arctic"], ["scorching", "frosty"],
    ["big", "small"], ["enormous", "minuscule"], ["colossal", "diminutive"],
    ["immense", "microscopic"], ["mammoth", "infinitesimal"], ["gargantuan", "tiny"],
    ["fast", "slow"], ["rapid", "sluggish"], ["swift", "lethargic"],
    ["expeditious", "leisurely"], ["brisk", "gradual"],
    ["good", "bad"], ["excellent", "terrible"], ["superb", "atrocious"],
    ["exemplary", "deplorable"], ["outstanding", "abysmal"], ["impeccable", "dreadful"],
    ["smart", "stupid"], ["astute", "obtuse"], ["intelligent", "foolish"],
    ["beautiful", "ugly"], ["gorgeous", "hideous"], ["exquisite", "grotesque"],
    ["stunning", "repulsive"], ["ravishing", "unsightly"],
    ["strong", "weak"], ["robust", "frail"], ["stalwart", "feeble"],
    ["sturdy", "fragile"], ["formidable", "debilitated"],
    ["easy", "difficult"], ["effortless", "arduous"], ["straightforward", "formidable"],
    ["facile", "onerous"], ["elementary", "laborious"],
    ["rich", "poor"], ["affluent", "destitute"], ["wealthy", "impoverished"],
    ["true", "false"], ["authentic", "fake"], ["genuine", "spurious"],
    ["veritable", "fraudulent"], ["bona fide", "bogus"],
    ["bright", "dark"], ["luminous", "murky"], ["radiant", "shadowy"],
    ["brilliant", "somber"], ["resplendent", "tenebrous"],
    ["old", "young"], ["ancient", "modern"], ["antiquated", "contemporary"],
    ["archaic", "new"], ["venerable", "nascent"],
    ["success", "failure"], ["friend", "enemy"], ["love", "hate"],
    ["accept", "reject"], ["increase", "decrease"],
    ["open", "closed"], ["visible", "invisible"],
  ];

  // Idiom database
  IdiomDatabase = {
    "hot": [
      { phrase: "hot dog", meaning: "food" },
      { phrase: "hot water", meaning: "trouble" },
      { phrase: "hot air", meaning: "empty talk" },
      { phrase: "hot shot", meaning: "important person" },
      { phrase: "hot potato", meaning: "controversial issue" },
      { phrase: "hot head", meaning: "quick temper" },
    ],
    "cold": [
      { phrase: "cold feet", meaning: "nervousness" },
      { phrase: "cold shoulder", meaning: "ignore" },
      { phrase: "cold turkey", meaning: "stop abruptly" },
      { phrase: "cold blood", meaning: "cruelty/calm" },
      { phrase: "cold war", meaning: "tension" },
    ],
    "big": [
      { phrase: "big deal", meaning: "important/sarcasm" },
      { phrase: "big shot", meaning: "important person" },
      { phrase: "big picture", meaning: "overall view" },
    ],
    "small": [
      { phrase: "small talk", meaning: "casual conversation" },
      { phrase: "small fry", meaning: "unimportant" },
    ],
    "good": [
      { phrase: "good riddance", meaning: "relief at departure" },
      { phrase: "good grief", meaning: "exclamation" },
    ],
    "bad": [
      { phrase: "bad blood", meaning: "hostility" },
      { phrase: "bad apple", meaning: "troublemaker" },
    ],
    "happy": [
      { phrase: "happy medium", meaning: "compromise" },
      { phrase: "happy go lucky", meaning: "carefree" },
    ],
    "fast": [
      { phrase: "fast track", meaning: "accelerated path" },
      { phrase: "fast food", meaning: "quick service food" },
      { phrase: "fast asleep", meaning: "deeply sleeping" },
    ],
    "dark": [
      { phrase: "dark horse", meaning: "unknown competitor" },
      { phrase: "in the dark", meaning: "uninformed" },
    ],
    "old": [
      { phrase: "old hat", meaning: "outdated" },
      { phrase: "old school", meaning: "traditional" },
    ],
    "new": [
      { phrase: "new blood", meaning: "fresh members" },
      { phrase: "brand new", meaning: "completely new" },
    ],
  };

  // Proper noun patterns
  ProperNounPatterns = [
    "hot springs", "cold spring", "cold war", "new york", "new jersey",
    "new orleans", "new hampshire", "new mexico", "new zealand", "new england",
    "great britain", "great lakes", "big apple", "big ben", "red sea",
    "white house", "black sea", "dead sea", "good friday", "old testament", "new testament",
  ];

  TitlePatterns = [
    "president", "senator", "governor", "mayor", "judge",
    "doctor", "professor", "reverend", "mr", "mrs", "ms", "miss", "dr", "prof",
    "sir", "lord", "lady", "king", "queen", "prince", "princess",
    "general", "colonel", "captain", "lieutenant",
  ];

  // Negation words
  NegatorWords = new Set([
    "not", "never", "no", "hardly", "barely", "scarcely",
    "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't",
    "can't", "couldn't", "shouldn't", "don't", "doesn't", "didn't",
    "haven't", "hasn't", "hadn't", "cannot", "without",
    "neither", "nor", "none", "nothing", "nowhere",
  ]);

  DiminisherWords = new Set([
    "slightly", "somewhat", "rather", "fairly",
    "mildly", "marginally", "partially",
  ]);

  IntensifierWords = new Set([
    "very", "extremely", "really", "quite", "so", "too",
    "incredibly", "absolutely", "completely", "totally",
    "utterly", "thoroughly", "highly", "deeply",
  ]);

  DATA_LOADED = true;
}

/**
 * Load DistilBERT pipelines using Transformers.js
 * Uses feature-extraction for embeddings, fill-mask for syntax scoring
 */
async function loadModel() {
  if (WorkerState.modelLoaded || WorkerState.modelLoading) return;

  WorkerState.modelLoading = true;

  try {
    // env is already configured at module load time (see imports section)
    // Additional runtime check for global ort object (set by ONNX Runtime)
    if (typeof self.ort !== 'undefined' && self.ort.env?.wasm) {
      self.ort.env.wasm.proxy = false;
      self.ort.env.wasm.numThreads = 1;
      console.log('[AI Worker] Runtime: configured ort.env.wasm.proxy = false');
    }

    // Report progress
    postMessage({ type: 'status', status: 'loading', message: 'Loading feature extraction model...' });

    // Load feature-extraction pipeline for embeddings (returns hidden states directly)
    // pipeline is imported at module level
    WorkerState.extractor = await pipeline('feature-extraction', 'Xenova/distilbert-base-uncased', {
      quantized: true,
    });

    postMessage({ type: 'status', status: 'loading', message: 'Loading fill-mask model...' });

    // Load fill-mask pipeline for syntax scoring
    WorkerState.fillMask = await pipeline('fill-mask', 'Xenova/distilbert-base-uncased', {
      quantized: true,
    });

    WorkerState.modelLoaded = true;
    WorkerState.modelLoading = false;

    postMessage({ type: 'status', status: 'ready', message: 'Models loaded successfully' });
    console.log('[AI Worker] Both pipelines loaded successfully');

  } catch (error) {
    WorkerState.modelLoading = false;
    postMessage({ type: 'error', message: `Failed to load model: ${error.message}` });
    throw error;
  }
}

/**
 * Pre-compute embeddings for all vocabulary words
 */
async function precomputeVocabulary() {
  if (!WorkerState.modelLoaded) return;

  postMessage({ type: 'status', status: 'caching', message: 'Pre-computing embeddings...' });

  const allWords = new Set();

  // Collect all words from vocabulary
  for (const [simpleWord, candidates] of Object.entries(GREVocabularyDatabase)) {
    allWords.add(simpleWord);
    for (const candidate of candidates) {
      allWords.add(candidate.word);
    }
  }

  // Add custom vocabulary words
  for (const [word, candidates] of WorkerState.customVocabulary.entries()) {
    allWords.add(word);
    for (const candidate of candidates) {
      allWords.add(candidate.word);
    }
  }

  let processed = 0;
  const total = allWords.size;

  for (const word of allWords) {
    if (!WorkerState.embeddingCache.has(word.toLowerCase())) {
      await getEmbedding(word);
    }
    processed++;

    if (processed % 20 === 0) {
      postMessage({
        type: 'status',
        status: 'caching',
        message: `Pre-computing embeddings... ${Math.round(processed / total * 100)}%`
      });
    }
  }

  // Pre-compute context vectors from examples
  await precomputeContextVectors();

  postMessage({ type: 'status', status: 'ready', message: 'Embeddings cached' });
}

/**
 * Pre-compute context vectors from example sentences
 */
async function precomputeContextVectors() {
  for (const [simpleWord, candidates] of Object.entries(GREVocabularyDatabase)) {
    for (const candidate of candidates) {
      if (!WorkerState.contextCache.has(candidate.word.toLowerCase())) {
        const contextVectors = [];

        for (const example of candidate.examples.slice(0, 3)) {
          const vector = await getContextVector(example, candidate.word);
          if (vector) {
            contextVectors.push(vector);
          }
        }

        if (contextVectors.length > 0) {
          WorkerState.contextCache.set(candidate.word.toLowerCase(), contextVectors);
        }
      }
    }
  }
}

// ============================================================================
// EMBEDDING & SCORING FUNCTIONS
// ============================================================================

/**
 * Get embedding for a single word using feature-extraction pipeline
 */
async function getEmbedding(word) {
  const lower = word.toLowerCase();

  if (WorkerState.embeddingCache.has(lower)) {
    return WorkerState.embeddingCache.get(lower);
  }

  if (!WorkerState.modelLoaded || !WorkerState.extractor) {
    return null;
  }

  try {
    // Use feature-extraction pipeline - it returns embeddings directly
    // pooling: 'mean' averages all token embeddings into one vector
    const output = await WorkerState.extractor(word, { pooling: 'mean', normalize: true });

    // Output is a Tensor - extract data as Float32Array
    const embedding = new Float32Array(output.data);

    WorkerState.embeddingCache.set(lower, embedding);
    return embedding;

  } catch (error) {
    console.error(`Error getting embedding for "${word}":`, error);
    return null;
  }
}

/**
 * Get context vector for a sentence using feature-extraction pipeline
 * Returns mean-pooled embedding of the entire sentence for context comparison
 */
async function getContextVector(sentence, targetWord) {
  if (!WorkerState.modelLoaded || !WorkerState.extractor) {
    return null;
  }

  try {
    // Use feature-extraction on the full sentence
    // This captures contextual meaning better than extracting specific token positions
    const output = await WorkerState.extractor(sentence, { pooling: 'mean', normalize: true });

    return new Float32Array(output.data);

  } catch (error) {
    console.error(`Error getting context vector:`, error);
    return null;
  }
}

/**
 * Compute cosine similarity between two embeddings
 */
function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) {
    return 0;
  }

  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }

  return dot; // Vectors are already normalized
}

/**
 * Score syntax fitness using fill-mask pipeline
 */
async function scoreSyntax(maskedSentence, candidate) {
  if (!WorkerState.modelLoaded || !WorkerState.fillMask) {
    return -10;
  }

  try {
    // fill-mask pipeline returns array of predictions with scores
    const predictions = await WorkerState.fillMask(maskedSentence, { topk: 100 });

    // Find the candidate in predictions and return its score
    const candidateLower = candidate.toLowerCase();

    for (const pred of predictions) {
      // Check if prediction matches our candidate (case-insensitive)
      if (pred.token_str && pred.token_str.toLowerCase().trim() === candidateLower) {
        // Return log of score (to match expected range)
        return Math.log(pred.score + 1e-10);
      }
    }

    // Candidate not in top 100 predictions - low score
    return -10;

  } catch (error) {
    console.error(`Error scoring syntax:`, error);
    return -10;
  }
}

// ============================================================================
// LAYER 0: FAST CHECKS (Non-AI)
// ============================================================================

/**
 * Check if two words are antonyms
 */
function isAntonym(word1, word2) {
  const w1 = word1.toLowerCase();
  const w2 = word2.toLowerCase();

  for (const pair of AntonymPairs) {
    if ((pair[0] === w1 && pair[1] === w2) ||
        (pair[0] === w2 && pair[1] === w1)) {
      return true;
    }
  }
  return false;
}

/**
 * Check if word is part of an idiom
 */
function checkIdiom(sentence, targetWord) {
  const lower = targetWord.toLowerCase();
  const sentenceLower = sentence.toLowerCase();

  if (!IdiomDatabase[lower]) {
    return { isIdiom: false, meaning: null };
  }

  for (const idiom of IdiomDatabase[lower]) {
    if (sentenceLower.includes(idiom.phrase)) {
      return { isIdiom: true, meaning: idiom.meaning };
    }
  }

  return { isIdiom: false, meaning: null };
}

/**
 * Check if word is part of a proper noun
 */
function checkProperNoun(sentence, targetWord) {
  const lower = targetWord.toLowerCase();
  const sentenceLower = sentence.toLowerCase();

  // Check known proper nouns
  for (const pattern of ProperNounPatterns) {
    if (sentenceLower.includes(pattern) && pattern.includes(lower)) {
      return { isProperNoun: true, reason: `Part of "${pattern}"` };
    }
  }

  // Check title + capitalized word pattern
  for (const title of TitlePatterns) {
    const titleRegex = new RegExp(`\\b${title}\\.?\\s+([A-Z][a-z]+)`, 'i');
    const match = sentence.match(titleRegex);
    if (match) {
      const targetIdx = sentenceLower.indexOf(lower);
      const matchStart = sentenceLower.indexOf(match[0].toLowerCase());
      const matchEnd = matchStart + match[0].length;

      if (targetIdx >= matchStart && targetIdx < matchEnd) {
        return { isProperNoun: true, reason: "Title/name context" };
      }
    }
  }

  return { isProperNoun: false, reason: null };
}

/**
 * Check for negation context
 */
function checkNegation(sentence, targetWord) {
  const lower = targetWord.toLowerCase();
  const sentenceLower = sentence.toLowerCase();
  const targetIdx = sentenceLower.indexOf(lower);

  if (targetIdx === -1) {
    return { isNegated: false, expanded: targetWord };
  }

  // Get preceding text and split into words
  const precedingText = sentenceLower.substring(0, targetIdx).trim();
  const precedingWords = precedingText.split(/\s+/).slice(-4);

  // Check for negators and diminishers
  for (const word of precedingWords) {
    const cleaned = word.replace(/[.,!?;:'\"()-]/g, '');
    if (NegatorWords.has(cleaned) || DiminisherWords.has(cleaned)) {
      return { isNegated: true, expanded: targetWord };
    }
  }

  // Check for intensifiers (expand context)
  if (precedingWords.length > 0) {
    const lastWord = precedingWords[precedingWords.length - 1].replace(/[.,!?;:'\"()-]/g, '');
    if (IntensifierWords.has(lastWord)) {
      return { isNegated: false, expanded: `${lastWord} ${targetWord}` };
    }
  }

  return { isNegated: false, expanded: targetWord };
}

// ============================================================================
// MAIN PIPELINE: V8 Processing Logic
// ============================================================================

/**
 * Process a single substitution through the V8 pipeline
 */
async function processSubstitution(sentence, original, candidate) {
  const startTime = performance.now();

  // Check model readiness
  if (!WorkerState.modelLoaded) {
    return {
      original,
      candidate,
      passed: false,
      reason: BlockReason.MODEL_NOT_READY,
      similarity: 0,
      syntaxScore: 0,
      semanticScore: 0,
      timeMs: performance.now() - startTime,
    };
  }

  // LAYER 0: Antonym check
  if (isAntonym(original, candidate)) {
    return {
      original,
      candidate,
      passed: false,
      reason: BlockReason.ANTONYM_DETECTED,
      similarity: 0,
      syntaxScore: 0,
      semanticScore: 0,
      timeMs: performance.now() - startTime,
    };
  }

  // LAYER 1: Embedding similarity
  const embOriginal = await getEmbedding(original);
  const embCandidate = await getEmbedding(candidate);
  const similarity = cosineSimilarity(embOriginal, embCandidate);

  if (similarity < CONFIG.EMBEDDING_MIN) {
    return {
      original,
      candidate,
      passed: false,
      reason: BlockReason.NOT_SIMILAR_ENOUGH,
      similarity,
      syntaxScore: 0,
      semanticScore: 0,
      timeMs: performance.now() - startTime,
    };
  }

  if (similarity > CONFIG.EMBEDDING_MAX) {
    return {
      original,
      candidate,
      passed: false,
      reason: BlockReason.TOO_SIMILAR,
      similarity,
      syntaxScore: 0,
      semanticScore: 0,
      timeMs: performance.now() - startTime,
    };
  }

  // LAYER 2: Proper noun check
  const properNounResult = checkProperNoun(sentence, original);
  if (properNounResult.isProperNoun) {
    return {
      original,
      candidate,
      passed: false,
      reason: BlockReason.PROPER_NOUN,
      similarity,
      syntaxScore: 0,
      semanticScore: 0,
      timeMs: performance.now() - startTime,
    };
  }

  // LAYER 3: Idiom check
  const idiomResult = checkIdiom(sentence, original);
  if (idiomResult.isIdiom) {
    return {
      original,
      candidate,
      passed: false,
      reason: BlockReason.IDIOM_DETECTED,
      similarity,
      syntaxScore: 0,
      semanticScore: 0,
      timeMs: performance.now() - startTime,
    };
  }

  // LAYER 4: Negation check
  const negationResult = checkNegation(sentence, original);
  if (negationResult.isNegated) {
    return {
      original,
      candidate,
      passed: false,
      reason: BlockReason.NEGATION_CONTEXT,
      similarity,
      syntaxScore: 0,
      semanticScore: 0,
      timeMs: performance.now() - startTime,
    };
  }

  // LAYER 5: Contextual validation
  const originalContextVector = await getContextVector(sentence, negationResult.expanded);
  if (!originalContextVector) {
    return {
      original,
      candidate,
      passed: false,
      reason: BlockReason.NOT_IN_VOCAB,
      similarity,
      syntaxScore: 0,
      semanticScore: 0,
      timeMs: performance.now() - startTime,
    };
  }

  // Create masked sentence for syntax scoring
  const maskedSentence = sentence.replace(
    new RegExp(`\\b${escapeRegExp(negationResult.expanded)}\\b`, 'i'),
    '[MASK]'
  );

  // LAYER 6: Syntax scoring
  const syntaxScore = await scoreSyntax(maskedSentence, candidate);

  // LAYER 7: Semantic scoring
  let semanticScore = 0;

  // Try cached context vectors
  const cachedVectors = WorkerState.contextCache.get(candidate.toLowerCase());
  if (cachedVectors && cachedVectors.length > 0) {
    semanticScore = Math.max(
      ...cachedVectors.map(v => cosineSimilarity(originalContextVector, v))
    );
  }

  // Hybrid scoring: trust high embedding similarity
  if (similarity >= CONFIG.EMBEDDING_TRUST_THRESHOLD) {
    semanticScore = Math.max(semanticScore, similarity);
  }

  // FINAL VERDICT
  let passed = false;
  let reason = BlockReason.SYNTAX_FAILED;

  if (syntaxScore > CONFIG.SYNTAX_FLOOR) {
    if (semanticScore > CONFIG.SEMANTIC_FLOOR) {
      passed = true;
      reason = BlockReason.PASSED;
    } else {
      reason = BlockReason.SEMANTIC_FAILED;
    }
  } else {
    // Syntax failed, check semantic override
    if (semanticScore > CONFIG.SEMANTIC_OVERRIDE) {
      passed = true;
      reason = BlockReason.PASSED;
    }
  }

  return {
    original,
    candidate,
    passed,
    reason,
    similarity,
    syntaxScore,
    semanticScore,
    timeMs: performance.now() - startTime,
  };
}

/**
 * Find the best substitution for a word in context
 */
async function findBestSubstitution(sentence, word) {
  const lower = word.toLowerCase();

  // Get candidates from vocabulary
  let candidates = GREVocabularyDatabase[lower] || [];

  // Also check custom vocabulary
  if (WorkerState.customVocabulary.has(lower)) {
    candidates = [...candidates, ...WorkerState.customVocabulary.get(lower)];
  }

  if (candidates.length === 0) {
    return null;
  }

  const results = [];

  for (const candidate of candidates) {
    const result = await processSubstitution(sentence, word, candidate.word);
    if (result.passed) {
      results.push({
        ...result,
        candidateInfo: candidate,
      });
    }
  }

  if (results.length === 0) {
    return null;
  }

  // Select best by syntax score
  results.sort((a, b) => b.syntaxScore - a.syntaxScore);
  return results[0];
}

/**
 * Process entire text (paragraph mode)
 */
async function processText(text, maxDensity = CONFIG.MAX_DENSITY) {
  const startTime = performance.now();

  // Split into words
  const words = text.split(/\s+/);
  const maxSubs = Math.max(1, Math.ceil(words.length * maxDensity));

  // Limit processing to prevent crashes
  const wordsToProcess = words.slice(0, CONFIG.MAX_WORDS_PER_BATCH);

  // Find all potential substitutions
  const allResults = [];
  const processedWords = new Set();

  for (const word of wordsToProcess) {
    const clean = word.toLowerCase().replace(/[.,!?;:'\"()-]/g, '');

    if (processedWords.has(clean)) continue;
    if (clean.length < 2) continue;

    // Check if word exists in vocabulary
    if (GREVocabularyDatabase[clean] || WorkerState.customVocabulary.has(clean)) {
      const result = await findBestSubstitution(text, clean);
      if (result) {
        allResults.push(result);
        processedWords.add(clean);
      }
    }
  }

  // Sort by syntax score and take top N
  allResults.sort((a, b) => b.syntaxScore - a.syntaxScore);
  const selectedSubs = allResults.slice(0, maxSubs);

  // Apply substitutions
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
        semanticScore: sub.semanticScore,
      });
      modifiedText = newText;
    }
  }

  return {
    originalText: text,
    modifiedText,
    substitutionsAttempted: allResults.length,
    substitutionsMade: substitutions.length,
    substitutions,
    totalTimeMs: performance.now() - startTime,
  };
}

/**
 * Get vocabulary words that exist in text
 */
function findVocabularyWordsInText(text) {
  const words = text.toLowerCase().split(/\s+/);
  const found = [];

  for (const word of words) {
    const clean = word.replace(/[.,!?;:'\"()-]/g, '');
    if (GREVocabularyDatabase[clean] || WorkerState.customVocabulary.has(clean)) {
      found.push(clean);
    }
  }

  return [...new Set(found)];
}

// ============================================================================
// CUSTOM VOCABULARY MANAGEMENT
// ============================================================================

/**
 * Add custom vocabulary from CSV
 */
async function addCustomVocabulary(data) {
  // data should be array of { word, synonym, definition?, examples? }
  for (const entry of data) {
    const lower = entry.word.toLowerCase();

    if (!WorkerState.customVocabulary.has(lower)) {
      WorkerState.customVocabulary.set(lower, []);
    }

    WorkerState.customVocabulary.get(lower).push({
      word: entry.synonym,
      pos: entry.pos || 'unknown',
      domain: entry.domain || 'general',
      definition: entry.definition || '',
      examples: entry.examples || [],
    });

    // Pre-compute embedding for new word
    await getEmbedding(entry.synonym);
  }

  return { success: true, count: data.length };
}

/**
 * Clear custom vocabulary
 */
function clearCustomVocabulary() {
  WorkerState.customVocabulary.clear();
  return { success: true };
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// ============================================================================
// MESSAGE HANDLER
// ============================================================================

self.onmessage = async function(event) {
  const { type, id, data } = event.data;

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
          modelLoaded: WorkerState.modelLoaded,
          modelLoading: WorkerState.modelLoading,
          embeddingsCached: WorkerState.embeddingCache.size,
          contextsCached: WorkerState.contextCache.size,
          customVocabSize: WorkerState.customVocabulary.size,
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

      case 'addCustomVocabulary':
        result = await addCustomVocabulary(data.vocabulary);
        break;

      case 'clearCustomVocabulary':
        result = clearCustomVocabulary();
        break;

      case 'getVocabulary':
        result = {
          default: Object.keys(GREVocabularyDatabase),
          custom: Array.from(WorkerState.customVocabulary.keys()),
        };
        break;

      default:
        result = { error: `Unknown message type: ${type}` };
    }

    postMessage({ type: 'response', id, result });

  } catch (error) {
    postMessage({ type: 'error', id, error: error.message });
  }
};

// Auto-initialize on load
// Data is loaded via ES module import (data.js attaches to self)
loadDataFromSelf();
postMessage({ type: 'status', status: 'loaded', message: 'Worker loaded, awaiting initialization' });
