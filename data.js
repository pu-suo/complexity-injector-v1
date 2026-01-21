/**
 * Complexity Injector - Data Module
 * Ported from unified_complexifier_v2.py
 *
 * Contains: GREVocabularyDatabase, AntonymDatabase, IdiomDatabase,
 *           ProperNounDetector patterns, NegationDetector word lists
 */

// ============================================================================
// THRESHOLDS & CONFIGURATION (calibrated for quantized browser model)
// ============================================================================
const CONFIG = {
  // Embedding thresholds (relaxed for quantized DistilBERT)
  EMBEDDING_MIN: 0.50,           // Minimum similarity to be considered
  EMBEDDING_MAX: 0.99,           // Maximum similarity (too similar = no complexification)
  EMBEDDING_TRUST_THRESHOLD: 0.60,  // Use embedding as fallback for semantic score

  // Syntax & semantic scoring (relaxed for browser model)
  SYNTAX_FLOOR: -6.0,            // Minimum syntax score to pass
  SEMANTIC_FLOOR: 0.10,          // Minimum semantic score (low to accept basic synonyms)
  SEMANTIC_OVERRIDE: 0.80,       // If semantic > 0.80, override poor syntax

  // Paragraph processing
  MAX_DENSITY: 0.50,             // Max 50% of words can be substituted (for testing)
  MAX_WORDS_PER_BATCH: 1000,     // Limit processing to prevent crashes
};

// ============================================================================
// BLOCK REASONS (Enum equivalent)
// ============================================================================
const BlockReason = {
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

// ============================================================================
// GRE VOCABULARY DATABASE
// Schema: { simple_word: [{ word, pos, domain, definition, examples }] }
// ============================================================================
const GREVocabularyDatabase = {
  // Temperature
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

  // Size
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

  // Speed
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

  // Quality
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

  // Emotions - Happy
  "happy": [
    { word: "elated", pos: "adj", domain: "emotional", definition: "ecstatically happy", examples: ["She was elated by the news.", "The elated crowd cheered loudly.", "He felt elated after winning."] },
    { word: "jubilant", pos: "adj", domain: "emotional", definition: "feeling or expressing great happiness", examples: ["The jubilant fans celebrated.", "A jubilant atmosphere filled the room.", "They were jubilant over the victory."] },
    { word: "ecstatic", pos: "adj", domain: "emotional", definition: "feeling overwhelming happiness", examples: ["She was ecstatic about the promotion.", "The ecstatic response surprised everyone.", "He felt ecstatic seeing her again."] },
    { word: "euphoric", pos: "adj", domain: "emotional", definition: "intensely happy and confident", examples: ["The euphoric feeling lasted for days.", "A euphoric crowd filled the streets.", "She felt euphoric after the achievement."] },
    { word: "exuberant", pos: "adj", domain: "emotional", definition: "filled with lively energy and excitement", examples: ["The exuberant children played happily.", "His exuberant personality attracted friends.", "The exuberant celebration continued."] },
  ],

  // Emotions - Sad
  "sad": [
    { word: "melancholy", pos: "adj", domain: "emotional", definition: "a feeling of pensive sadness", examples: ["A melancholy mood settled over her.", "The melancholy music touched their hearts.", "He felt melancholy on rainy days."] },
    { word: "despondent", pos: "adj", domain: "emotional", definition: "in low spirits from loss of hope", examples: ["The despondent athlete gave up.", "She grew despondent after the rejection.", "The despondent patient needed support."] },
    { word: "morose", pos: "adj", domain: "emotional", definition: "sullen and ill-tempered", examples: ["His morose attitude worried his friends.", "The morose expression never left his face.", "She became morose after the loss."] },
    { word: "forlorn", pos: "adj", domain: "emotional", definition: "pitifully sad and lonely", examples: ["The forlorn puppy waited by the door.", "A forlorn hope remained.", "She looked forlorn sitting alone."] },
    { word: "woeful", pos: "adj", domain: "emotional", definition: "characterized by or full of sorrow", examples: ["The woeful tale brought tears.", "A woeful expression crossed her face.", "The woeful situation demanded attention."] },
  ],

  // Difficulty
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

  // Light
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

  // Movement
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

  // Cognition
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

  // Strength
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

  // Truth
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

  // Temporal
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

  // Appearance
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

  // Other qualities
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

// ============================================================================
// ANTONYM DATABASE
// Hand-curated antonym pairs (supplemental to prevent antonym substitution)
// ============================================================================
const AntonymPairs = [
  // Emotion pairs
  ["happy", "sad"], ["happy", "miserable"], ["happy", "unhappy"],
  ["elated", "despondent"], ["jubilant", "morose"],
  ["ecstatic", "melancholy"], ["euphoric", "forlorn"],

  // Temperature pairs
  ["hot", "cold"], ["scalding", "frigid"], ["sweltering", "glacial"],
  ["torrid", "arctic"], ["scorching", "frosty"],

  // Size pairs
  ["big", "small"], ["enormous", "minuscule"], ["colossal", "diminutive"],
  ["immense", "microscopic"], ["mammoth", "infinitesimal"], ["gargantuan", "tiny"],

  // Speed pairs
  ["fast", "slow"], ["rapid", "sluggish"], ["swift", "lethargic"],
  ["expeditious", "leisurely"], ["brisk", "gradual"],

  // Quality pairs
  ["good", "bad"], ["excellent", "terrible"], ["superb", "atrocious"],
  ["exemplary", "deplorable"], ["outstanding", "abysmal"], ["impeccable", "dreadful"],

  // Intelligence pairs
  ["smart", "stupid"], ["astute", "obtuse"], ["intelligent", "foolish"],

  // Appearance pairs
  ["beautiful", "ugly"], ["gorgeous", "hideous"], ["exquisite", "grotesque"],
  ["stunning", "repulsive"], ["ravishing", "unsightly"],

  // Strength pairs
  ["strong", "weak"], ["robust", "frail"], ["stalwart", "feeble"],
  ["sturdy", "fragile"], ["formidable", "debilitated"],

  // Difficulty pairs
  ["easy", "difficult"], ["effortless", "arduous"], ["straightforward", "formidable"],
  ["facile", "onerous"], ["elementary", "laborious"],

  // Wealth pairs
  ["rich", "poor"], ["affluent", "destitute"], ["wealthy", "impoverished"],

  // Truth pairs
  ["true", "false"], ["authentic", "fake"], ["genuine", "spurious"],
  ["veritable", "fraudulent"], ["bona fide", "bogus"],

  // Light pairs
  ["bright", "dark"], ["luminous", "murky"], ["radiant", "shadowy"],
  ["brilliant", "somber"], ["resplendent", "tenebrous"],

  // Age pairs
  ["old", "young"], ["ancient", "modern"], ["antiquated", "contemporary"],
  ["archaic", "new"], ["venerable", "nascent"],

  // Action pairs
  ["success", "failure"], ["friend", "enemy"], ["love", "hate"],
  ["accept", "reject"], ["increase", "decrease"],
  ["open", "closed"], ["visible", "invisible"],
];

// ============================================================================
// IDIOM DATABASE
// Patterns that should NOT be substituted
// ============================================================================
const IdiomDatabase = {
  "hot": [
    { phrase: "hot dog", meaning: "food" },
    { phrase: "hot water", meaning: "trouble" },
    { phrase: "hot air", meaning: "empty talk" },
    { phrase: "hot shot", meaning: "important person" },
    { phrase: "hot potato", meaning: "controversial issue" },
    { phrase: "hot head", meaning: "quick temper" },
    { phrase: "hot under the collar", meaning: "angry" },
  ],
  "cold": [
    { phrase: "cold feet", meaning: "nervousness" },
    { phrase: "cold shoulder", meaning: "ignore" },
    { phrase: "cold turkey", meaning: "stop abruptly" },
    { phrase: "cold blood", meaning: "cruelty/calm" },
    { phrase: "cold war", meaning: "tension without conflict" },
    { phrase: "cold fish", meaning: "unemotional person" },
  ],
  "big": [
    { phrase: "big deal", meaning: "important/sarcasm" },
    { phrase: "big shot", meaning: "important person" },
    { phrase: "big picture", meaning: "overall view" },
    { phrase: "big cheese", meaning: "important person" },
    { phrase: "big mouth", meaning: "talks too much" },
  ],
  "small": [
    { phrase: "small talk", meaning: "casual conversation" },
    { phrase: "small fry", meaning: "unimportant people" },
    { phrase: "small potatoes", meaning: "insignificant" },
    { phrase: "small world", meaning: "coincidence" },
  ],
  "good": [
    { phrase: "good riddance", meaning: "relief at departure" },
    { phrase: "good grief", meaning: "exclamation" },
    { phrase: "good for nothing", meaning: "useless" },
    { phrase: "good samaritan", meaning: "helpful stranger" },
  ],
  "bad": [
    { phrase: "bad blood", meaning: "hostility" },
    { phrase: "bad apple", meaning: "troublemaker" },
    { phrase: "bad egg", meaning: "dishonest person" },
    { phrase: "bad rap", meaning: "unfair criticism" },
  ],
  "happy": [
    { phrase: "happy medium", meaning: "compromise" },
    { phrase: "happy go lucky", meaning: "carefree" },
    { phrase: "happy camper", meaning: "satisfied person" },
  ],
  "fast": [
    { phrase: "fast track", meaning: "accelerated path" },
    { phrase: "fast food", meaning: "quick service food" },
    { phrase: "fast and loose", meaning: "irresponsible" },
    { phrase: "fast asleep", meaning: "deeply sleeping" },
  ],
  "slow": [
    { phrase: "slow poke", meaning: "slow person" },
    { phrase: "slow burn", meaning: "gradual anger" },
    { phrase: "slow motion", meaning: "reduced speed" },
  ],
  "run": [
    { phrase: "run of the mill", meaning: "ordinary" },
    { phrase: "run amok", meaning: "go wild" },
    { phrase: "run the gamut", meaning: "cover full range" },
  ],
  "walk": [
    { phrase: "walk of life", meaning: "occupation/position" },
    { phrase: "walk the walk", meaning: "act on words" },
    { phrase: "walk on eggshells", meaning: "be careful" },
  ],
  "dark": [
    { phrase: "dark horse", meaning: "unknown competitor" },
    { phrase: "dark side", meaning: "negative aspect" },
    { phrase: "in the dark", meaning: "uninformed" },
  ],
  "bright": [
    { phrase: "bright side", meaning: "positive aspect" },
    { phrase: "bright idea", meaning: "clever thought" },
    { phrase: "bright and early", meaning: "very early" },
  ],
  "old": [
    { phrase: "old hat", meaning: "outdated" },
    { phrase: "old flame", meaning: "former lover" },
    { phrase: "old school", meaning: "traditional" },
  ],
  "new": [
    { phrase: "new blood", meaning: "fresh members" },
    { phrase: "new leaf", meaning: "fresh start" },
    { phrase: "brand new", meaning: "completely new" },
  ],
};

// ============================================================================
// PROPER NOUN PATTERNS
// Known proper nouns that should NOT be substituted
// ============================================================================
const ProperNounPatterns = [
  "hot springs",
  "cold spring",
  "cold war",
  "new york",
  "new jersey",
  "new orleans",
  "new hampshire",
  "new mexico",
  "new zealand",
  "new england",
  "great britain",
  "great lakes",
  "big apple",
  "big ben",
  "red sea",
  "white house",
  "black sea",
  "dead sea",
  "good friday",
  "old testament",
  "new testament",
];

// Titles that indicate proper nouns when followed by capitalized words
const TitlePatterns = [
  "president", "senator", "governor", "mayor", "judge",
  "doctor", "professor", "reverend", "bishop", "cardinal",
  "mr", "mrs", "ms", "miss", "dr", "prof",
  "sir", "lord", "lady", "duke", "duchess",
  "king", "queen", "prince", "princess",
  "general", "colonel", "captain", "lieutenant",
  "chief", "director", "chairman", "ceo",
];

// ============================================================================
// NEGATION DETECTOR WORD LISTS
// ============================================================================
const NegatorWords = new Set([
  "not", "never", "no", "hardly", "barely", "scarcely",
  "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't",
  "can't", "couldn't", "shouldn't", "don't", "doesn't", "didn't",
  "haven't", "hasn't", "hadn't", "cannot", "without",
  "neither", "nor", "none", "nothing", "nowhere",
]);

const DiminisherWords = new Set([
  "slightly", "somewhat", "a bit", "a little",
  "kind of", "sort of", "rather", "fairly",
  "mildly", "marginally", "partially",
]);

const IntensifierWords = new Set([
  "very", "extremely", "really", "quite", "so", "too",
  "incredibly", "absolutely", "completely", "totally",
  "utterly", "thoroughly", "highly", "deeply",
]);

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get all vocabulary words (simple words)
 */
function getVocabularyWords() {
  return Object.keys(GREVocabularyDatabase);
}

/**
 * Get substitution candidates for a word
 */
function getSubstitutions(word) {
  const lower = word.toLowerCase();
  return GREVocabularyDatabase[lower] || [];
}

/**
 * Check if a word pair are antonyms
 */
function isAntonymPair(word1, word2) {
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
 * Check if a word in context is part of an idiom
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
      const targetIdx = sentence.toLowerCase().indexOf(lower);
      const matchStart = sentence.toLowerCase().indexOf(match[0].toLowerCase());
      const matchEnd = matchStart + match[0].length;

      if (targetIdx >= matchStart && targetIdx < matchEnd) {
        return { isProperNoun: true, reason: "Title/name context" };
      }
    }
  }

  // Check mid-sentence capitalization
  const words = sentence.split(/\s+/);
  for (let i = 1; i < words.length; i++) {
    const word = words[i].replace(/[.,!?;:'\"()-]/g, '');
    if (word.toLowerCase() === lower && word[0] === word[0].toUpperCase()) {
      // Check if previous word ends with sentence terminator
      const prevWord = words[i - 1];
      if (!prevWord.match(/[.!?]$/)) {
        return { isProperNoun: true, reason: "Mid-sentence capitalization" };
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
  const precedingWords = precedingText.split(/\s+/).slice(-4); // Last 4 words

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
// ES MODULE EXPORTS
// ============================================================================
export {
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
  getVocabularyWords,
  getSubstitutions,
  isAntonymPair,
  checkIdiom,
  checkProperNoun,
  checkNegation,
};
