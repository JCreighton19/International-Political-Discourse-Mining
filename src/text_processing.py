"""
Shared text-processing configuration for the UNGDC project.

Every notebook that cleans or tokenizes speeches imports from here, so the
stopword list, normalization rules, and tokenizer cannot drift apart between
notebooks.

Pipeline (applied to each speech):
    raw text
      -> normalize_text()      Unicode/quote cleanup, contractions, possessives,
                               hyphenated-compound repair, spelling variants
      -> split_sentences()     sentence boundaries (phrases must not span them)
      -> tokenize()            lowercase word tokens, stopwords preserved
"""

import re
import unicodedata
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

# Negations change the meaning of a statement, so they are kept as tokens.
NEGATIONS = {"no", "nor", "not", "never", "neither"}

# Generic diplomatic/procedural vocabulary.
PROCEDURAL_STOPWORDS = {
    "mr", "mrs", "madam", "chairman", "chairperson",
    "delegation", "delegate", "delegates", "session", "plenary",
}

# Ceremonial vocabulary that carries little thematic content.
CEREMONIAL_STOPWORDS = {
    "congratulations", "congratulate", "pleasure", "appreciation",
    "behalf", "occasion", "thanks", "wish",
}

# Words in scikit-learn's list that are content words in this corpus.
#   third    -> "third world"           system   -> "international system"
#   interest -> "national interest"     un       -> "UN" (abbreviation)
#   serious  -> "serious concern"       fire     -> "open fire", "cease fire"
# Note: sklearn removes "interest" but not "interests", which would otherwise
# split one concept inconsistently.
RETAINED_CONTENT_WORDS = {"third", "system", "interest", "un", "serious", "fire"}

# Institutional terms explicitly retained (they form phrases such as
# "general assembly", "secretary general", "president of the assembly").
RETAINED_INSTITUTIONAL = {"assembly", "secretary", "president"}

STOPWORDS = (
    (set(ENGLISH_STOP_WORDS) - NEGATIONS - RETAINED_CONTENT_WORDS)
    | PROCEDURAL_STOPWORDS
    | CEREMONIAL_STOPWORDS
) - RETAINED_INSTITUTIONAL

# A learned phrase may not start or end with one of these words
# (rejects "has been", "my country", "not only", "would like", ...).
PHRASE_EDGE_STOPWORDS = STOPWORDS | NEGATIONS


def phrase_connector_words():
    """
    Connector words for gensim Phrases: gensim never lets a phrase start or
    end with a connector word but allows them in the middle ("peace and
    security", "rule of law"). Adding every stopword enforces the edge rule
    during phrase detection itself, so a grammatical phrase such as
    "session of the general" can never consume the "general" of
    "general assembly".
    """
    from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS
    return frozenset(ENGLISH_CONNECTOR_WORDS | PHRASE_EDGE_STOPWORDS)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

# Typographic quotes -> ASCII. U+FFFD (replacement character) occurs only in
# VAT_80_2025.txt, where every instance stands for a quote or apostrophe.
_QUOTE_TRANSLATION = str.maketrans({
    # single quotes / apostrophes / prime / acute accent / replacement char
    **{chr(c): "'" for c in (0x2018, 0x2019, 0x201A, 0x201B, 0x2032, 0x00B4,
                             0x0060, 0xFFFD)},
    # double quotes and guillemets
    **{chr(c): '"' for c in (0x201C, 0x201D, 0x201E, 0x00AB, 0x00BB)},
    # Unicode hyphen and non-breaking hyphen -> ASCII hyphen
    chr(0x2010): "-", chr(0x2011): "-",
    # soft hyphen -> removed
    chr(0x00AD): "",
})

# Contractions (expanded so negation is represented consistently as "not").
_CONTRACTIONS = [
    (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "cannot"),
    (r"\bshan't\b", "shall not"),
    (r"n't\b", " not"),
]

# Possessives: "organization's" -> "organization", "nations'" -> "nations".
_POSSESSIVES = [
    (r"'s\b", ""),
    (r"(?<=s)'(?=\W|$)", ""),
]

# Common abbreviations whose trailing period would otherwise be read as a
# sentence boundary.
_ABBREVIATIONS = re.compile(
    r"\b(mr|mrs|ms|messrs|dr|st|ste|no|nos|vol|art|arts|para|paras|gen|lt|col|"
    r"prof|sr|jr|rev|hon|amb|op|cf|vs|etc)\."
)

# Multi-word expressions protected before the hyphen-compound repair runs.
# A single underscore keeps them as one token.
_FIXED_EXPRESSIONS = [
    (r"\bvis[\s-]*[a\u00e0][\s-]*vis\b", "vis_a_vis"),
    (r"\bdar[\s-]+es[\s-]+salaam\b", "dar es salaam"),
    (r"\btwenty[\s-]+first\b", "twenty_first"),
]

# Prefixes that cannot stand alone as words; compounds built on them are
# always written closed (co-operation -> cooperation, non-aligned -> nonaligned,
# self-determination -> selfdetermination).
BOUND_PREFIXES = {
    "co", "non", "re", "pre", "sub", "inter", "intra", "multi", "under",
    "over", "counter", "anti", "socio", "neo", "bi", "de", "dis", "un",
    "super", "post", "trans", "supra", "macro", "micro", "geo", "hydro",
    "thermo", "radio", "bio", "mid", "semi", "pro", "extra", "ultra", "mis",
    "out", "self",
}

# Hyphenated pairs whose closed form is a *different* word in this corpus.
# These are never changed by the learned compound map.
COMPOUND_EXCLUSIONS = {
    ("vis", "a"), ("s", "vi"), ("dar", "es"),
    ("re", "sign"), ("re", "signed"), ("re", "form"), ("re", "formed"),
    ("re", "creation"), ("re", "present"), ("re", "sort"), ("re", "cover"),
    ("re", "count"), ("re", "act"), ("re", "treat"), ("re", "lay"),
    ("re", "mark"), ("re", "serve"), ("re", "solve"), ("re", "store"),
    ("pre", "sent"), ("pro", "test"), ("con", "tent"), ("in", "valid"),
}

# Orthographic variants -> one canonical form. The target is UN editorial
# (Oxford) spelling, which is the majority form in this corpus:
# -ize, -our, -re, defence, programme. "Viet Nam" and "per cent" become single
# tokens. Applied after hyphenated compounds are repaired.
_IZE_STEMS = (
    "organ|recogn|real|mobil|stabil|character|civil|global|criminal|minim|"
    "maxim|priorit|util|author|apolog|normal|democrat|militar|neutral|"
    "central|decentral|industrial|modern|liberal|national|privat|harmon|"
    "symbol|summar|legal|marginal|jeopard|sympath|final|capital|"
    "operational|institutional|polar|immun|revital|stigmat|categor|critic|"
    "emphas|special|international|colon|decolon|urban|equal|optim|"
    "galvan|familiar|fertil|vital|subsid|visual"
)

SPELLING_RULES = [
    # Single-token forms for multi-word variants
    (r"\bviet[\s-]+nam", "vietnam"),
    (r"\bper[\s-]+cent\b", "percent"),
    # -ise -> -ize (restricted to known stems to avoid "rise", "wise", ...)
    (rf"\b({_IZE_STEMS})is(e|es|ed|ing|ation|ations|er|ers)\b", r"\1iz\2"),
    # -or -> -our
    (r"\blabor(s|ed|ing|er|ers)?\b", r"labour\1"),
    (r"\bfavor(s|ed|ing|able|ably|ite|ites)?\b", r"favour\1"),
    (r"\bhonor(s|ed|ing|able|ably|ary)?\b", r"honour\1"),
    (r"\bneighbor(s|ing|hood|hoods|ly|liness)?\b", r"neighbour\1"),
    (r"\bendeavor(s|ed|ing)?\b", r"endeavour\1"),
    (r"\bbehavior(s|al)?\b", r"behaviour\1"),
    (r"\bcolor(s|ed)?\b", r"colour\1"),
    (r"\bharbor(s|ed|ing)?\b", r"harbour\1"),
    (r"\b(vig|rig|hum|arm|sav)or\b", r"\1our"),
    # -er -> -re
    (r"\bcenter(s)?\b", r"centre\1"),
    (r"\bcentered\b", "centred"),
    (r"\btheater(s)?\b", r"theatre\1"),
    (r"\bfiber(s)?\b", r"fibre\1"),
    # -se -> -ce
    (r"\bdefense(s|less)?\b", r"defence\1"),
    (r"\boffense(s)?\b", r"offence\1"),
    # programme, dialogue, fulfil
    (r"\bprogram(s)?\b", r"programme\1"),
    (r"\bdialog(s)?\b", r"dialogue\1"),
    (r"\bfulfill(ment|s)?\b", r"fulfil\1"),
]
_SPELLING_RULES_COMPILED = [(re.compile(p), r) for p, r in SPELLING_RULES]

_HYPHEN_PAIR = re.compile(r"\b([a-z]+)-\s*([a-z]+)\b")


def _basic_normalize(text):
    """Unicode, quotes, lowercase, contractions, possessives, fixed phrases."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_QUOTE_TRANSLATION)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    for pattern, repl in _CONTRACTIONS + _POSSESSIVES + _FIXED_EXPRESSIONS:
        text = re.sub(pattern, repl, text)
    # Drop numeric ordinals ("25th meeting" would otherwise leave "th").
    text = re.sub(r"\b\d+(?:st|nd|rd|th)\b", " ", text)
    # Remove dashes used as punctuation (figure/en/em dash, double hyphen).
    text = re.sub(r"[\u2012-\u2015]|--+", " ", text)
    return text


def learn_compound_map(texts, min_hyphenated=3, min_closed=10):
    """
    Learn how to write compounds that appear both hyphenated ("a-b") and
    closed ("ab") in the corpus, so that each compound has a single form.

    A pair is considered when the hyphenated form occurs at least
    `min_hyphenated` times and the closed form at least `min_closed` times.
    This repairs two different problems:
      * spelling variants:  co-operation / cooperation, peace-keeping
      * line-wrap breaks:   "com-<newline>munity", "re-<newline>sources"

    Direction:
      * "join"  (a-b -> ab) when the first part is a bound prefix (co-, non-),
        either part is a stopword (well-being, so-called), or the closed form
        is the corpus majority (cooperation, peacekeeping, community).
      * "split" (ab -> a b) otherwise, i.e. the hyphenated form dominates and
        both parts are words (secretarygeneral -> secretary general), so the
        compound matches its far more common spaced/hyphenated spelling.

    Returns (compound_map, review_rows). compound_map has keys "join"
    {(a, b): "ab"} and "split" {"ab": "a b"}; review_rows lists every
    decision with its counts so it can be inspected.
    """
    hyphen_counts = Counter()
    word_counts = Counter()

    for text in texts:
        text = _basic_normalize(text)
        hyphen_counts.update(_HYPHEN_PAIR.findall(text))
        word_counts.update(re.findall(r"[a-z]+", text))

    compound_map = {"join": {}, "split": {}}
    review = []
    for (a, b), n_hyph in hyphen_counts.items():
        closed = a + b
        n_closed = word_counts[closed]
        if n_hyph < min_hyphenated or n_closed < min_closed:
            continue

        if (a, b) in COMPOUND_EXCLUSIONS:
            direction = "excluded"
        elif (
            a in BOUND_PREFIXES
            or a in STOPWORDS or b in STOPWORDS
            or n_closed >= n_hyph
        ):
            direction = "join"
            compound_map["join"][(a, b)] = closed
        else:
            direction = "split"
            compound_map["split"][closed] = f"{a} {b}"

        review.append({
            "hyphenated": f"{a}-{b}",
            "closed": closed,
            "hyphenated_count": n_hyph,
            "closed_count": n_closed,
            "direction": direction,
        })

    return compound_map, review


def normalize_text(text, compound_map=None):
    """
    Return a cleaned, lowercase version of a speech that keeps punctuation
    (needed for sentence splitting) but harmonizes spelling and encoding.
    """
    if not isinstance(text, str):
        return ""

    text = _basic_normalize(text)

    if compound_map:
        join = compound_map["join"]
        split = compound_map["split"]
        text = _HYPHEN_PAIR.sub(
            lambda m: join.get((m.group(1), m.group(2)), m.group(0)),
            text,
        )
        if split:
            text = re.sub(
                r"\b(" + "|".join(map(re.escape, split)) + r")\b",
                lambda m: split[m.group(1)],
                text,
            )

    for pattern, repl in _SPELLING_RULES_COMPILED:
        text = pattern.sub(repl, text)

    # Preserve dotted acronyms: u.s. -> u_s, u.s.s.r. -> u_s_s_r
    text = re.sub(
        r"\b(?:[a-z]\.){2,}",
        lambda m: m.group(0).rstrip(".").replace(".", "_"),
        text,
    )

    # Drop the period after common abbreviations ("mr." -> "mr").
    text = _ABBREVIATIONS.sub(r"\1", text)

    return text


# ---------------------------------------------------------------------------
# Sentence splitting and tokenization
# ---------------------------------------------------------------------------

# Sentence/clause boundaries. Newlines are NOT boundaries because many
# speeches are hard-wrapped mid-sentence.
_SENTENCE_BOUNDARY = re.compile(r"[.!?;:]+(?=\s|$|[\"')\]])")

_TOKEN = re.compile(r"[a-z]+(?:_[a-z]+)*(?:'[a-z]+)?")


def tokenize(text):
    """Word tokens with stopwords preserved (input should be normalized)."""
    return _TOKEN.findall(text)


def split_sentences(text):
    """Split normalized text into sentence-level token lists."""
    sentences = []
    for chunk in _SENTENCE_BOUNDARY.split(text):
        tokens = tokenize(chunk)
        if tokens:
            sentences.append(tokens)
    return sentences


def remove_stopwords(tokens):
    """Drop stopwords and one-character tokens."""
    return [
        t for t in tokens
        if t not in STOPWORDS and len(t.replace("_", "")) > 1
    ]


# ---------------------------------------------------------------------------
# Phrase filtering (used with gensim Phrases, delimiter="__")
# ---------------------------------------------------------------------------

PHRASE_DELIMITER = "__"


def is_valid_phrase(token, max_parts=4):
    """
    A learned phrase is kept only if it
      * does not start or end with a stopword/negation
        (rejects "has__been", "my__country", "not__only", "would__like"), and
      * has at most `max_parts` words, so long chains such as
        "session__of__the__general__assembly" do not absorb core concepts.
    """
    parts = token.split(PHRASE_DELIMITER)
    if len(parts) == 1:
        return True
    if len(parts) > max_parts:
        return False
    return (
        parts[0] not in PHRASE_EDGE_STOPWORDS
        and parts[-1] not in PHRASE_EDGE_STOPWORDS
    )


def apply_phrases(sentence, bigram_phraser, trigram_phraser=None, max_parts=4):
    """
    Apply phrase models to one tokenized sentence, then repair rejected
    phrases:
      * a rejected trigram-stage phrase is split and re-phrased with the
        bigram model only ("session__of__the__general__assembly" becomes
        "session of the general__assembly"),
      * a rejected bigram is split into its words.
    Stopwords are still present in the output; call remove_stopwords() next.
    """
    tokens = bigram_phraser[sentence]
    if trigram_phraser is not None:
        tokens = trigram_phraser[tokens]

    output = []
    for token in tokens:
        if is_valid_phrase(token, max_parts):
            output.append(token)
            continue

        words = token.split(PHRASE_DELIMITER)
        for sub in bigram_phraser[words]:
            if is_valid_phrase(sub, max_parts):
                output.append(sub)
            else:
                output.extend(sub.split(PHRASE_DELIMITER))
    return output
