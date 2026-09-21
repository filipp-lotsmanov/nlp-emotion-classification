"""Text normalisation, as a sequence of small functions applied in a fixed order.

Order matters more than any individual rule. Two examples that decide the shape
of the training data:

- :func:`mask_numbers` runs *after* :func:`expand_slang`, so ``b4`` becomes
  ``before`` rather than ``b[NUM]``.
- :func:`mark_shouting` runs after every masking step, so a placeholder inserted
  earlier (``[URL]``, ``[NAME]``) is not itself read as a shouted word and
  wrapped in a second ``[CAPS]`` marker.

The step that matters most for what comes later is :func:`mark_shouting`. It
lowercases everything and puts a literal ``[CAPS]`` token in front of each
all-caps word, so shouting survives as a feature instead of being erased. The
error analysis in ``analysis/`` then found ALL-CAPS to be the single strongest
predictor of a wrong answer -- an error rate of 56.8% against 6.9%. That finding
only exists because this function kept the signal, and it is the reason the
order here is pinned by tests rather than left to whoever edits next.
"""

from __future__ import annotations

import re
from collections.abc import Callable

Step = Callable[[str], str]

# Ranges lifted from the original build. Deliberately not `emoji`-the-package:
# the set of code points is part of what the dataset is, so it is written down
# here rather than tracking a dependency's release notes.
EMOJI = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols and pictographs
    "\U0001f680-\U0001f6ff"  # transport and map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"  # dingbats
    "\U000024c2-\U0001f251"
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U0001fa00-\U0001faff"
    "]+",
    flags=re.UNICODE,
)

EMOTICONS = re.compile(
    r":-?\)|:-?\(|:-?[DPOp]|:-?/|:-?\||;-?\)|</?3|[xX]D|>:-?\(|D:|:-?\*|:\^"
    r"|=[)DP/]|8-?\)|B-?\)|[oO]_[oO]|\^_?\^|-_-|T_T|TT|><"
)

QUOTES = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‹": "'",
    "›": "'",
    "`": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',
    "–": "-",
    "—": "-",
    "−": "-",
}

# Canonical placeholder names, and the spellings seen in the source corpora.
# The Cyrillic variants are here because the same normalisation ran over the
# Russian transcripts in the pipeline.
PLACEHOLDERS: dict[str, tuple[str, ...]] = {
    "NAME": ("NAME", "ИМЕНИ"),
    "RELIGION": ("RELIGION", "РЕЛИГИЯ"),
    "CITY": ("CITY", "ГОРОД"),
    "LOCATION": ("LOCATION", "МЕСТО"),
    "COUNTRY": ("COUNTRY", "СТРАНА"),
    "ORGANIZATION": ("ORGANIZATION", "ORG"),
    "URL": ("URL",),
    "TAG": ("TAG", "TEG"),
    "NUM": ("NUM", "NUMBER"),
}

# The entity lists the original build actually used. spaCy was tried first and
# was not installed in that environment, so this keyword pass is what produced
# the published dataset -- which is lucky, because it makes the build
# reproducible from a pandas install rather than a model download.
NAMES = [
    "Trump",
    "Biden",
    "Obama",
    "Clinton",
    "Bush",
    "Putin",
    "Xi",
    "Donald",
    "Joe",
    "Barack",
    "Hillary",
    "George",
    "John",
    "Michael",
    "David",
    "James",
    "Robert",
    "William",
    "Mary",
    "Jennifer",
    "Linda",
    "Elizabeth",
    "Jessica",
]
RELIGIONS = [
    "Christianity",
    "Islam",
    "Judaism",
    "Buddhism",
    "Hinduism",
    "Catholic",
    "Orthodox",
    "Protestant",
    "Muslim",
    "Christian",
    "Jewish",
    "Buddhist",
    "Hindu",
    "atheist",
    "agnostic",
]
CITIES = (
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "London",
    "Paris",
    "Berlin",
    "Moscow",
    "Tokyo",
    "Beijing",
    "Mumbai",
    "Dubai",
    "Singapore",
    "Sydney",
    "Toronto",
)

SLANG: dict[str, str] = {
    # internet shorthand
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "rofl": "rolling on floor laughing",
    "omg": "oh my god",
    "wtf": "what the fuck",
    "wth": "what the hell",
    "ffs": "for fuck sake",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "afaik": "as far as i know",
    "iirc": "if i recall correctly",
    "btw": "by the way",
    "fyi": "for your information",
    "ftw": "for the win",
    "idk": "i do not know",
    "idc": "i do not care",
    "jk": "just kidding",
    "lmk": "let me know",
    "nvm": "never mind",
    "asap": "as soon as possible",
    "brb": "be right back",
    "gtg": "got to go",
    "g2g": "got to go",
    "ttyl": "talk to you later",
    "tmi": "too much information",
    "irl": "in real life",
    "rip": "rest in peace",
    "til": "today i learned",
    "nsfw": "not safe for work",
    "tldr": "too long did not read",
    "tl;dr": "too long did not read",
    "ama": "ask me anything",
    "dae": "does anyone else",
    "yolo": "you only live once",
    "fomo": "fear of missing out",
    # contractions and short forms
    "u": "you",
    "ur": "your",
    "r": "are",
    "y": "why",
    "bc": "because",
    "bcuz": "because",
    "cuz": "because",
    "cos": "because",
    "b4": "before",
    "gr8": "great",
    "l8r": "later",
    "m8": "mate",
    "h8": "hate",
    "w8": "wait",
    "ppl": "people",
    "plz": "please",
    "pls": "please",
    "thx": "thanks",
    "tks": "thanks",
    "ty": "thank you",
    "tysm": "thank you so much",
    "np": "no problem",
    "yw": "you are welcome",
    "ya": "you",
    "yea": "yes",
    "yeah": "yes",
    "nah": "no",
    "nope": "no",
    "yup": "yes",
    "yep": "yes",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "gimme": "give me",
    "lemme": "let me",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "do not know",
    "shoulda": "should have",
    "woulda": "would have",
    "coulda": "could have",
    "shouldve": "should have",
    "wouldve": "would have",
    "couldve": "could have",
    # reactions
    "ikr": "i know right",
    "ik": "i know",
    "fr": "for real",
    "ngl": "not gonna lie",
    "tho": "though",
    "prolly": "probably",
    "obv": "obviously",
    "def": "definitely",
    "probs": "probably",
    "srsly": "seriously",
    "sry": "sorry",
    "mb": "my bad",
    # time
    "rn": "right now",
    "atm": "at the moment",
    "eta": "estimated time of arrival",
    "24/7": "twenty four seven",
    # platforms
    "dm": "direct message",
    "pm": "private message",
    "rt": "retweet",
    "fb": "facebook",
    "ig": "instagram",
    "yt": "youtube",
    "sub": "subscribe",
    "unsub": "unsubscribe",
    "fav": "favorite",
    "fave": "favorite",
    # expressions
    "af": "as fuck",
    "asf": "as fuck",
    "bae": "babe",
    "bff": "best friends forever",
    "bf": "boyfriend",
    "gf": "girlfriend",
    "gg": "good game",
    "wp": "well played",
    "gl": "good luck",
    "hf": "have fun",
    "gj": "good job",
    "nm": "nothing much",
    "sup": "what is up",
    "wassup": "what is up",
    "whatup": "what is up",
    # misc
    "amirite": "am i right",
    "abt": "about",
    "abo": "about",
    "w/": "with",
    "w/o": "without",
    "b/c": "because",
    "op": "overpowered",
    "ftfy": "fixed that for you",
    "oc": "original content",
    "goat": "greatest of all time",
    "bday": "birthday",
    "congrats": "congratulations",
    "pic": "picture",
    "pics": "pictures",
    "vid": "video",
    "vids": "videos",
    "msg": "message",
    "msgs": "messages",
    "aka": "also known as",
    "etc": "et cetera",
    "vs": "versus",
    "thru": "through",
    "approx": "approximately",
    "min": "minute",
    "mins": "minutes",
    "sec": "second",
    "secs": "seconds",
    "hr": "hour",
    "hrs": "hours",
    "wk": "week",
    "wks": "weeks",
    "yr": "year",
    "yrs": "years",
    "mo": "month",
    "mos": "months",
    "afk": "away from keyboard",
    "ez": "easy",
    "noob": "newbie",
    "n00b": "newbie",
    "pwn": "own",
    "rekt": "wrecked",
    "kek": "lol",
    "txt": "text",
    "cya": "see you",
    "l8": "late",
    "2day": "today",
    "2nite": "tonight",
    "2moro": "tomorrow",
    "2mrw": "tomorrow",
    "tmr": "tomorrow",
    "tmrw": "tomorrow",
    "tdy": "today",
    "yday": "yesterday",
    "yest": "yesterday",
}

_PLACEHOLDER_PATTERNS = tuple(
    (re.compile(r"\[" + re.escape(spelling) + r"\]", re.IGNORECASE), f"[{canonical}]")
    for canonical, spellings in PLACEHOLDERS.items()
    for spelling in spellings
)
_NAME_PATTERNS = tuple((re.compile(rf"\b{re.escape(n)}\b"), "[NAME]") for n in NAMES)
_RELIGION_PATTERNS = tuple(
    (re.compile(rf"\b{re.escape(r)}\b", re.IGNORECASE), "[RELIGION]") for r in RELIGIONS
)
_CITY_PATTERNS = tuple((re.compile(rf"\b{re.escape(c)}\b"), "[CITY]") for c in CITIES)
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#\w+")
_NUMBER = re.compile(r"\b\d+\.?\d*\b")
_WHITESPACE = re.compile(r"\s+")
_PUNCT_SPACING = re.compile(r"\s+([.,!?;:])")
_REPEAT = re.compile(r"(.)\1{2,}")


def normalize_quotes(text: str) -> str:
    """Smart quotes and long dashes to their ASCII equivalents."""
    for fancy, plain in QUOTES.items():
        text = text.replace(fancy, plain)
    return text


def normalize_placeholders(text: str) -> str:
    """``[Name]``, ``[NAME]``, ``[ИМЕНИ]`` all become ``[NAME]``."""
    for pattern, canonical in _PLACEHOLDER_PATTERNS:
        text = pattern.sub(canonical, text)
    return text


def strip_emoji(text: str) -> str:
    """Remove pictographic emoji and the common ASCII emoticons."""
    return EMOTICONS.sub("", EMOJI.sub("", text))


def expand_slang(text: str) -> str:
    """Expand shorthand a word at a time, ignoring surrounding punctuation."""
    words = []
    for word in text.split():
        stripped = word.lower().strip(".,!?;:\"'")
        words.append(SLANG.get(stripped, word))
    return " ".join(words)


def mask_entities(text: str) -> str:
    """Replace URLs, handles, hashtags, numbers and known entities with markers.

    The entity lists are short and English-only. That is a real limitation of
    the source build rather than an oversight here, and ``docs/dataset.md``
    says so: a name outside the list survives into the training text.
    """
    text = _URL.sub("[URL]", text)
    text = _MENTION.sub("[TAG]", text)
    text = _HASHTAG.sub("[TAG]", text)
    text = _NUMBER.sub("[NUM]", text)
    for patterns in (_NAME_PATTERNS, _RELIGION_PATTERNS, _CITY_PATTERNS):
        for pattern, marker in patterns:
            text = pattern.sub(marker, text)
    return text


def collapse_whitespace(text: str) -> str:
    """Tabs, newlines and runs of spaces to a single space, then trim."""
    return _WHITESPACE.sub(" ", text).strip()


def mark_shouting(text: str) -> str:
    """Lowercase everything, preceding each all-caps word with ``[CAPS]``.

    Erasing case would delete the strongest error signal in the whole dataset;
    keeping raw case would make ``ANGRY`` and ``angry`` different tokens. The
    marker keeps the signal and merges the vocabulary. Placeholders inserted by
    earlier steps pass through untouched.
    """
    out = []
    for word in text.split():
        if word.startswith("[") and word.endswith("]"):
            out.append(word)
            continue
        bare = word.strip(".,!?;:'\"")
        if len(bare) >= 2 and bare.isalpha() and bare.isupper():
            out.append("[CAPS]")
        out.append(word.lower())
    return " ".join(out)


def normalize_punctuation(text: str) -> str:
    """Cap runs of punctuation, keeping one repeat as a signal that it was there."""
    text = re.sub(r"!{2,}", "!!", text)
    text = re.sub(r"\?{2,}", "??", text)
    text = re.sub(
        r"[?!]{3,}",
        lambda m: "!!" if m.group().count("!") > m.group().count("?") else "??",
        text,
    )
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r",{2,}", ",", text)
    return _PUNCT_SPACING.sub(r"\1", text)


def squeeze_repeats(text: str) -> str:
    """``soooo`` to ``soo``: cap a repeated character at two, keeping the emphasis."""
    out = []
    for word in text.split():
        if word.startswith("[") and word.endswith("]"):
            out.append(word)
        else:
            out.append(_REPEAT.sub(r"\1\1", word))
    return " ".join(out)


def drop_repeated_placeholders(text: str) -> str:
    """``[TAG] [TAG] [TAG]`` to ``[TAG]``. A tweet of nothing but handles is not signal."""
    out: list[str] = []
    for word in text.split():
        is_placeholder = word.startswith("[") and word.endswith("]")
        if is_placeholder and out and out[-1] == word:
            continue
        out.append(word)
    return " ".join(out)


# The order the published dataset was built in, split where the original build
# removed duplicates. That split is not cosmetic: deduplicating before the case
# and punctuation passes keeps 671 rows that deduplicating after would collapse,
# because "Really?!" and "REALLY!!!!" are distinct here and identical later.
# Reproducing the published set means reproducing the split, and
# `test_deduplicating_late_would_lose_rows` pins the difference.
BEFORE_DEDUPE = (
    normalize_quotes,
    normalize_placeholders,
    strip_emoji,
    expand_slang,
    mask_entities,
    collapse_whitespace,
)

AFTER_DEDUPE = (
    mark_shouting,
    collapse_whitespace,
    normalize_punctuation,
    squeeze_repeats,
)

# Applied once the length filter has run, so a text that is only repeated
# placeholders is dropped for being short rather than quietly rescued.
FINALLY = (
    drop_repeated_placeholders,
    collapse_whitespace,
)

PIPELINE = BEFORE_DEDUPE + AFTER_DEDUPE + FINALLY


def _apply(text: str, steps: tuple[Step, ...]) -> str:
    for step in steps:
        text = step(text)
    return text


def clean_early(text: str) -> str:
    """Everything the original build did before removing duplicates."""
    return _apply(text, BEFORE_DEDUPE)


def clean_late(text: str) -> str:
    """Everything it did after: case, punctuation, repeated characters."""
    return _apply(text, AFTER_DEDUPE)


def tidy(text: str) -> str:
    """The last pass, after the length filter."""
    return _apply(text, FINALLY)


def clean_text(text: str) -> str:
    """Every normalisation step, in order. For a single text, outside a build."""
    return _apply(text, PIPELINE)
