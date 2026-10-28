import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|ru|kz|net|org|top|xyz|info|site|club)\b/?\S*)", re.I)
PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{8,}\d")
NUM_RE = re.compile(r"\d+")
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]")
TOKEN_RE = re.compile(r"[a-zа-яё]+")

SPECIAL = {"urltoken", "phonetoken", "numtoken", "emojitoken"}

# латинские «двойники» кириллических букв (приём обфускации спама)
LAT2CYR = str.maketrans({"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
                         "k": "к", "m": "м", "t": "т", "b": "в", "h": "н"})
CYR_RE = re.compile(r"[а-яё]")
LAT_RE = re.compile(r"[a-z]")

STOPWORDS = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было
вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж
говорил там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего
раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы
нее сейчас были куда зачем всех никогда можно при два об другой хоть после над больше тот через эти нас про всего
них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда
конечно всю между""".split())


def clean_text(text: str) -> str:
    t = str(text).lower().replace("ё", "е")
    t = URL_RE.sub(" urltoken ", t)
    t = PHONE_RE.sub(" phonetoken ", t)
    t = NUM_RE.sub(" numtoken ", t)
    t = EMOJI_RE.sub(" emojitoken ", t)
    out = []
    for tok in TOKEN_RE.findall(t):
        if CYR_RE.search(tok) and LAT_RE.search(tok):          # слово смешанного алфавита -> кириллица
            tok = tok.translate(LAT2CYR)
        out.append(tok)
    return " ".join(out)


def tokenize(text: str):
    return clean_text(text).split()


_ENDINGS = sorted("""иями ями ами ого его ому ему ыми ими ией ий ый ой ая яя ое ее ые ие ых их ую юю ом ем ах ях
ов ев ей ам ям ою ею ия ью ть ют ут ят ат ешь ишь ет ит им ла ли ло ал ил ыл ется ится ются ится ся
а я у ю о е ы и ь й""".split(), key=len, reverse=True)


def stem(word: str) -> str:
    """Упрощённый стеммер для русского (отсечение окончаний, остаток >= 3 букв)."""
    if word in SPECIAL or not CYR_RE.search(word) or len(word) <= 3:
        return word
    for end in _ENDINGS:
        if word.endswith(end) and len(word) - len(end) >= 3:
            return word[: -len(end)]
    return word


def preprocess(text: str, remove_stop: bool = True) -> list[str]:
    toks = tokenize(text)
    if remove_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks]


def preprocess_str(text: str) -> str:
    return " ".join(preprocess(text))
