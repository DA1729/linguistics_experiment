import hashlib
import random

language_names = dict(line.split(' ', 1) for line in """fra_Latn French
deu_Latn German
arb_Arab Modern Standard Arabic
jpn_Jpan Japanese
hin_Deva Hindi
spa_Latn Spanish
rus_Cyrl Russian
zho_Hans Chinese (Simplified)
swh_Latn Swahili
tam_Taml Tamil
kor_Hang Korean
tur_Latn Turkish
ben_Beng Bengali
ita_Latn Italian
ell_Grek Greek
heb_Hebr Hebrew
tha_Thai Thai
vie_Latn Vietnamese
ind_Latn Indonesian
fin_Latn Finnish
pol_Latn Polish
ukr_Cyrl Ukrainian
nld_Latn Dutch
por_Latn Portuguese
pes_Arab Western Persian
urd_Arab Urdu
tel_Telu Telugu
mal_Mlym Malayalam
kan_Knda Kannada
mar_Deva Marathi
guj_Gujr Gujarati
pan_Guru Eastern Panjabi
npi_Deva Nepali
sin_Sinh Sinhala
mya_Mymr Burmese
khm_Khmr Khmer
lao_Laoo Lao
bod_Tibt Tibetan
kat_Geor Georgian
hye_Armn Armenian
amh_Ethi Amharic
hau_Latn Hausa
yor_Latn Yoruba
zul_Latn Zulu
som_Latn Somali
ron_Latn Romanian
hun_Latn Hungarian
swe_Latn Swedish
ces_Latn Czech
kaz_Cyrl Kazakh""".splitlines())


def select_pool(config: dict) -> list[str]:
    custom = config['languages']
    count = config['language_count']
    if not custom and not 2 <= count <= len(language_names):
        raise ValueError('language_count must be 2..50, or supply explicit languages')
    pool = list(custom or list(language_names)[:count])
    if len(pool) < 2 or len(set(pool)) != len(pool) or 'eng_Latn' in pool:
        raise ValueError('Use at least two distinct intermediate languages, excluding English')
    if any(not isinstance(code, str) for code in pool):
        raise ValueError('Language identifiers must be strings')
    return pool


def derive_seed(seed: int, *parts: object) -> int:
    key = '\0'.join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big')


def make_schedule(pool: list[str], rounds: int, seed: int) -> list[str]:
    if len(set(pool)) != len(pool) or len(pool) < 2:
        raise ValueError('Schedules need at least two distinct languages')
    rng = random.Random(seed)
    path = []
    while len(path) < rounds:
        block = rng.sample(pool, len(pool))
        # Rejection sampling keeps the next permutation uniform conditional on its boundary.
        while path and block[0] == path[-1]:
            block = rng.sample(pool, len(pool))
        path.extend(block)
    return path[:rounds]


def observation_rounds(settings: dict) -> list[int]:
    rounds, every = settings['rounds'], settings['observe_every']
    values = {0, rounds, *settings['observation_rounds']}
    if every:
        values.update(range(0, rounds + 1, every))
    return sorted(r for r in values if 0 <= r <= rounds)
