#
# Copyright 2024 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
import logging

from typing import Any, Iterable, List, TYPE_CHECKING, Optional, Set, Tuple

if TYPE_CHECKING:
    from torch import IntTensor, FloatTensor

__all__ = ['CodePointTokenizer', 'OctetTokenizer']

logger = logging.getLogger(__name__)


LANG_TO_ISO = {'acoli': 'ach',
               'afar': 'aar',
               'afrikaans': 'afr',
               'akan': 'aka',
               'akkadian': 'akk',
               'albanian': 'sqi',
               'alemannic': 'gsw',
               'amharic': 'amh',
               'ancient_greek': 'grc',
               'arabic': 'ara',
               'aragonese': 'arg',
               'aramaic': 'arc',
               'armenian': 'hye',
               'asturian': 'ast',
               'aymara': 'aym',
               'azerbaijani': 'aze',
               'balinese': 'ban',
               'bambara': 'bam',
               'bashkir': 'bak',
               'basque': 'eus',
               'belarusian': 'bel',
               'bengali': 'ben',
               'bihari': 'bih',
               'bislama': 'bis',
               'bosnian': 'bos',
               'breton': 'bre',
               'bulgarian': 'bul',
               'burmese': 'mya',
               'catalan': 'cat',
               'cebuano': 'ceb',
               'chichewa': 'nya',
               'chinese': 'cmn',
               'church_slavonic': 'chu',
               'classical_armenian': 'xcl',
               'corsican': 'cos',
               'crimean_tatar': 'crh',
               'croatian': 'hrv',
               'czech': 'ces',
               'dagbani': 'dag',
               'danish': 'dan',
               'dutch': 'nld',
               'egyptian': 'egy',
               'elamite': 'elx',
               'english': 'eng',
               'esperanto': 'epo',
               'estonian': 'est',
               'etruscan': 'ett',
               'ewe': 'ewe',
               'fanti': 'fat',
               'faroese': 'fao',
               'fijian': 'fij',
               'finnish': 'fin',
               'french': 'fra',
               'friulian': 'fur',
               'fulah': 'ful',
               'galician': 'glg',
               'geez': 'gez',
               'georgian': 'kat',
               'german': 'deu',
               'german_shorthand': 'qaa',
               'greenlandic': 'kal',
               'guarani': 'grn',
               'gujarati': 'guj',
               'haitian_creole': 'hat',
               'hausa': 'hau',
               'hawaiian': 'haw',
               'hebrew': 'heb',
               'hieroglyphic_luwian': 'hlu',
               'hindi': 'hin',
               'hittite': 'hit',
               'hmong': 'hmn',
               'hungarian': 'hun',
               'hurrian': 'xhu',
               'icelandic': 'isl',
               'igbo': 'ibo',
               'ikposo': 'kpo',
               'iloko': 'ilo',
               'indonesian': 'ind',
               'interlingua': 'ina',
               'interlingue': 'ile',
               'inupiaq': 'ipk',
               'irish': 'gle',
               'italian': 'ita',
               'japanese': 'jpn',
               'javanese': 'jav',
               'kabyle': 'kab',
               'kannada': 'kan',
               'kazakh': 'kaz',
               'khasi': 'kha',
               'kikuyu': 'kik',
               'kimbundu': 'kmb',
               'kinyarwanda': 'kin',
               'kirundi': 'run',
               'klingon': 'tlh',
               'korean': 'kor',
               'kurdish': 'kur',
               'kyrgyz': 'kir',
               'ladino': 'lad',
               'lao': 'lao',
               'latgalian': 'ltg',
               'latin': 'lat',
               'latvian': 'lav',
               'limburgish': 'lim',
               'lingala': 'lin',
               'lithuanian': 'lit',
               'lombard': 'lmo',
               'luganda': 'lug',
               'luxembourgish': 'ltz',
               'macedonian': 'mkd',
               'magahi': 'mag',
               'malagasy': 'mlg',
               'malay': 'msa',
               'malayalam': 'mal',
               'maltese': 'mlt',
               'manx': 'glv',
               'maori': 'mri',
               'marathi': 'mar',
               'masai': 'mas',
               'mauritian_creole': 'mfe',
               'middle_dutch': 'dum',
               'middle_french': 'frm',
               'minangkabau': 'min',
               'modern_greek': 'ell',
               'mongolian': 'mon',
               'nauru': 'nau',
               'nepali': 'nep',
               'newari': 'new',
               'northern_sotho': 'nso',
               'norwegian': 'nor',
               'nyankole': 'nyn',
               'occitan': 'oci',
               'odia': 'ori',
               'old_persian': 'peo',
               'oromo': 'orm',
               'ottoman_turkish': 'ota',
               'pangasinan': 'pag',
               'papiamento': 'pap',
               'pashto': 'pus',
               'persian': 'fas',
               'picard': 'pcd',
               'polish': 'pol',
               'portuguese': 'por',
               'punjabi': 'pan',
               'quechua': 'que',
               'romanian': 'ron',
               'romansh': 'roh',
               'russian': 'rus',
               'samoan': 'smo',
               'sango': 'sag',
               'sanskrit': 'san',
               'sardinian': 'srd',
               'scots': 'sco',
               'scottish_gaelic': 'gla',
               'serbian': 'srp',
               'serbian_cyrl': 'qab',
               'seychellois_creole': 'crs',
               'shona': 'sna',
               'sicilian': 'scn',
               'sidamo': 'sid',
               'sindhi': 'snd',
               'sinhala': 'sin',
               'slovak': 'slk',
               'slovenian': 'slv',
               'soga': 'xog',
               'somali': 'som',
               'southern_dagaare': 'dga',
               'southern_ndebele': 'nbl',
               'southern_sotho': 'sot',
               'spanish': 'spa',
               'sumerian': 'sux',
               'sundanese': 'sun',
               'swahili': 'swa',
               'swati': 'ssw',
               'swedish': 'swe',
               'syriac': 'syr',
               'tagalog': 'tgl',
               'tajik': 'tgk',
               'tamil': 'tam',
               'tatar': 'tat',
               'telugu': 'tel',
               'thai': 'tha',
               'tibetan': 'bod',
               'tigrinya': 'tir',
               'tocharian': 'txb',
               'tok_pisin': 'tpi',
               'tongan': 'ton',
               'tsonga': 'tso',
               'tswana': 'tsn',
               'turkish': 'tur',
               'turkmen': 'tuk',
               'ugaritic': 'uga',
               'ukrainian': 'ukr',
               'undetermined': 'und',
               'urdu': 'urd',
               'uyghur': 'uig',
               'uzbek': 'uzb',
               'venda': 'ven',
               'venetian': 'vec',
               'vietnamese': 'vie',
               'volapuk': 'vol',
               'waray': 'war',
               'welsh': 'cym',
               'west_frisian': 'fry',
               'wolaytta': 'wal',
               'wolof': 'wol',
               'xhosa': 'xho',
               'yiddish': 'yid',
               'yoruba': 'yor',
               'zhuang': 'zha',
               'zulu': 'zul'}

ISO_TO_IDX = {'ara': 0,
              'cat': 1,
              'ces': 2,
              'chu': 3,
              'cmn': 4,
              'cos': 5,
              'deu': 6,
              'dum': 7,
              'eng': 8,
              'fas': 9,
              'fin': 10,
              'fra': 11,
              'frm': 12,
              'grc': 13,
              'heb': 14,
              'ita': 15,
              'jpn': 16,
              'kat': 17,
              'lad': 18,
              'lat': 19,
              'mal': 20,
              'new': 21,
              'nld': 22,
              'nor': 23,
              'oci': 24,
              'ota': 25,
              'pcd': 26,
              'por': 27,
              'qaa': 28,
              'rus': 29,
              'san': 30,
              'spa': 31,
              'swe': 32,
              'syr': 33,
              'ukr': 34,
              'und': 35,
              'urd': 36,
              'yid': 37,
              'ron': 38,
              'qab': 39,
              'lav': 40,
              'gle': 41,
              'slv': 42,
              'lit': 43,
              'pol': 44,
              'gez': 45,
              'xcl': 46,
              'vie': 47,
              'slk': 48,
              'kor': 49,
              'dan': 50,
              'ell': 51,
              'ilo': 52,
              'xho': 53,
              'hun': 54,
              'est': 55,
              'cym': 56,
              'mlt': 57,
              'eus': 58,
              'msa': 59,
              'hrv': 60,
              'gla': 61,
              'hat': 62,
              'tlh': 63,
              'afr': 64,
              'roh': 65,
              'bul': 66,
              'sco': 67,
              'kin': 68,
              'glg': 69,
              'mlg': 70,
              'jav': 71,
              'kal': 72,
              'ast': 73,
              'srp': 74,
              'yor': 75,
              'glv': 76,
              'vol': 77,
              'lmo': 78,
              'tgl': 79,
              'ltg': 80,
              'ind': 81,
              'ton': 82,
              'war': 83,
              'wol': 84,
              'hye': 85,
              'run': 86,
              'sun': 87,
              'isl': 88,
              'tam': 89,
              'lug': 90,
              'aka': 91,
              'sot': 92,
              'orm': 93,
              'fij': 94,
              'bre': 95,
              'haw': 96,
              'ina': 97,
              'bis': 98,
              'uzb': 99,
              'epo': 100,
              'hau': 101,
              'nya': 102,
              'tso': 103,
              'ile': 104,
              'sag': 105,
              'mfe': 106,
              'hmn': 107,
              'lim': 108,
              'mri': 109,
              'tpi': 110,
              'crs': 111,
              'kab': 112,
              'fry': 113,
              'smo': 114,
              'ltz': 115,
              'ban': 116,
              'ven': 117,
              'zul': 118,
              'srd': 119,
              'gsw': 120,
              'kha': 121,
              'ibo': 122,
              'nau': 123,
              'kmb': 124,
              'aar': 125,
              'vec': 126,
              'zha': 127,
              'ssw': 128,
              'que': 129,
              'sna': 130,
              'kik': 131,
              'fao': 132,
              'mkd': 133,
              'swa': 134,
              'mon': 135,
              'grn': 136,
              'bos': 137,
              'aym': 138,
              'ceb': 139,
              'som': 140,
              'tuk': 141,
              'tsn': 142,
              'scn': 143,
              'aze': 144,
              'sqi': 145,
              'tat': 146,
              'akk': 147,
              'egy': 148,
              'sux': 149,
              'txb': 150,
              'ett': 151,
              'peo': 152,
              'tur': 153,
              'arc': 154,
              'elx': 155,
              'ewe': 156,
              'pap': 157,
              'xhu': 158,
              'tgk': 159,
              'lin': 160,
              'fur': 161,
              'nyn': 162,
              'sid': 163,
              'amh': 164,
              'tir': 165,
              'dag': 166,
              'wal': 167,
              'ful': 168,
              'ach': 169,
              'dga': 170,
              'xog': 171,
              'kpo': 172,
              'mas': 173,
              'fat': 174,
              'bam': 175,
              'uig': 176,
              'min': 177,
              'bel': 178,
              'lao': 179,
              'nbl': 180,
              'guj': 181,
              'tha': 182,
              'mya': 183,
              'hin': 184,
              'pag': 185,
              'kan': 186,
              'tel': 187,
              'hlu': 188,
              'hit': 189,
              'uga': 190,
              'snd': 191,
              'mar': 192,
              'ipk': 193,
              'mag': 194,
              'bod': 195,
              'kir': 196,
              'kaz': 197,
              'ben': 198,
              'sin': 199,
              'bak': 200,
              'arg': 201,
              'pan': 202,
              'nso': 203,
              'ori': 204,
              'nep': 205,
              'kur': 206,
              'bih': 207,
              'crh': 208,
              'pus': 209}

# Maps dataset language tags from the pretraining data to ISO codes
# resolvable through ISO_TO_IDX.
DATASET_TAG_TO_ISO = {
    'Acoli': 'ach',
    'Afar': 'aar',
    'Afrikaans': 'afr',
    'Akan': 'aka',
    'Akkadian': 'akk',
    'Albanian': 'sqi',
    'Alemannic': 'gsw',
    'Amharic': 'amh',
    'Arabic': 'ara',
    'Aragonese': 'arg',
    'Aramaic': 'arc',
    'Armenian': 'hye',
    'Asturian': 'ast',
    'Aymara': 'aym',
    'Azerbaijani': 'aze',
    'Balinese': 'ban',
    'Bambara': 'bam',
    'Bashkir': 'bak',
    'Basque': 'eus',
    'Belarusian': 'bel',
    'Bengali': 'ben',
    'Bihari': 'bih',
    'Bislama': 'bis',
    'Bosnian': 'bos',
    'Breton': 'bre',
    'Bulgarian': 'bul',
    'Burmese': 'mya',
    'Catalan': 'cat',
    'Cebuano': 'ceb',
    'Chichewa': 'nya',
    'Chinese': 'cmn',
    'Corsican': 'cos',
    'Crimean Tatar': 'crh',
    'Croatian': 'hrv',
    'Czech': 'ces',
    'Dagbani': 'dag',
    'Danish': 'dan',
    'Demotic Egyptian': 'egy',
    'Dutch': 'nld',
    'Earlier Egyptian': 'egy',
    'Early Modern Korean': 'kor',
    'Egyptian': 'egy',
    'Elamite': 'elx',
    'English': 'eng',
    'Esperanto': 'epo',
    'Estonian': 'est',
    'Etruscan': 'ett',
    'Ewe': 'ewe',
    'Fanti': 'fat',
    'Faroese': 'fao',
    'Fijian': 'fij',
    'Finnish': 'fin',
    'French': 'fra',
    'Friulian': 'fur',
    'Fulah': 'ful',
    'Galician': 'glg',
    "Ge'ez": 'gez',
    'Georgian': 'kat',
    'German': 'deu',
    'Greek': 'ell',
    'Greek, Modern': 'ell',
    'Greenlandic': 'kal',
    'Guarani': 'grn',
    'Gujarati': 'guj',
    'Haitian Creole': 'hat',
    'Haitian Creole (Latin script)': 'hat',
    'Hanmun': 'cmn',
    'Hausa': 'hau',
    'Hawaiian': 'haw',
    'Hebrew': 'heb',
    'Hindi': 'hin',
    'Hittite': 'hit',
    'Hmong': 'hmn',
    'Hungarian': 'hun',
    'Hurrian': 'xhu',
    'Icelandic': 'isl',
    'Igbo': 'ibo',
    'Ikposo': 'kpo',
    'Iloko': 'ilo',
    'Indonesian': 'ind',
    'Interlingua': 'ina',
    'Interlingue': 'ile',
    'Inupiaq': 'ipk',
    'Iranian Persian': 'fas',
    'Irish': 'gle',
    'Italian': 'ita',
    'Japanese': 'jpn',
    'Japanese (Japanese script)': 'jpn',
    'Javanese': 'jav',
    'Kabyle': 'kab',
    'Kannada': 'kan',
    'Kazakh': 'kaz',
    'Khasi': 'kha',
    'Kikuyu': 'kik',
    'Kimbundu': 'kmb',
    'Kinyarwanda': 'kin',
    'Kirundi': 'run',
    'Klingon': 'tlh',
    'Korean': 'kor',
    'Kurdish': 'kur',
    'Kyrgyz': 'kir',
    'Lao': 'lao',
    'Late Egyptian': 'egy',
    'Latgalian': 'ltg',
    'Latin': 'lat',
    'Latvian': 'lav',
    'Limburgish': 'lim',
    'Lingala': 'lin',
    'Lithuanian': 'lit',
    'Lombard': 'lmo',
    'Luganda': 'lug',
    'Luxembourgish': 'ltz',
    'Macedonian': 'mkd',
    'Magahi': 'mag',
    'Malagasy': 'mlg',
    'Malay': 'msa',
    'Malayalam': 'mal',
    'Maltese': 'mlt',
    'Manx': 'glv',
    'Marathi': 'mar',
    'Masai': 'mas',
    'Mauritian Creole': 'mfe',
    'Middle Korean': 'kor',
    'Minangkabau': 'min',
    'Modern Korean': 'kor',
    'Mongolian': 'mon',
    'M\u0101ori': 'mri',
    'Nauru': 'nau',
    'Nepali': 'nep',
    'Nigerian Fulfulde': 'ful',
    'Northern Sotho': 'nso',
    'Northern Uzbek': 'uzb',
    'Norwegian': 'nor',
    'Norwegian Bokm\u00e5l': 'nor',
    'Norwegian Nynorsk': 'nor',
    'Nyankole': 'nyn',
    'Occitan': 'oci',
    'Odia': 'ori',
    'Old Persian': 'peo',
    'Oromo': 'orm',
    'Pangasinan': 'pag',
    'Papiamento': 'pap',
    'Pashto': 'pus',
    'Persian': 'fas',
    'Plateau Malagasy': 'mlg',
    'Polish': 'pol',
    'Portugueuse': 'por',
    'Portuguese': 'por',
    'Punjabi': 'pan',
    'Quechua': 'que',
    'Romanian': 'ron',
    'Romansh': 'roh',
    'Russian': 'rus',
    'Samoan': 'smo',
    'Sango': 'sag',
    'Sanskrit': 'san',
    'Sardinian': 'srd',
    'Scots': 'sco',
    'Scottish Gaelic': 'gla',
    'Serbian': 'srp',
    'Seychellois Creole': 'crs',
    'Shona': 'sna',
    'Sicilian': 'scn',
    'Sidamo': 'sid',
    'Sindhi': 'snd',
    'Sinhala': 'sin',
    'Slovak': 'slk',
    'Slovenian': 'slv',
    'Soga': 'xog',
    'Somali': 'som',
    'Southern Dagaare': 'dga',
    'Southern Ndebele': 'nbl',
    'Southern Sotho': 'sot',
    'Spanish': 'spa',
    'Standard Latvian': 'lav',
    'Standard Malay (Latin script)': 'msa',
    'Sumerian': 'sux',
    'Sundanese': 'sun',
    'Swahil': 'swa',
    'Swahili': 'swa',
    'Swati': 'ssw',
    'Swedish': 'swe',
    'Syriac': 'syr',
    'Tagalog': 'tgl',
    'Tajik': 'tgk',
    'Tamil': 'tam',
    'Tatar': 'tat',
    'Telugu': 'tel',
    'Thai': 'tha',
    'Tibetan': 'bod',
    'Tigrinya': 'tir',
    'Tocharian': 'txb',
    'Tocharian A': 'txb',
    'Tocharian B': 'txb',
    'Tocharian (Skt.; Khot.)': 'txb',
    'Tocharian (Skt.; TA)': 'txb',
    'Tocharian (Skt.; TB)': 'txb',
    'Tok Pisin': 'tpi',
    'Tongan': 'ton',
    'Traditional Chinese': 'cmn',
    'Tsonga': 'tso',
    'Tswana': 'tsn',
    'Turkish': 'tur',
    'Turkmen': 'tuk',
    'Ugaritic': 'uga',
    'Ukrainian': 'ukr',
    'Unknown': 'und',
    'Urdu': 'urd',
    'Uyghur': 'uig',
    'Uzbek': 'uzb',
    'Venda': 'ven',
    'Venetian': 'vec',
    'Vietnamese': 'vie',
    'Volap\u00fck': 'vol',
    'Waray': 'war',
    'Welsh': 'cym',
    'West Frisian': 'fry',
    'Wolaytta': 'wal',
    'Wolof': 'wol',
    'Xhosa': 'xho',
    'Yiddish': 'yid',
    'Yoruba': 'yor',
    'Zhuang': 'zha',
    'Zulu': 'zul',
    'hlu': 'hlu',
}

LANG_IDX_TO_ISO = {v: k for k, v in ISO_TO_IDX.items()}
ISO_TO_LANG = {v: k for k, v in LANG_TO_ISO.items()}


def resolve_lang(lang: str) -> Optional[str]:
    """
    Resolves a language identifier to an ISO 639-3 code known to the
    tokenizer.

    Uses the iso639-lang package to parse language names, ISO 639-1, and
    ISO 639-3 codes.

    Returns:
        The ISO 639-3 code or None if the language is not recognized or
        not supported by the tokenizer.
    """
    from iso639 import Lang
    from iso639.exceptions import InvalidLanguageValue

    try:
        iso3 = Lang(lang).pt3
    except InvalidLanguageValue:
        return None
    if iso3 in ISO_TO_IDX:
        return iso3
    return None


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
LANG_OFFSET = 3
CODEPOINT_OFFSET = LANG_OFFSET + len(ISO_TO_IDX)

# Backwards-compatible aliases used in older code paths.
OFFSET = CODEPOINT_OFFSET
TOKEN_NUM = CODEPOINT_OFFSET


class CodePointTokenizer(object):
    """
    Dynamic Unicode code point tokenizer.

    Token layout:
    - 0: PAD
    - 1: BOS
    - 2: EOS
    - 3..CODEPOINT_OFFSET-1: language tokens
    - CODEPOINT_OFFSET+: registered Unicode code points
    """
    pad_id = PAD_ID
    bos_id = BOS_ID
    eos_id = EOS_ID

    def __init__(self,
                 codepoints: Optional[Iterable[int]] = None,
                 frozen: bool = False):
        self._codepoint_to_idx: dict[int, int] = {}
        self._idx_to_codepoint: list[int] = []
        self._frozen = False
        if codepoints:
            self.register_codepoint_values(codepoints)
        if frozen:
            self.freeze()

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def vocab_size(self) -> int:
        return CODEPOINT_OFFSET + len(self._idx_to_codepoint)

    @property
    def max_label(self) -> int:
        return self.vocab_size - 1

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def codepoints(self) -> list[int]:
        return list(self._idx_to_codepoint)

    def freeze(self):
        self._frozen = True

    def thaw(self):
        self._frozen = False

    def _register_single_codepoint(self, codepoint: int) -> bool:
        if codepoint < 0 or codepoint > 0x10FFFF:
            raise ValueError(f'Invalid code point: {codepoint}')
        if codepoint in self._codepoint_to_idx:
            return False
        if self._frozen:
            raise ValueError(f'Code point U+{codepoint:04X} is not registered and tokenizer is frozen.')
        idx = len(self._idx_to_codepoint)
        self._codepoint_to_idx[codepoint] = idx
        self._idx_to_codepoint.append(codepoint)
        return True

    def register_codepoint_values(self, codepoints: Iterable[int]) -> list[int]:
        """
        Registers raw Unicode code point values and returns newly-added values.
        """
        new_codepoints = []
        for codepoint in codepoints:
            cp = int(codepoint)
            if self._register_single_codepoint(cp):
                new_codepoints.append(cp)
        return new_codepoints

    def register_codepoints(self, text: str) -> list[int]:
        """
        Registers all code points occurring in ``text`` and returns newly-added
        code point values.
        """
        return self.register_codepoint_values(ord(char) for char in text)

    def _get_lang_token(self, lang: str) -> int:
        # Accept both language names and ISO-639-3 tags.
        normalized = DATASET_TAG_TO_ISO.get(lang, lang)
        if normalized in LANG_TO_ISO:
            normalized = LANG_TO_ISO[normalized]
        return LANG_OFFSET + ISO_TO_IDX.get(normalized, ISO_TO_IDX['und'])

    def _token_for_codepoint(self, codepoint: int) -> int:
        idx = self._codepoint_to_idx.get(codepoint)
        if idx is None:
            if self._frozen:
                raise ValueError(f'Code point U+{codepoint:04X} is not registered and tokenizer is frozen.')
            idx = len(self._idx_to_codepoint)
            self._codepoint_to_idx[codepoint] = idx
            self._idx_to_codepoint.append(codepoint)
        return CODEPOINT_OFFSET + idx

    def codepoint_for_token(self, token_id: int) -> Optional[int]:
        idx = token_id - CODEPOINT_OFFSET
        if idx < 0 or idx >= len(self._idx_to_codepoint):
            return None
        return self._idx_to_codepoint[idx]

    def token_for_codepoint(self, codepoint: int) -> Optional[int]:
        idx = self._codepoint_to_idx.get(codepoint)
        if idx is None:
            return None
        return CODEPOINT_OFFSET + idx

    def encode(self,
               text: str,
               langs: Optional[List[str]] = None,
               add_bos: bool = True,
               add_eos: bool = True) -> List[int]:
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        if langs:
            tokens.extend(self._get_lang_token(lang) for lang in langs)
        tokens.extend(self._token_for_codepoint(ord(char)) for char in text)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def _decode_language_tokens(self, ids: Iterable[int]) -> Set[str]:
        return {
            LANG_IDX_TO_ISO.get(int(token - LANG_OFFSET), 'und')
            for token in ids
            if LANG_OFFSET <= int(token) < CODEPOINT_OFFSET
        }

    def decode(self, ids: 'IntTensor') -> Tuple[str, Set[str]]:
        token_ids = [int(token) for token in ids]
        lang_ids = self._decode_language_tokens(token_ids)
        chars = []
        for token in token_ids:
            codepoint = self.codepoint_for_token(token)
            if codepoint is not None:
                chars.append(chr(codepoint))
        return ''.join(chars), lang_ids

    def decode_with_confs(self,
                          ids: 'IntTensor',
                          confidences: 'FloatTensor') -> Tuple[str, List[float], Set[str]]:
        token_ids = [int(token) for token in ids]
        lang_ids = self._decode_language_tokens(token_ids)
        if hasattr(confidences, 'tolist'):
            confidences = confidences.tolist()
        text = []
        confs = []
        for token, conf in zip(token_ids, confidences):
            codepoint = self.codepoint_for_token(token)
            if codepoint is None:
                continue
            text.append(chr(codepoint))
            confs.append(float(conf))
        return ''.join(text), confs, lang_ids

    def save(self) -> dict[str, Any]:
        """
        Serializes tokenizer state for dataset metadata and checkpoints.
        """
        return {'version': 1,
                'frozen': self._frozen,
                'codepoints': self._idx_to_codepoint}

    @classmethod
    def load(cls, state: Optional[dict[str, Any]]) -> 'CodePointTokenizer':
        if not state:
            return cls()
        codepoints = state.get('codepoints', [])
        frozen = bool(state.get('frozen', True))
        return cls(codepoints=codepoints, frozen=frozen)

    @classmethod
    def merge(cls, tokenizers: Iterable['CodePointTokenizer']) -> 'CodePointTokenizer':
        codepoints = []
        seen: set[int] = set()
        for tokenizer in tokenizers:
            for codepoint in tokenizer.codepoints:
                if codepoint in seen:
                    continue
                seen.add(codepoint)
                codepoints.append(codepoint)
        return cls(codepoints=codepoints, frozen=True)


# Backwards-compatible alias.
OctetTokenizer = CodePointTokenizer
