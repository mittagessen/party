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

__all__ = ['CodePointTokenizer']

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

LANG_IDX_TO_ISO = {v: k for k, v in ISO_TO_IDX.items()}
ISO_TO_LANG = {v: k for k, v in LANG_TO_ISO.items()}

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
LANG_OFFSET = 3
CODEPOINT_OFFSET = LANG_OFFSET + len(ISO_TO_IDX)


class CodePointTokenizer(object):
    """
    Dynamic Unicode code point tokenizer.

    Token layout:
    - 0: PAD
    - 1: BOS
    - 2: EOS
    - 3..CODEPOINT_OFFSET-1: language tokens
    - CODEPOINT_OFFSET+: registered Unicode code points

    The tokenizer can be in either a thawed state (new code points are
    registered on demand during encoding) or a frozen state (encoding an
    unseen code point raises). Datasets register their code points at compile
    time; once a model is built around a tokenizer the tokenizer is frozen so
    that the vocabulary is stable.
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
        return LANG_OFFSET + ISO_TO_IDX.get(lang, ISO_TO_IDX['und'])

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
        """
        Encode text into token IDs.

        Args:
            text: The input text to be encoded, unbatched.
            langs: List of lang tokens to insert between BOS and first text token.
            add_bos: Whether to prepend BOS to the input, defaults to True.
            add_eos: Whether to append EOS to the input, defaults to True.

        Returns:
            List[int]: The encoded token IDs.
        """
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
        """Decode a sequence of token IDs into a string and language tags.

        Args:
            ids: The input token IDs to be decoded.

        Returns:
            A tuple containing the decoded text and any language tags in
            the input tensor.
        """
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
        """Decode a sequence of token IDs into a string, with per-code-point
        confidences and language tags.
        """
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
                'codepoints': list(self._idx_to_codepoint)}

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
