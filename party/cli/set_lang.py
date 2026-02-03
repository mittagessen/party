#
# Copyright 2015 Benjamin Kiessling
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
"""
party.cli.set_lang
~~~~~~~~~~~~~~~~~~

Command line driver for setting language attributes on XML lines.
"""
import click
import logging

from lxml import etree
from pathlib import Path
from collections import defaultdict

from party.tokenizer import ISO_TO_IDX, LANG_TO_ISO

from .pred import _parse_page_custom, dict_to_page_custom

logging.captureWarnings(True)
logger = logging.getLogger('party')


def _detect_filetype(doc):
    """
    Detects whether an XML document is ALTO or PageXML.
    """
    root = doc.getroot()
    ns = root.nsmap.get(None, '')
    if 'alto' in ns.lower() or root.tag.lower().endswith('alto'):
        return 'alto'
    elif 'page' in ns.lower() or 'pagecontent' in ns.lower():
        return 'page'
    # fallback: check for known elements
    if doc.find('.//{*}TextLine/{*}String') is not None:
        return 'alto'
    if doc.find('.//{*}TextLine/{*}TextEquiv') is not None:
        return 'page'
    raise ValueError('Unable to determine XML format (not ALTO or PageXML).')


def _set_lang_alto(doc, lang):
    """
    Sets LANG attribute on each TextLine element in an ALTO document.
    """
    lines = doc.findall('.//{*}TextLine')
    for line in lines:
        line.set('LANG', lang)
    return doc


def _set_lang_page(doc, lang):
    """
    Sets language in custom attribute on each TextLine element in a PageXML document.
    """
    lines = doc.findall('.//{*}TextLine')
    for line in lines:
        custom_str = line.get('custom', '')
        cs = _parse_page_custom(custom_str)
        cs['language'] = [{'type': lang}]
        line.set('custom', dict_to_page_custom(cs))
    return doc


@click.command('set-lang',
               epilog=f"""
\b
Language codes known to party:
{chr(10).join(' -> '.join((k.replace("_", " ").title(), v)) for k, v in LANG_TO_ISO.items())}
                       """)
@click.argument('lang', type=click.Choice(list(ISO_TO_IDX.keys())))
@click.argument('files', nargs=-1, required=True,
                type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('-o', '--output', default=None,
              type=click.Path(dir_okay=True, path_type=Path),
              help='Output directory. If not set, files are modified in-place.')
def set_lang(lang, files, output):
    """
    Sets language on every TextLine element in ALTO/PageXML files.

    LANG is a 3-letter ISO 639-3 language code. FILES are one or more
    ALTO or PageXML files to modify.
    """
    for fpath in files:
        with open(fpath, 'rb') as fp:
            doc = etree.parse(fp)
        filetype = _detect_filetype(doc)
        if filetype == 'alto':
            _set_lang_alto(doc, lang)
        elif filetype == 'page':
            _set_lang_page(doc, lang)
        else:
            raise click.ClickException(f'{fpath} has unknown XML format.')

        out_path = fpath
        if output:
            if output.is_dir():
                out_path = output / fpath.name
            else:
                out_path = output

        with open(out_path, 'wb') as fo:
            fo.write(etree.tostring(doc, encoding='UTF-8', xml_declaration=True))
        click.echo(f'Set language to {lang} on {fpath} -> {out_path}')
