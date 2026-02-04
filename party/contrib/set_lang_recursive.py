#!/usr/bin/env python
"""
Recursively applies `party set-lang` to all ALTO and PageXML files in a
directory tree organized by language name.

Expected directory structure:

    root/
        english/
            file1.xml
            file2.alto
        middle_french/
            file3.xml
        ...

Subdirectory names are natural language names (lower case, spaces replaced
by underscores) as listed in party.tokenizer.LANG_TO_ISO.
"""
import click

from pathlib import Path
from lxml import etree

from party.tokenizer import LANG_TO_ISO
from party.cli.set_lang import _detect_filetype, _set_lang_alto, _set_lang_page


SUFFIXES = {'.xml', '.alto'}


@click.command()
@click.argument('root', type=click.Path(exists=True, file_okay=False, path_type=Path))
def cli(root):
    """Sets language on ALTO/PageXML files in subdirectories named by language.

    ROOT is a directory containing subdirectories named after languages
    (e.g. 'english', 'middle_french') with ALTO or PageXML files inside.
    """
    for lang_dir in sorted(root.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang_name = lang_dir.name
        if lang_name not in LANG_TO_ISO:
            click.echo(f'Skipping unknown language directory: {lang_name}', err=True)
            continue
        iso_code = LANG_TO_ISO[lang_name]
        files = sorted(f for f in lang_dir.rglob('*') if f.is_file() and f.suffix.lower() in SUFFIXES)
        if not files:
            click.echo(f'Skipping {lang_name}: no XML/ALTO files found', err=True)
            continue
        for fpath in files:
            doc = etree.parse(fpath)
            try:
                filetype = _detect_filetype(doc)
            except ValueError:
                click.echo(f'Skipping {fpath}: unknown XML format', err=True)
                continue
            if filetype == 'alto':
                _set_lang_alto(doc, iso_code)
            elif filetype == 'page':
                _set_lang_page(doc, iso_code)
            with open(fpath, 'wb') as fo:
                fo.write(etree.tostring(doc, encoding='UTF-8', xml_declaration=True))
        click.echo(f'{lang_name} ({iso_code}): processed {len(files)} file(s)')


if __name__ == '__main__':
    cli()
