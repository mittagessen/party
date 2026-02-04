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

from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, Tuple

from lxml import etree

from party.tokenizer import LANG_TO_ISO
from party.cli.set_lang import _detect_filetype, _set_lang_alto, _set_lang_page


SUFFIXES = {'.xml', '.alto'}


def _process_file(args: Tuple[str, str]) -> Tuple[str, bool]:
    """Process a single XML file, setting its language attribute.

    Returns a tuple of (file path string, success boolean).
    """
    fpath, iso_code = args
    try:
        doc = etree.parse(fpath)
    except etree.XMLSyntaxError:
        return (fpath, False)
    try:
        filetype = _detect_filetype(doc)
    except ValueError:
        return (fpath, False)
    if filetype == 'alto':
        _set_lang_alto(doc, iso_code)
    elif filetype == 'page':
        _set_lang_page(doc, iso_code)
    with open(fpath, 'wb') as fo:
        fo.write(etree.tostring(doc, encoding='UTF-8', xml_declaration=True))
    return (fpath, True)


def _iter_tasks(root: Path) -> Iterator[Tuple[str, str]]:
    """Lazily yield (file path, iso code) tasks by traversing lang directories."""
    for lang_dir in sorted(root.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang_name = lang_dir.name
        if lang_name not in LANG_TO_ISO:
            click.echo(f'Skipping unknown language directory: {lang_name}', err=True)
            continue
        iso_code = LANG_TO_ISO[lang_name]
        for fpath in lang_dir.rglob('*'):
            if fpath.is_file() and fpath.suffix.lower() in SUFFIXES:
                yield (str(fpath), iso_code)


@click.command()
@click.argument('root', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('-j', '--workers', default=None, type=int,
              help='Number of worker processes (default: number of CPUs)')
def cli(root, workers):
    """Sets language on ALTO/PageXML files in subdirectories named by language.

    ROOT is a directory containing subdirectories named after languages
    (e.g. 'english', 'middle_french') with ALTO or PageXML files inside.
    """
    lang_file_counts = {}

    with Pool(processes=workers) as pool:
        for fpath_str, success in pool.imap_unordered(_process_file, _iter_tasks(root)):
            lang_name = Path(fpath_str).relative_to(root).parts[0]
            iso_code = LANG_TO_ISO[lang_name]
            if lang_name not in lang_file_counts:
                lang_file_counts[lang_name] = (iso_code, 0, 0)
            _, ok, fail = lang_file_counts[lang_name]
            if success:
                lang_file_counts[lang_name] = (iso_code, ok + 1, fail)
            else:
                lang_file_counts[lang_name] = (iso_code, ok, fail + 1)
                click.echo(f'Skipping {fpath_str}: malformed or unknown XML format', err=True)

    for lang_name in sorted(lang_file_counts):
        iso_code, ok, fail = lang_file_counts[lang_name]
        msg = f'{lang_name} ({iso_code}): processed {ok} file(s)'
        if fail:
            msg += f', {fail} skipped'
        click.echo(msg)


if __name__ == '__main__':
    cli()
