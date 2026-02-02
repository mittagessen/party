"""
Rewrites images stored in JPEG-XL format inside party arrow dataset files
to JPEG, overwriting the original files in place.
"""
import io
import sys

import pillow_jxl  # noqa: F401
import pyarrow as pa
from PIL import Image


def rewrite_arrow_file(path: str) -> None:
    line_struct = pa.struct([('text', pa.string()),
                             ('curve', pa.list_(pa.float32())),
                             ('bbox', pa.list_(pa.float32()))])
    page_struct = pa.struct([('im', pa.binary()),
                             ('lang', pa.string()),
                             ('lines', pa.list_(line_struct))])
    schema = pa.schema([('pages', page_struct)])

    with pa.memory_map(path, 'rb') as source:
        table = pa.ipc.open_file(source).read_all()
        metadata = table.schema.metadata

    new_rows = []
    converted = 0
    for page_scalar in table.column('pages'):
        page = page_scalar.as_py()
        im_bytes = page['im']
        im = Image.open(io.BytesIO(im_bytes))
        fmt = 'JPEG' if im.format == 'JPEG XL' else im.format
        if im.format == 'JPEG XL':
            converted += 1
        im = im.convert('RGB')
        im = im.resize((2560, 1920))
        buf = io.BytesIO()
        im.save(buf, format=fmt, quality=95)
        page['im'] = buf.getvalue()
        new_rows.append(pa.scalar(page, page_struct))

    print(f'{path}: converted {converted}/{len(new_rows)} pages')

    new_table = pa.table([pa.array(new_rows, page_struct)], schema=schema)
    new_table = new_table.replace_schema_metadata(metadata)
    with pa.OSFile(path, 'wb') as sink:
        with pa.ipc.new_file(sink, schema=new_table.schema) as writer:
            for batch in new_table.to_batches():
                writer.write(batch)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <arrow_file> [arrow_file ...]', file=sys.stderr)
        sys.exit(1)
    for path in sys.argv[1:]:
        rewrite_arrow_file(path)
