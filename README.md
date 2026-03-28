# party

party is **PA**ge-wise **R**ecognition of **T**ext-**y**. It is a replacement for conventional text recognizers in ATR system using conventional baseline+bounding polygon (where it eliminates the need for bounding polygons) and bounding box line data models. 

Party consists of a Swin vision transformer encoder, baseline positional embeddings, and a [tiny Llama decoder](https://github.com/mittagessen/bytellama) trained on octet tokenization.

![party model functional block diagram](/images/party_diag.png)

The last iteration of party introduced language tokens to permit steering the model's output toward a specific language on a per-line level during inference. Their use is optional but recommended on difficult material to prevent random switching between languages. When no language tag is given the model will generate one or more tokens indicating which languages it thinks are contained in an individual line.

The new base model with language token support is currently embargoed but a model fine-tuned for a number of European languages is available [here](https://zenodo.org/records/15764161).

While the model performs quite well on languages and scripts that are commonly found in the training data, **it is generally expected that it requires fine-tuning for practical use, in particular to ensure alignment with desired transcription guidelines.**

## Installation

    $ pip install .

## Fine Tuning

Party needs to be trained on datasets precompiled from PageXML or ALTO files containing line-wise transcriptions and baseline information for each line. The binary dataset format is **NOT** compatible with kraken (and pre-language token datasets need to be recompiled) but the process of compilation is fairly similar:

        $ party compile -o dataset.arrow *.xml

The current compilation code determines the language on a page level by traversing its path upward until a path component matches a known language identifier, e.g. `/english/path/to/german/dir/with/file.xml` would be assigned `German` as a language because of `german` being the first path component matching a language string in the absolute path. The help screen of `party compile` will print a list of known language identifiers. Files which cannot be assigned a known language will have the `undetermined` value assigned. New ones can be added in `party/tokenizer.py` on request. A better solution parsing out ALTO and PageXML language attributes on the line level is in the works. 

To fine-tune the pretrained base model dataset files in listed in manifest files on all available GPUs:

        $ party --workers 32 train --load 10.5281/zenodo.15764161 -t train.lst -e val.lst

With the default parameters both baseline and bounding box prompts are randomly sampled from the training data. It is suggested that you fine-tune the model with uni-modal line embeddings by only selecting the line format that your segmentation method produces, i.e.:

        $ party train --load 10.5281/zenodo.15764161 -t train.lst -e val.lst --prompt-mode curves

or:

        $ party train --load 10.5281/zenodo.15764161 -t train.lst -e val.lst --prompt-mode boxes

To continue training from an existing checkpoint:

        $ party train --load checkpoint_03-0.0640.ckpt -t train.lst -e val.lst

## Checkpoint conversion

Checkpoints need to be converted into a safetensors format before being usable for inference and testing.

        $  party convert -o model.safetensors checkpoint.ckpt

## Inference

Inference and teseting requires a working [kraken](https://kraken.re) installation.

To recognize text in pre-segmented page images in PageXML or ALTO with the pretrained model run:

        $ party -d cuda:0 ocr -i in.xml out.xml --load-from-repo 10.5281/zenodo.15764161 

The paths to the image file(s) is automatically extracted from the XML input file(s).

When no language tag is given the model will predict by itself which languages are found in it. To explicitly prompt the model to predict a particular language use one of the identifiers below:

|Language|Identifier|
|--------|----------|
|Ancient Greek|grc|
|Catalan|cat|
|Church Slavonic|chu|
|Corsican|cos|
|Czech|ces|
|Dutch|nld|
|English|eng|
|Finnish|fin|
|French|fra|
|German|deu|
|German Shorthand|qaa|
|Irish|gle|
|Italian|ita|
|Latin|lat|
|Latvian|lav|
|Lithuanian|lit|
|Middle Dutch|dum|
|Middle French|frm|
|Newari|new|
|Norwegian|nor|
|Occitan|oci|
|Persian|fas|
|Picard|pcd|
|Polish|pol|
|Portuguese|por|
|Romanian|ron|
|Russian|rus|
|Serbian (cyrillic)|qab|
|Slovenian|slv|
|Spanish|spa|
|Swedish|swe|
|Ukrainian|ukr|

with the `-l` option of the `ocr` subcomand:

    $ party -d cuda:0 ocr -i in.xml out.xml --load-from-repo 10.5281/zenodo.15764161 -l grc

A single language can be defined per call.

When the recognizer supports both curves and box prompts, curves are selected by default. To select a prompt type explicitly you can use the `--curves` and `--boxes` switches:

        $ party -d cuda:0 ocr -i in.xml out.xml --curves --compile
        $ party -d cuda:0 ocr -i in.xml out.xml --boxes --compile

Inference from a converted checkpoint:

        $ party -d cuda:0 ocr -i in.xml out.xml --curves --load-from-file model.safetensors

## Testing

Testing for now only works from binary dataset files. As with for inference curve prompts are selected if the model supports both, but an explicit line prompt type can be selected.

        $  party -d cuda:0 test --curves --load-from-file arabic.safetensors  */*.arrow
        $  party -d cuda:0 test --boxes --load-from-file arabic.safetensors  */*.arrow
        $  party -d cuda:0 test --curves --load-from-repo 10.5281/zenodo.15764161 */*.arrow
        $  party -d cuda:0 test --boxes --load-from-repo 10.5281/zenodo.15764161 */*.arrow

## Performance

Training and inference resource consumption is highly dependent on various optimizations being enabled. Torch compilation which is required for various attention optimizations is enabled per default but lower precision training which isn't supported on CPU needs to be configured manually with `party --precision bf16-mixed ...`. It is possible to reduce training memory requirements substantially by freezing the visual encoder with the `--freeze-encoder` option.

Moderate speedups on CPU are possible with intra-op parallelism (`party --threads 4 ocr ...`).

Quantization isn't yet supported.
