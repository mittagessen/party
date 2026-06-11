# party

party is **PA**ge-wise **R**ecognition of **T**ext-**y**. It is a replacement for conventional text recognizers in ATR system using conventional baseline+bounding polygon (where it eliminates the need for bounding polygons) and bounding box line data models. 

Party consists of a Swin vision transformer encoder, baseline positional embeddings, and a [tiny Llama decoder](https://github.com/mittagessen/bytellama) trained on octet tokenization.

![party model functional block diagram](/images/party_diag.png)

The last iteration of party introduced language tokens to permit steering the model's output toward a specific language on a per-line level during inference. Their use is optional but recommended on difficult material to prevent random switching between languages. When no language tag is given the model will generate one or more tokens indicating which languages it thinks are contained in an individual line.

The base model with language token support is available [here](https://doi.org/10.5281/zenodo.14616980).

While the model performs quite well on languages and scripts that are commonly found in the training data, **it is generally expected that it requires fine-tuning for practical use, in particular to ensure alignment with desired transcription guidelines.**

## Installation

    $ pip install .

## Fine Tuning

Party needs to be trained on datasets precompiled from PageXML or ALTO files containing line-wise transcriptions and baseline information for each line. The binary dataset format is **NOT** compatible with kraken (and pre-language token datasets need to be recompiled) but the process of compilation is fairly similar:

        $ party compile -o dataset.arrow *.xml

The language tokens are derived from line-level language metadata in the XML files. If your files do not contain any (most don't) the token will be set to `und`. There is a helper subcommand that can be used to rewrite files and set languages for every line:

        $ party set-lang ces *.xml

To fine-tune the pretrained base model dataset files in listed in manifest files on all available GPUs:

        $ party train --load-from-repo 10.5281/zenodo.14616980 --workers 32 -t train.lst -e val.lst

With the default parameters both baseline and bounding box prompts are randomly sampled from the training data. It is suggested that you fine-tune the model with uni-modal line embeddings by only selecting the line format that your segmentation method produces, i.e.:

        $ party train --load-from-repo 10.5281/zenodo.14616980 -t train.lst -e val.lst --prompt-mode curves

or:

        $ party train --load-from-repo 10.5281/zenodo.14616980 -t train.lst -e val.lst --prompt-mode boxes

To continue training from an existing weights file or checkpoint:

        $ party train --load-from-file best.safetensors -t train.lst -e val.lst

It is also possible to resume training, restoring all states:

        $ party train --resume checkpoint_03-0.0640.ckpt

## Checkpoint conversion

Checkpoints need to be converted into a safetensors format before being usable for inference and testing.

        $  ketos convert -o model.safetensors checkpoint.ckpt

## Inference

Inference and testing requires a working [kraken](https://kraken.re) installation. It will be installed automatically alongside party.

To recognize text in pre-segmented page images in PageXML or ALTO download the base model and then run inference like usual through kraken:

        $ kraken get 10.5281/zenodo.14616980
        Processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 518.4/518.4 MB 0:00:00 0:02:01
        Model dir: /home/mittagessen/.local/share/htrmopo/c4c16bf2-be07-592d-8486-b9b0ef0ac468 (model files: model.safetensors)
        $ kraken --precision bfloat16 -d cuda:0 -f xml -a -i in.xml out.xml ocr -B 32 -m /home/mittagessen/.local/share/htrmopo/c4c16bf2-be07-592d-8486-b9b0ef0ac468/model.safetensors

The paths to the image file(s) is automatically extracted from the XML input file(s).

## Testing

Testing for now only works from binary dataset files. As with for inference curve prompts are selected if the model supports both, but an explicit line prompt type can be selected.

        $  party -d cuda:0 test --curves --load-from-file arabic.safetensors  */*.arrow
        $  party -d cuda:0 test --boxes --load-from-file arabic.safetensors  */*.arrow
        $  party -d cuda:0 test --curves --load-from-repo 10.5281/zenodo.14616980 */*.arrow
        $  party -d cuda:0 test --boxes --load-from-repo 10.5281/zenodo.14616980 */*.arrow

## Performance

Training and inference resource consumption is highly dependent on various optimizations being enabled. Torch compilation which is required for various attention optimizations is enabled per default but lower precision training which isn't supported on CPU needs to be configured manually with `party --precision bf16-mixed ...`. It is possible to reduce training memory requirements substantially by freezing the visual encoder with the `--freeze-encoder` option.

Moderate speedups on CPU are possible with intra-op parallelism (`party --threads 4 ocr ...`).

Quantization isn't yet supported.
