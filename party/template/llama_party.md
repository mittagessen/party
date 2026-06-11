---
summary: Pretrained multilingual Party base model
authors:
  - name: Benjamin Kiessling
    affiliation: ALMAnaCH, Inria
    orcid: 0000-0001-9543-7827
license: Apache-2.0
software_name: kraken
software_hints:
- segmentation=both
- version>=7.0
language:
- ara
- cat
- ces
- chu
- cos
- deu
- dum
- eng
- fas
- fin
- fra
- frm
- grc
- heb
- ita
- kat
- lat
- mal
- nld
- nor
- oci
- ota
- pcd
- por
- rus
- slk
- spa
- swe
- syr
- ukr
- urd
- vie
- yid
script:
- Arab
- Aran
- Cyrl
- Cyrs
- Geor
- Glag
- Grek
- Hebr
- Latf
- Latn
- Mlym
- Syrc
tags:
- automatic-text-recognition
- multilingual
- multiscriptal
- multimodal
model_type:
- recognition
metrics:
  cer: 0.10
base_model:
- https://huggingface.co/timm/swin_base_patch4_window12_384.ms_in22k
- https://huggingface.co/mittagessen/bytellama-43m-cc
datasets:
- https://github.com/ubtue/gt-fraktur
- https://zenodo.org/records/17252677
- https://zenodo.org/records/4095301
- https://github.com/vivianpl/HPGTR
- https://gitlab.huma-num.fr/ecrinum/anthologia/htr_cpgr23
- https://zenodo.org/records/5578251
- https://zenodo.org/records/5595669
- https://zenodo.org/records/5578136
- https://zenodo.org/records/15473122
- https://github.com/MehreenMehreen/muharaf
- https://github.com/calfa-co/rasam-dataset
- https://github.com/calfa-co/tarima
- https://github.com/OpenITI/arabic_ms_data
- https://github.com/OpenITI/arabic_print_data
- https://github.com/FoNDUE-HTR/FONDUE-CA-PRINT-20
- https://github.com/HTRomance-Project/middle-ages-in-spain
- https://github.com/PSL-Chartes-HTR-Students/HN2021-OCR-Poesie-Corse
- https://github.com/HTR-School-Vienna/2024--medieval-czech
- https://github.com/HTR-School-Vienna/2023--medieval-czech
- https://github.com/HTR-School-Vienna/2025--medieval-czech
- https://zenodo.org/records/7467034
- https://zenodo.org/records/11209325
- https://doi.org/10.5281/zenodo.11191457
- https://zenodo.org/records/13769222
- https://github.com/FoNDUE-HTR/FONDUE-NE-MSS-17-PR
- https://github.com/FoNDUE-HTR/FONDUE-EN-PRINT-20
- https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
- https://zenodo.org/records/4243023
- https://doi.org/10.5281/zenodo.8038689
- https://github.com/sloanelab-org/HTR-Model
- https://americanphilosophicalsociety.github.io/RevCityDocs/
- https://github.com/OCR-D/gt_structure_text
- https://zenodo.org/records/4599472
- https://github.com/LaurieHoeben/Verard-corpus
- https://github.com/banq-dcn/Copiste-d-un-jour
- https://github.com/Front-Justice/corpus-HTR-lignes-mixtes
- https://github.com/alix-tz/dataset-celestine-doniau-danest
- https://github.com/FoNDUE-HTR/FONDUE-FR-MSS-19
- https://github.com/FoNDUE-HTR/FONDUE-FR-MSS-19-PR
- https://github.com/FoNDUE-HTR/FONDUE-FR-PRINT-20
- https://github.com/jpmjpmjpm/genauto-td-htr
- https://github.com/Front-Justice/htr-front-justice
- https://nakala.fr/10.34847/nkl.48ad8b8d
- https://zenodo.org/records/13784411
- https://github.com/alix-tz/moonshines
- https://zenodo.org/records/5654841
- https://github.com/PonteIneptique/valais-recensement
- https://github.com/HTR-United/tapuscorpus
- https://github.com/HTR-United/timeuscorpus
- https://github.com/HisMoDoc-HTR/TitresNobiliaires_17_18
- https://github.com/FoNDUE-HTR/FONDUE-MLT-ART
- https://github.com/ksefil/NuBIS-OCR
- https://github.com/JulianHelmchen/2024--medieval-german
- https://github.com/HTR-School-Vienna/2025--early-modern-german
- https://zenodo.org/records/5153263
- https://github.com/UB-Mannheim/charlottenburger-amtsschrifttum
- https://github.com/Digital-History-Bonn/Chronicling-Germany-Code
- https://github.com/UB-Mannheim/dach-gt
- https://github.com/tboenig/DTGT
- https://zenodo.org/records/15303398
- https://github.com/UB-Mannheim/Fibeln
- https://github.com/FoNDUE-HTR/FONDUE-DE-MSS-16-PR
- https://github.com/FoNDUE-HTR/FONDUE-DE-MSS-18
- https://github.com/FoNDUE-HTR/FONDUE-DE-MSS-19-PR
- https://github.com/FoNDUE-HTR/FONDUE-DE-MSS-20-PR
- https://github.com/FoNDUE-HTR/FONDUE-MLT-PRINT-TEST
- https://github.com/FoNDUE-HTR/FoNDUE_Kunsthistorisches-UZH_Archivdatenbank
- https://zenodo.org/records/3333627
- https://github.com/UB-Mannheim/Hakenkreuzbanner
- https://zenodo.org/records/17978574
- https://zenodo.org/records/7466928
- https://zenodo.org/records/5179361
- https://github.com/bertsky/mkn-kurrent-gt
- https://zenodo.org/records/3387369
- https://zenodo.org/records/13881575
- https://github.com/UB-Mannheim/reichsanzeiger-gt
- https://github.com/UB-Mannheim/Weisthuemer
- https://zenodo.org/records/11046062
- https://github.com/HTR-School-Vienna/2025-hebrew
- https://zenodo.org/records/13760586
- https://github.com/vedph/episearch-htr
- https://github.com/FoNDUE-HTR/FONDUE-IT-PRINT-20
- https://github.com/FoNDUE-HTR/FONDUE-IT-PRINT-20-PR
- https://github.com/HTRogene/italian
- https://github.com/HTRomance-Project/medieval-italian
- https://github.com/Giorgiaagostini/LiDi1.0-project
- https://htr-school-vienna.github.io/2025--late-medieval-latin/
- https://gitlab.rlp.net/adwmainz/projekte/burchards-dekret-digital/data.git
- https://github.com/rescribe/carolineminuscule-groundtruth
- https://github.com/HTR-School-Vienna/2025--Carolingian_Latin-
- https://github.com/HTR-United/CREMMA-Medieval-LAT
- https://zenodo.org/records/4780947
- https://github.com/malamatenia/Eutyches
- https://github.com/FoNDUE-HTR/FONDUE-LA-MSS-16-PR
- https://github.com/FoNDUE-HTR/FONDUE-LA-MSS-17-PR
- https://github.com/FoNDUE-HTR/FONDUE-LA-MSS-MA
- https://github.com/FoNDUE-HTR/FONDUE-LA-PRINT-16
- https://github.com/HTR-School-Vienna/2023--late-medieval-latin
- https://github.com/HTR-School-Vienna/2024--late-medieval-latin
- https://github.com/HTRogene/latin
- https://github.com/HTRomance-Project/medieval-latin
- https://dl.acm.org/doi/abs/10.1007/978-3-032-04624-6_20
- https://github.com/parisbible/ground_truth
- https://zenodo.org/records/7537204
- https://doi.org/10.11588/data/L2KRZO
- https://zenodo.org/records/10005366
- https://github.com/HTR-United/cremma-medieval
- https://github.com/PSL-Chartes-HTR-Students/HN2021-Boccace
- https://github.com/Gallicorpora/HTR-imprime-16e-siecle
- https://github.com/Gallicorpora/HTR-incunable-15e-siecle
- https://github.com/Gallicorpora/HTR-MSS-15e-Siecle
- https://github.com/Gallicorpora/HTR-imprime-18e-siecle
- https://github.com/CIHAM-HTR/Fabliaux
- https://github.com/FoNDUE-HTR/FONDUE-FR-AAEB-16
- https://github.com/FoNDUE-HTR/FONDUE-FR-AAEB-17
- https://github.com/FoNDUE-HTR/FONDUE-FR-MSS-18
- https://github.com/FoNDUE-HTR/FONDUE-FR-PRINT-16
- https://github.com/FoNDUE-HTR/FONDUE-FR-PRINT-17
- https://github.com/SETAFDH/HTR-SETAF-Jean-Michel
- https://github.com/SETAFDH/HTR-SETAF-LesFaictzJCH
- https://github.com/SETAFDH/HTR-SETAF-Pierre-de-Vingle
- https://github.com/HTRogene/french
- https://github.com/HTRomance-Project/medieval-french
- https://github.com/Gallicorpora/HTR-imprime-17e-siecle
- https://github.com/CIHAM-HTR/Liber
- https://github.com/e-ditiones/OCR17
- https://github.com/PSL-Chartes-HTR-Students/TNAH-2021-DecameronFR
- https://github.com/LaBretelle/transcription-chastel
- https://doi.org/10.5281/zenodo.15030337
- https://zenodo.org/records/10255840
- https://github.com/HTRogene/occitan
- https://github.com/Arch-W/iForal-Dataset
- https://zenodo.org/records/13986218
- https://zenodo.org/records/11218527
- https://github.com/DesenrollandoElCordel/FoNDUE-Spanish-chapbooks-Dataset
- https://github.com/FoNDUE-HTR/FONDUE-ES-MSS-19-PR
- https://github.com/FoNDUE-HTR/FONDUE-ES-PRINT-19
- https://github.com/Proyecto-Ocupacion-Araucania-UChile/HTR_Araucania_XIX
- https://github.com/HTRogene/spanish
- https://zenodo.org/records/1322666
- https://zenodo.org/records/3945088
- https://zenodo.org/records/14679534
- https://zenodo.org/records/4599624
- https://huggingface.co/collections/Riksarkivet/training-data-for-swedish-lion-libre
- https://doi.org/10.5281/zenodo.18157525
- https://zenodo.org/records/14714089 
---
# Llama Party

Party is *pa*ge-wise *r*ecognition of *t*ext-*y*. It is a replacement for
conventional text recognizers in automatic text recognition pipelines that
utilize either bounding box or baseline+bounding polygon segmentation methods
for layout analysis.

Llama party is a full-page generative text recognizer that has been pretrained
on a large corpus of multilingual historical, contemporary, and born-digital
document page images, both handwritten and machine-printed.

## Architecture

The recognizer is a deep fusion multimodal model consisting of a Swin vision
encoder and a tiny Llama (40m parameters) decoder trained with octet
tokenization. The network is prompted with the line positions through
positional embeddings added to the encoder hidden state. Line prompts can be
encoded either as bounding boxes or as baseline+bounding curves, so the model
is compatible with both segmentation paradigms.

During training the encoder weights were initialized with an ImageNet-22k
pretrained Swin-base from pytorch-image-models, the decoder weights came from a
custom [Llama 3.2](https://github.com/mittagessen/bytellama) pretrained on the
historical and cultural subset of the [Common
Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) tokenized with a
ByT5-style octet tokenizer.

## Uses

Llama party is a recognition foundation model primarily targeted at automatic
text recognition for the humanities. While it produces fairly accurate output
on an impressive range of material it is intended to be fine-tuned on some
target dataset to ensure compliance with desired transcription guidelines.

## Transcription guidelines, Normalization, and Transformations

No attempts have been made to normalize the datasets or to only use data
adhering to common transcription guidelines. While some subsets of the corpus
are internally consistent, only a very small subset of the languages in the
training data only contain datasets from a single source.

Text was normalized to Unicode NFD and whitespace was normalized during
training.

## Bias, Risks, and Limitations

The training corpus is heavily skewed towards a couple of languages (English,
French, German, Latin, and Middle French) and frequently incorporates datasets
of esoteric material transcribed for specific purposes. Especially
machine-printed and born-digital material lack diversity, so error rates will
most likely vary considerably across languages and document type.

Some additional limitations are to be expected:

- For a number of languages the training material is small and/or comes from a
  single source (e.g. Ancient Greek, Finnish, Georgian, Malayalam, Slovak),
  resulting in markedly higher error rates that will require fine-tuning.
- Some transcriptions resolved abbreviations while others did not. Inconsistent
  output is to be expected, in particular for European manuscripts in Latin
  script.
- As the model predicts 8-bit UTF-8 code units directly the lack of consistent
  Unicode normalization can cause slightly different code point streams during
  prediction.

## How to Get Started with the Model

Install `kraken` through pip and the `party` package from
[github](https://github.com/mittagessen/party) and follow the instructions.
Inference is handled through kraken:

```
$ kraken -f alto -a -i input.xml out.xml ocr -m -B 32 ~/model.safetensors
```

## Training Details

### Training Data

The model has been pretrained on a large collection of publicly available ATR
datasets, in addition to a number of restricted datasets.

|Language|Datasets|
|---|---|
|Ancient Greek|[EPARCHOS](https://zenodo.org/records/4095301)<br>[HPGTR](https://github.com/vivianpl/HPGTR)<br>[HTR_CPgr23](https://gitlab.huma-num.fr/ecrinum/anthologia/htr_cpgr23)<br>[Stavronikita Monastery Greek Handwritten Document Collection No. 114](https://zenodo.org/records/5578251)<br>[Stavronikita Monastery Greek Handwritten Document Collection No. 53](https://zenodo.org/records/5595669)<br>[Stavronikita Monastery Greek Handwritten Document Collection No. 79](https://zenodo.org/records/5578136)<br>11 private datasets|
|Arabic|[Agapet](https://zenodo.org/records/15473122)<br>[Muharaf: Manuscripts of Handwritten Arabic Dataset](https://github.com/MehreenMehreen/muharaf)<br>[RASAM dataset](https://github.com/calfa-co/rasam-dataset)<br>[TariMa](https://github.com/calfa-co/tarima)<br>[arabic_ms_data](https://github.com/OpenITI/arabic_ms_data)<br>[OpenITI Arabic Print Data](https://github.com/OpenITI/arabic_print_data)|
|Catalan|[FONDUE-CA-PRINT-20](https://github.com/FoNDUE-HTR/FONDUE-CA-PRINT-20)<br>[htromance-spa](https://github.com/HTRomance-Project/middle-ages-in-spain)<br>1 private dataset|
|Church Slavonic|3 private datasets|
|Corsican|[OCR Corse](https://github.com/PSL-Chartes-HTR-Students/HN2021-OCR-Poesie-Corse)|
|Czech|[2024--medieval-czech-main](https://github.com/HTR-School-Vienna/2024--medieval-czech)<br>[2023--medieval-czech](https://github.com/HTR-School-Vienna/2023--medieval-czech)<br>[HTR Winter School 2025 - Medieval Czech - Biblioteka Jagiellonska BJ Rkp 441 IV](https://github.com/HTR-School-Vienna/2025--medieval-czech)<br>[Paderov Bible handwriting ground truth](https://zenodo.org/records/7467034)|
|Dutch|[6000 ground truth of VOC and notarial deeds / HTR of VOC, WIC and notarial deeds](https://zenodo.org/records/11209325)<br>[ARletta](https://doi.org/10.5281/zenodo.11191457)<br>[Dagboek Ernest Clarysse](https://zenodo.org/records/13769222)<br>[FONDUE-NE-MSS-17-PR](https://github.com/FoNDUE-HTR/FONDUE-NE-MSS-17-PR)|
|English|[FONDUE-EN-PRINT-20](https://github.com/FoNDUE-HTR/FONDUE-EN-PRINT-20)<br>[IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)<br>[jcrs_train](https://zenodo.org/records/4243023)<br>[JosephHookerHTR](https://doi.org/10.5281/zenodo.8038689)<br>[sloanelab](https://github.com/sloanelab-org/HTR-Model)<br>[The Revolutionary City / RevCity documentation](https://americanphilosophicalsociety.github.io/RevCityDocs/)<br>[OCR-D gt_structure_text](https://github.com/OCR-D/gt_structure_text)<br>2 private datasets|
|Finnish|[NewsEye/READ OCR Finnish Newspapers](https://zenodo.org/records/4599472)|
|French|[Antoine Verard extracts](https://github.com/LaurieHoeben/Verard-corpus)<br>[Copiste-d-un-jour](https://github.com/banq-dcn/Copiste-d-un-jour)<br>[corpus-HTR-lignes-mixtes](https://github.com/Front-Justice/corpus-HTR-lignes-mixtes)<br>[dataset-celestine-doniau-danest](https://github.com/alix-tz/dataset-celestine-doniau-danest)<br>[FONDUE-FR-MSS-16](https://github.com/FoNDUE-HTR/FONDUE-FR-MSS-19)<br>[FONDUE-FR-MSS-19-PR](https://github.com/FoNDUE-HTR/FONDUE-FR-MSS-19-PR)<br>[FONDUE-FR-PRINT-20](https://github.com/FoNDUE-HTR/FONDUE-FR-PRINT-20)<br>[genauto-td-htr](https://github.com/jpmjpmjpm/genauto-td-htr)<br>[HTR Front Justice](https://github.com/Front-Justice/htr-front-justice)<br>[La Correspondance Doucet-Rene Jean](https://nakala.fr/10.34847/nkl.48ad8b8d)<br>[Memoire sur St Domingue par H. M. Michel](https://zenodo.org/records/13784411)<br>[Moonshines](https://github.com/alix-tz/moonshines)<br>[NewsEye READ AS French Newspapers](https://zenodo.org/records/5654841)<br>[Recensement Valaisan (Valais Time Machine)](https://github.com/PonteIneptique/valais-recensement)<br>[Tapus Corpus](https://github.com/HTR-United/tapuscorpus)<br>[TIMEUS Corpus](https://github.com/HTR-United/timeuscorpus)<br>[TitresNobiliaires_17_18](https://github.com/HisMoDoc-HTR/TitresNobiliaires_17_18)<br>[FONDUE-MLT-ART](https://github.com/FoNDUE-HTR/FONDUE-MLT-ART)<br>[NuBIS-OCR](https://github.com/ksefil/NuBIS-OCR)<br>1 private dataset|
|Georgian|15 private datasets|
|German|[gt-fraktur](https://github.com/ubtue/gt-fraktur)<br>[german_kurrent_handwritten_text_lines](https://zenodo.org/records/17252677)<br>[OCR-D gt_structure_text](https://github.com/OCR-D/gt_structure_text)<br>[FONDUE-MLT-ART](https://github.com/FoNDUE-HTR/FONDUE-MLT-ART)<br>[2024--German-Group-main](https://github.com/JulianHelmchen/2024--medieval-german)<br>[2025--Early-Modern-German](https://github.com/HTR-School-Vienna/2025--early-modern-german)<br>[Bullinger Digital Gwalther handwriting ground truth](https://zenodo.org/records/5153263)<br>[charlottenburger-amtsschrifttum](https://github.com/UB-Mannheim/charlottenburger-amtsschrifttum)<br>[Chronicling Germany](https://github.com/Digital-History-Bonn/Chronicling-Germany-Code)<br>[dach-gt](https://github.com/UB-Mannheim/dach-gt)<br>[DigiTheo Ground Truth](https://github.com/tboenig/DTGT)<br>[Dresdner Hofdiarium](https://zenodo.org/records/15303398)<br>[Fibeln](https://github.com/UB-Mannheim/Fibeln)<br>[FONDUE-DE-MSS-16-PR](https://github.com/FoNDUE-HTR/FONDUE-DE-MSS-16-PR)<br>[FONDUE-DE-MSS-18](https://github.com/FoNDUE-HTR/FONDUE-DE-MSS-18)<br>[FONDUE-DE-MSS-19-PR](https://github.com/FoNDUE-HTR/FONDUE-DE-MSS-19-PR)<br>[FONDUE-DE-MSS-20-PR](https://github.com/FoNDUE-HTR/FONDUE-DE-MSS-20-PR)<br>[FONDUE-MLT-PRINT-TEST](https://github.com/FoNDUE-HTR/FONDUE-MLT-PRINT-TEST)<br>[FoNDUE_Kunsthistorisches-UZH_Archivdatenbank](https://github.com/FoNDUE-HTR/FoNDUE_Kunsthistorisches-UZH_Archivdatenbank)<br>[Ground truth for Neue Zurcher Zeitung black letter](https://zenodo.org/records/3333627)<br>[Hakenkreuzbanner](https://github.com/UB-Mannheim/Hakenkreuzbanner)<br>[inzigkofen](https://zenodo.org/records/17978574)<br>[Klosterneuburg, Stiftsbibl., Cod. 48](https://zenodo.org/records/7466928)<br>[koenigsfelden](https://zenodo.org/records/5179361)<br>[mkn-kurrent-gt](https://github.com/bertsky/mkn-kurrent-gt)<br>[NewsEye / READ OCR Austrian Newspapers](https://zenodo.org/records/3387369)<br>[nuremberg_letterbooks](https://zenodo.org/records/13881575)<br>[reichsanzeiger-gt](https://github.com/UB-Mannheim/reichsanzeiger-gt)<br>[Weisthuemer](https://github.com/UB-Mannheim/Weisthuemer)<br>[Training Data Incunabula Reichenau](https://zenodo.org/records/11046062)<br>1 private dataset|
|German (shorthand)|1 private dataset|
|Hebrew|[2025-hebrew](https://github.com/HTR-School-Vienna/2025-hebrew)<br>2 private datasets|
|Italian|[Diario del Soldato Bruno Celestino](https://zenodo.org/records/13760586)<br>[EpiSearch HTR](https://github.com/vedph/episearch-htr)<br>[FONDUE-IT-PRINT-20](https://github.com/FoNDUE-HTR/FONDUE-IT-PRINT-20)<br>[FONDUE-IT-PRINT-20-PR](https://github.com/FoNDUE-HTR/FONDUE-IT-PRINT-20-PR)<br>[HTRogène Medieval Italian Manuscripts](https://github.com/HTRogene/italian)<br>[HTRomance, Medieval Italian corpus of ground-truth for Handwritten Text Recognition](https://github.com/HTRomance-Project/medieval-italian)<br>[LiDi1.0-project](https://github.com/Giorgiaagostini/LiDi1.0-project)<br>2 private datasets|
|Latin|[OCR-D gt_structure_text](https://github.com/OCR-D/gt_structure_text)<br>[NuBIS-OCR](https://github.com/ksefil/NuBIS-OCR)<br>[Training Data Incunabula Reichenau](https://zenodo.org/records/11046062)<br>[2025--late-medieval-latin-main](https://htr-school-vienna.github.io/2025--late-medieval-latin/)<br>[burchards-dekret-digital](https://gitlab.rlp.net/adwmainz/projekte/burchards-dekret-digital/data.git)<br>[Caroline Minuscule ground truth](https://github.com/rescribe/carolineminuscule-groundtruth)<br>[Carolingian Latin Group HTR Wien Winter School 2025](https://github.com/HTR-School-Vienna/2025--Carolingian_Latin-)<br>[CREMMA Medii Aevi](https://github.com/HTR-United/CREMMA-Medieval-LAT)<br>[DISTINGUO Latin ground truth](https://zenodo.org/records/4780947)<br>[Eutyches](https://github.com/malamatenia/Eutyches)<br>[FONDUE-LA-MSS-16-PR](https://github.com/FoNDUE-HTR/FONDUE-LA-MSS-16-PR)<br>[FONDUE-LA-MSS-17-PR](https://github.com/FoNDUE-HTR/FONDUE-LA-MSS-17-PR)<br>[FONDUE-LA-MSS-MA](https://github.com/FoNDUE-HTR/FONDUE-LA-MSS-MA)<br>[FONDUE-LA-PRINT-16](https://github.com/FoNDUE-HTR/FONDUE-LA-PRINT-16)<br>[HTR Winter School 2023/2024 - Late Medieval Latin, ONB 3891](https://github.com/HTR-School-Vienna/2023--late-medieval-latin)<br>[HTR Winter School 2024/2025 - Late Medieval Latin, ONB 4135; ONB 4680](https://github.com/HTR-School-Vienna/2024--late-medieval-latin)<br>[HTRogène Medieval Latin Manuscripts](https://github.com/HTRogene/latin)<br>[HTRomance, Medieval Latin corpus of ground-truth for Handwritten Text Recognition](https://github.com/HTRomance-Project/medieval-latin)<br>[notarial_charter](https://dl.acm.org/doi/abs/10.1007/978-3-032-04624-6_20)<br>[Paris Bible Project](https://github.com/parisbible/ground_truth)<br>[Wien ONB Cod 2160 ground truth](https://zenodo.org/records/7537204)|
|Malayalam|[Ground Truth data for printed Malayalam](https://doi.org/10.11588/data/L2KRZO)|
|Middle Dutch|[data](https://zenodo.org/records/10005366)|
|Middle French|[Cremma Medieval](https://github.com/HTR-United/cremma-medieval)<br>[De la généalogie des dieux](https://github.com/PSL-Chartes-HTR-Students/HN2021-Boccace)<br>[Données imprimés du 16e siècle](https://github.com/Gallicorpora/HTR-imprime-16e-siecle)<br>[Données HTR incunables du 15e siècle](https://github.com/Gallicorpora/HTR-incunable-15e-siecle)<br>[Données HTR manuscrits du 15e siècle](https://github.com/Gallicorpora/HTR-MSS-15e-Siecle)<br>[Données imprimés du 18e siècle](https://github.com/Gallicorpora/HTR-imprime-18e-siecle)<br>[Fabliaux](https://github.com/CIHAM-HTR/Fabliaux)<br>[FONDUE-FR-AAEB-16](https://github.com/FoNDUE-HTR/FONDUE-FR-AAEB-16)<br>[FONDUE-FR-AAEB-17](https://github.com/FoNDUE-HTR/FONDUE-FR-AAEB-17)<br>[FONDUE-FR-MSS-18](https://github.com/FoNDUE-HTR/FONDUE-FR-MSS-18)<br>[FONDUE-FR-PRINT-16](https://github.com/FoNDUE-HTR/FONDUE-FR-PRINT-16)<br>[FONDUE-FR-PRINT-17](https://github.com/FoNDUE-HTR/FONDUE-FR-PRINT-17)<br>[HTR-SETAF-Jean-Michel](https://github.com/SETAFDH/HTR-SETAF-Jean-Michel)<br>[HTR-SETAF-LesFaictzJCH](https://github.com/SETAFDH/HTR-SETAF-LesFaictzJCH)<br>[HTR-SETAF-Pierre-de-Vingle](https://github.com/SETAFDH/HTR-SETAF-Pierre-de-Vingle)<br>[HTRogene French](https://github.com/HTRogene/french)<br>[HTRomance, Medieval French corpus of ground-truth for Handwritten Text Recognition](https://github.com/HTRomance-Project/medieval-french)<br>[Imprimés 17e siècle](https://github.com/Gallicorpora/HTR-imprime-17e-siecle)<br>[Liber](https://github.com/CIHAM-HTR/Liber)<br>[OCR17plus](https://github.com/e-ditiones/OCR17)<br>[TNAH-2021-DecameronFR](https://github.com/PSL-Chartes-HTR-Students/TNAH-2021-DecameronFR)<br>[transcription-chastel](https://github.com/LaBretelle/transcription-chastel)|
|Multilingual|[Training Data Incunabula Reichenau](https://zenodo.org/records/11046062)<br>[TranscriboQuest25_MedVernacReligio](https://doi.org/10.5281/zenodo.15030337)|
|Norwegian|[NorHand v3 / Dataset for Handwritten Text Recognition in Norwegian](https://zenodo.org/records/10255840)|
|Occitan|[HTRogène Medieval Occitan Manuscripts](https://github.com/HTRogene/occitan)<br>1 private dataset|
|Ottoman Turkish|[arabic_ms_data](https://github.com/OpenITI/arabic_ms_data)<br>[OpenITI Arabic Print Data](https://github.com/OpenITI/arabic_print_data)|
|Persian|[arabic_ms_data](https://github.com/OpenITI/arabic_ms_data)<br>[OpenITI Arabic Print Data](https://github.com/OpenITI/arabic_print_data)|
|Picard|1 private dataset|
|Portuguese|[iForal-Dataset](https://github.com/Arch-W/iForal-Dataset)<br>[Portuguese Handwriting 16th-19th c.](https://zenodo.org/records/13986218)|
|Russian|2 private datasets|
|Slovak|[Slovensky Supermodel P&T1](https://zenodo.org/records/11218527)|
|Spanish|[htromance-spa](https://github.com/HTRomance-Project/middle-ages-in-spain)<br>[FoNDUE Spanish chapbooks 19th c. Dataset](https://github.com/DesenrollandoElCordel/FoNDUE-Spanish-chapbooks-Dataset)<br>[FONDUE-ES-MSS-19-PR](https://github.com/FoNDUE-HTR/FONDUE-ES-MSS-19-PR)<br>[FONDUE-ES-PRINT-19](https://github.com/FoNDUE-HTR/FONDUE-ES-PRINT-19)<br>[HTR - Araucania manuscript XIX](https://github.com/Proyecto-Ocupacion-Araucania-UChile/HTR_Araucania_XIX)<br>[HTRogène Medieval Spanish Manuscripts](https://github.com/HTRogene/spanish)<br>[ohg](https://zenodo.org/records/1322666)<br>3 private datasets|
|Swedish|[Finnish Court Records-sub500](https://zenodo.org/records/3945088)<br>[kat57 Swedish ground truth dataset](https://zenodo.org/records/14679534)<br>[NewsEye / READ OCR training dataset from Swedish Newspapers](https://zenodo.org/records/4599624)<br>[riskarchiv](https://huggingface.co/collections/Riksarkivet/training-data-for-swedish-lion-libre)|
|Syriac|[zenodo.18157525](https://doi.org/10.5281/zenodo.18157525)<br>[winter_school_vienna](https://zenodo.org/records/14714089)<br>2 private datasets|
|Ukrainian|1 private dataset|
|Urdu|[OpenITI Arabic Print Data](https://github.com/OpenITI/arabic_print_data)|
|Vietnamese|1 private dataset|
|Yiddish|7 private datasets|

### Training Procedure and Hyperparameters

- **Training regime:** 5 × A40 GPU, BF16 mixed precision, Muon optimizer
  (with an auxiliary AdamW group for embeddings/heads), batch size 16, gradient
  accumulation 16, cosine schedule with 10000 iteration warmup, max LR 3e-4,
  min LR 1e-5, weight decay 1e-5, gradient clipping 1.0, NFD normalization,
  augmentation enabled, random sampling of bounding box and curve line prompts.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model was evaluated on held-out test splits drawn from the training data
sources.

#### Metrics

|Language|CER|
|---|---|
|Ancient Greek|40.76%|
|Arabic|7.34%|
|Catalan|5.92%|
|Church Slavonic|7.30%|
|Corsican|7.48%|
|Czech|11.03%|
|Dutch|8.14%|
|English|7.63%|
|Finnish|38.16%|
|French|7.49%|
|Georgian|20.70%|
|German|9.10%|
|Hebrew|12.23%|
|Italian|2.81%|
|Latin|7.47%|
|Malayalam|39.53%|
|Middle Dutch|13.31%|
|Middle French|1.47%|
|Multilingual|6.27%|
|Norwegian|1.51%|
|Ottoman Turkish|6.77%|
|Persian|5.41%|
|Portuguese|7.53%|
|Russian|7.99%|
|Shorthand German|4.22%|
|Slovak|19.59%|
|Spanish|6.07%|
|Swedish|6.10%|
|Syriac|5.13%|
|Ukrainian|8.21%|
|Urdu|3.91%|
|Yiddish|1.79%|
|**Aggregate (micro-average)**|**10.16%**|

## Citation
