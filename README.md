# VIST5
VIST5: An Adaptive, Retrieval-Augmented Language Model for Visualization-oriented Dialog

![VIST5 Example](./materials/vist5_example.png)

Demo Video: https://www.youtube.com/watch?v=rmqkxOTIcGo

## Setup

- clone repository from https://github.com/clause-bielefeld/VIST5
- download models and tokenizer from here: https://zenodo.org/uploads/10279066
- unzip each model (vist5, mpnet) and the tokenizer and put all unzipped folders into this directory: /app/src/backend/base_fastapi_backend/src/web_api/models/
- open a terminal
- cd into VIST5/app/src
- run docker-compose up
- open application at http://localhost:8080/vegalite_playground

## BibTeX Citation

If you use VIST5 in a scientific publication, we would appreciate using the following citations:

```

@inproceedings{Voigt2023_EMNLP,
author = {Henrik Voigt and Nuno Carvalhais and Monique Meuschke and Markus Reichstein and Sina Zarriess and Kai Lawonn},
title = {VIST5: An Adaptive, Retrieval-Augmented Language Model for Visualization-oriented Dialog},
booktitle ={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
publisher = {Association for Computational Linguistics},
year = {2023},
pages ={in print},
}

```
