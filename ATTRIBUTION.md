# Data and model attribution

The default corpus is the English `devtest` split of **FLORES-200**, released
by Meta / the NLLB Team. Cite NLLB Team et al. (2022),
[No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672).

The [upstream FLORES repository](https://github.com/facebookresearch/flores)
licenses FLORES-200 under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Retain attribution, the license link, and an indication of changes when sharing.
The selected corpus is an exact-text subset; trajectory and observation outputs
are machine translations. This project's records distinguish these from originals.
The accessible [DGME mirror](https://huggingface.co/datasets/DGME/FLORES-200)
labels its card CC-BY-4.0; this project conservatively records the upstream
CC-BY-SA-4.0 license instead. No access gate or license acceptance is bypassed.

[facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
is released under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Use this research workflow for noncommercial purposes and retain model attribution.
Consult the original licenses for redistribution and other uses.

Custom corpora remain subject to their own licenses. Set `dataset.local_license`
and retain their attribution alongside the experiment.
