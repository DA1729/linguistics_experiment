# Publishing this research repository

No commits, pushes, releases or history changes are performed by the analysis.

Commit the analysis source (`analysis/`), `config/analysis.toml`, `requirements_analysis.txt`, `analyze_local.sh`, tests, README/attribution/publication documentation, and `reports/translation_v1/`. Preserve the existing experiment-generation code and laptop subproject. The report directory contains the full Markdown/standalone HTML report, executive summary, audit, compact CSV tables, figures in PNG/PDF, examples, provenance and validation records.

Do **not** commit `.analysis_runtime/`, virtual environments, Hugging Face weights, reconstructed-data/embedding caches, logs under runtime folders, or temporary extraction directories. These are ignored. The original `data/` folder and ZIP files are already ignored; the 113 MiB experimental archive remains there untouched. Check `git status` and `git diff --stat` before committing. Derived outputs are deliberately kept much smaller than the raw archive.

Share the original ZIP and its `.sha256` sidecar separately through a suitable research-data repository, institutional storage or other chosen hosting. Record its stable URL/DOI in the public README when available; this project does not fabricate one. Recipients should place both files in `data/`, verify the sidecar and run `bash analyze_local.sh`. Keep the archive checksum in the report and do not rename inner files. A separate downloadable archive avoids committing a file that exceeds common ordinary-Git hosting limits.

The standalone report HTML embeds figures and can be read offline. Its supplementary CSV/example links require the report directory or repository alongside it. The Markdown report is designed for GitHub and uses relative image/table links. For a minimal discussion package, share `key_findings.md`, `research_report.html`, and the underlying tables/figure directory.

Retain `ATTRIBUTION.md` and dataset/model provenance. The FLORES source records upstream **CC-BY-SA-4.0**, with a mirror-card discrepancy documented in the audit; the report and examples contain source/translated text, so preserve attribution and assess applicable share-alike requirements when redistributing them. NLLB weights are **CC-BY-NC-4.0** and are not included here. The MiniLM model card lists **Apache-2.0**; weights and cached embeddings are not published by default. Do not assume a model license automatically settles all rights in generated outputs, or silently apply a new blanket repository license. The existing repository owner should choose any code/report license explicitly.

Publication does not make this a replicated result. The one-model, one-corpus, one-main-run limitations and distinction between exact equality and semantic proxies should accompany any shared findings.
