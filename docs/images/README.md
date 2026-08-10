Images referenced by the top-level README.

## architecture.png, price-curve.png, fatigue-curve.png

The three figures from the final report, rendered as standalone PNGs. To regenerate
after editing `report/report.tex`, mirror the change in `docs/figs_standalone.tex`,
then from `docs/`:

```
pdflatex figs_standalone.tex
python -c "import pypdfium2 as p; d = p.PdfDocument('figs_standalone.pdf'); \
[d[i].render(scale=3.2).to_pil().save(f'images/{n}.png') \
 for i, n in enumerate(['architecture','price-curve','fatigue-curve'])]"
```

Requires `pip install pypdfium2`.

## report-p*.png (updated 10 Aug 2026)

Page thumbnails of `report/report.pdf`, one per README preview tile. The suffix is the
PDF page number, so these filenames change whenever pagination does. The current report
is 8 pages: cover, then seven numbered body pages. From the repository root:

```
python -c "import pypdfium2 as p; d = p.PdfDocument('report/report.pdf'); \
[d[i].render(scale=1.6).to_pil().save(f'docs/images/{n}.png') \
 for i, n in [(0,'report-p1-cover'), (2,'report-p3-architecture'), \
              (4,'report-p5-price'), (6,'report-p7-fatigue')]]"
```

Scale 1.6 gives 953 x 1348 px per page. Re-pick the pages if the figures move: the four
tiles are meant to be the cover and the three report figures.

## vault-graph.png (updated 10 Aug 2026)

Obsidian graph view of the repository, unfiltered, so every script, artifact, log and
course note appears. To re-capture:
1. Open the repository root as an Obsidian vault (File > Open folder as vault).
2. Press Ctrl+G for graph view.
3. Optional, for the Vault notes on their own rather than the whole repository:
   - Filters: set the search box to `path:Vault` so only notes show.
   - Groups: `path:Vault/Competition` in one colour, `path:Vault/Topics` in another.
   - Display: raise "Link thickness" and enable "Text fade threshold".
4. Win+Shift+S to snip, save here as `vault-graph.png`. Crop the Obsidian pane header
   ("Graph view") and any capture-tool border off the edges before committing.
