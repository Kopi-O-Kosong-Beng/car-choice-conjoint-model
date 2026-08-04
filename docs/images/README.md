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

## vault-graph.png (updated 1 Aug 2026)

Obsidian graph view of the knowledge vault. To re-capture:
1. Open the repository root as an Obsidian vault (File > Open folder as vault).
2. Press Ctrl+G for graph view.
3. Optional, makes it far more readable:
   - Filters: set the search box to `path:Vault` so only notes show.
   - Groups: `path:Vault/Competition` in one colour, `path:Vault/Topics` in another.
   - Display: raise "Link thickness" and enable "Text fade threshold".
4. Win+Shift+S to snip, save here as `vault-graph.png`.
