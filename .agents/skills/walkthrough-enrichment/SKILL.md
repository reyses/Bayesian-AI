---
name: walkthrough-enrichment
description: Enforces the use of data tables, distribution graphs, plots, and GIFs in walkthrough artifacts instead of wall text.
---
# Walkthrough Enrichment

When creating or updating a `walkthrough.md` artifact (or any summary artifact), you MUST NOT use "wall of text" summaries. 

Instead, enrich the walkthrough in the same way we enrich formal research reports:
1. **Data Tables**: Use markdown tables to present metrics, test results, and parameters cleanly.
2. **Visualizations**: Actively write scripts in the `scratch/` directory to generate distribution graphs, plots, and animated GIFs that visualize the results.
3. **Embed Media**: Embed the generated plots and GIFs directly into the walkthrough using the `![caption](/absolute/path/to/media.png)` format.
4. **Scannability**: Keep text concise. Let the data and visual evidence do the talking.
