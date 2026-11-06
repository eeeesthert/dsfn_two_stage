# ABUS manuscript Word edition

`ABUS_two_stage_stitching_word.xml` is an editable Microsoft Word Flat OPC
edition of the supplied two-stage ABUS stitching manuscript. Flat OPC is the
single-file XML representation of a `.docx`: unlike a binary `.docx`, the file
can be displayed, reviewed, and downloaded by source-code hosting systems.
The duplicated LaTeX document was reduced to one copy, and the stray duplicate
`\end{document}` was omitted.

The document contains:

- the manuscript's Introduction, Related Work, Method, Experiments, Discussion,
  Conclusion, and 46-entry References structure;
- all 19 displayed method equations as native Office Math (OMML) objects;
- all four quantitative result/ablation tables as native Word tables; and
- explicit notes where the source still contains `XXX` experimental placeholders.

Rebuild it without external Python packages:

```bash
python docs/build_word_manuscript.py
```

The generated package can be checked with:

```bash
python -c "from xml.etree import ElementTree as ET; ET.parse('docs/ABUS_two_stage_stitching_word.xml')"
```

## Open and save in Microsoft Word

1. Download `docs/ABUS_two_stage_stitching_word.xml` from the repository.
2. In Microsoft Word, choose **File > Open** and select the downloaded XML file.
3. Choose **File > Save As > Word Document (`.docx`)** if a conventional DOCX
   file is required.

Equations and tables remain editable after opening or saving. The generated
`.docx` is intentionally not committed because this repository's pull-request
system does not support binary files.
