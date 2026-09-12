"""Turn a pathway *index* into the name a reader recognises.

SurvPath's pathway tokens are the columns of the signature CSV, in file order:
``SurvivalDatasetFactory._setup_omics_data_for_survpath`` reads
``datasets_csv/metadata/<type_of_path>_signatures.csv`` and appends one entry to
``omic_names`` per column (datasets/dataset_survival.py:147-158), so token *p* is
column *p*. Nothing downstream ever carried that string along, which is why the
interpretability CSVs said ``pathway_142`` and the figures were titled with a
bare integer — the published ones say "Epithelial Mesenchymal Transition".

The mapping is purely positional, so this module only needs the header row.

    names = PathwayNames.load(type_of_path="combine")
    names[142]                      -> 'Epithelial Mesenchymal Transition'
    names.raw(142)                  -> 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION'
    names.wrapped(142, width=22)    -> 'Epithelial Mesenchymal\\nTransition'

If the CSV cannot be found the object still works and returns ``pathway 142``,
so a figure renders with degraded labels rather than not at all.
"""
from __future__ import annotations

import os
import re
import textwrap

DEFAULT_METADATA_DIR = os.path.join("datasets_csv", "metadata")

#: Source prefixes carried in the signature column names. They identify the
#: database a signature came from, which is worth keeping as a short tag rather
#: than dropping silently — two collections can hold same-named pathways.
_SOURCE_PREFIXES = ("HALLMARK_", "REACTOME_", "KEGG_", "BIOCARTA_", "PID_",
                    "WP_", "GOBP_", "GOCC_", "GOMF_")

#: Words that should not be title-cased when prettifying.
_LOWER = {"of", "the", "in", "to", "and", "or", "by", "via", "for", "with",
          "from", "at", "on", "into", "through", "during", "a", "an"}

#: Tokens that must keep their exact casing (gene symbols, complexes, units).
_ACRONYMS = {
    "dna", "rna", "mrna", "trna", "rrna", "mirna", "sirna", "ncrna", "atp",
    "adp", "amp", "gtp", "gdp", "camp", "cgtp", "nad", "nadh", "nadph", "fad",
    "tca", "er", "ecm", "mhc", "tcr", "bcr", "il", "tnf", "tgf", "egf", "fgf",
    "vegf", "pdgf", "igf", "ngf", "hgf", "gpcr", "rtk", "mapk", "erk", "jnk",
    "akt", "pkb", "pka", "pkc", "pi3k", "mtor", "ampk", "jak", "stat", "nfkb",
    "ap-1", "hif", "p53", "rb", "myc", "ras", "raf", "mek", "src", "abl",
    "brca1", "brca2", "atm", "atr", "chk1", "chk2", "parp", "apc", "cdk",
    "cdc", "sumo", "ubl", "usp", "hsp", "abc", "slc", "cox", "lox", "nos",
    "ros", "rns", "uv", "ip3", "dag", "pip2", "pip3", "ca2+", "k+", "na+",
    "cl-", "h+", "hiv", "hcv", "hbv", "sars-cov-2", "g1", "g2", "s", "m",
    "g0", "wnt", "shh", "bmp", "smad", "yap", "taz", "emt", "nk", "th1",
    "th2", "th17", "treg", "cd4", "cd8", "fc", "igg", "ige", "iga", "igm",
}


def _prettify(raw):
    """'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION' -> 'Epithelial Mesenchymal
    Transition'; 'ADP_signalling_through_P2Y_purinoceptor_12' -> 'ADP signalling
    through P2Y purinoceptor 12'.

    The two naming styles in these files need opposite treatment — MSigDB's are
    SHOUTED and must be down-cased, Reactome's are already sentence case and
    must be left alone — so the case of the source string decides which happens.
    """
    name = str(raw)
    for pre in _SOURCE_PREFIXES:
        if name.upper().startswith(pre):
            name = name[len(pre):]
            break

    name = name.replace("_", " ").strip()
    name = re.sub(r"\s+", " ", name)
    if not name:
        return str(raw)

    letters = [c for c in name if c.isalpha()]
    shouted = bool(letters) and all(c.isupper() for c in letters)
    if not shouted:
        # Reactome-style: already human-readable, only the first letter is ours
        # to touch — down-casing it would wreck 'ADP', 'P2Y', 'TP53'.
        return name[0].upper() + name[1:]

    out = []
    for i, word in enumerate(name.split(" ")):
        low = word.lower()
        if low in _ACRONYMS:
            out.append(word.upper() if len(low) <= 5 else word)
        elif low in _LOWER and i > 0:
            out.append(low)
        elif any(ch.isdigit() for ch in word):
            out.append(word)                 # 'P2Y', 'IL6', 'G1' — leave as-is
        else:
            out.append(low.capitalize())
    return " ".join(out)


class PathwayNames:
    """Positional index -> pathway name, with a safe fallback."""

    def __init__(self, raw_names=()):
        self._raw = [str(n) for n in raw_names]
        self._pretty = [_prettify(n) for n in self._raw]

    # -- construction ------------------------------------------------------ #
    @classmethod
    def load(cls, type_of_path="combine", metadata_dir=None, path=None,
             verbose=True):
        """Read the signature CSV header, mirroring the dataset's own lookup."""
        if path is None:
            base = metadata_dir or os.environ.get(
                "INTERP_METADATA_DIR") or DEFAULT_METADATA_DIR
            path = os.path.join(base, f"{type_of_path}_signatures.csv")
        if not os.path.isfile(path):
            if verbose:
                print(f"[warn] pathway names: {path} not found -> panels will be "
                      f"labelled 'pathway <n>'. Point --metadata-dir at the "
                      f"directory holding <type_of_path>_signatures.csv.")
            return cls()

        import pandas as pd
        cols = list(pd.read_csv(path, nrows=0).columns)
        if verbose:
            print(f"[info] pathway names: {len(cols)} from {path}")
        return cls(cols)

    @classmethod
    def from_importance_csv(cls, df):
        """Recover names from a ``*_pathway_importance.csv`` that has them.

        Older runs wrote the placeholder ``pathway_<n>`` into that column; those
        are rejected here so the caller falls back to the signature CSV instead
        of printing the placeholder as if it were a name.
        """
        if df is None or "pathway" not in df or "pathway_idx" not in df:
            return cls()
        pairs = dict(zip(df["pathway_idx"].astype(int), df["pathway"].astype(str)))
        if not pairs or all(re.fullmatch(r"pathway[_ ]?\d+", v) for v in pairs.values()):
            return cls()
        n = max(pairs) + 1
        return cls([pairs.get(i, f"pathway_{i}") for i in range(n)])

    # -- lookup ------------------------------------------------------------ #
    def __len__(self):
        return len(self._raw)

    def __bool__(self):
        return bool(self._raw)

    def __getitem__(self, idx):
        i = int(idx)
        if 0 <= i < len(self._pretty):
            return self._pretty[i]
        return f"pathway {i}"

    def raw(self, idx):
        i = int(idx)
        return self._raw[i] if 0 <= i < len(self._raw) else f"pathway_{i}"

    def wrapped(self, idx, width=24, max_lines=2):
        """Name broken over at most ``max_lines`` lines, for a panel header."""
        lines = textwrap.wrap(self[idx], width=width) or [self[idx]]
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1].rstrip(" ,;") + "…"
        return "\n".join(lines)

    def check(self, n_pathways, verbose=True):
        """Warn when the CSV and the model disagree on the token count.

        A mismatch means the labels would be off by an unknown offset — the
        signature file is not the one the checkpoint was trained with — so it is
        safer to drop back to indices than to print confident wrong names.
        """
        if not self._raw:
            return self
        if len(self._raw) != n_pathways:
            if verbose:
                print(f"[warn] pathway names: signature CSV has {len(self._raw)} "
                      f"columns but the attention matrix has {n_pathways} "
                      f"pathways. These names would be misaligned, so falling "
                      f"back to 'pathway <n>'. Pass the "
                      f"--type-of-path/--metadata-dir this checkpoint was "
                      f"trained with.")
            return PathwayNames()
        return self
