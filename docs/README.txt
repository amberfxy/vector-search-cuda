Interview Canvas — LOCAL ONLY
=============================

The full file NVIDIA_INTERVIEW_CANVAS.md is gitignored and will NOT
appear after git pull. This folder exists so you know where to put it.

ON YOUR MAC — 2 steps:

1. Create the file location (if needed):
   mkdir -p docs

2. In Cursor Agent chat (local project open), paste:

   ---
   Write docs/NVIDIA_INTERVIEW_CANVAS.md (do NOT git commit).
   Full NVIDIA first-round interview canvas for vector-search-cuda:
   - PRIORITY MAP (P0/P1/P2)
   - SPOKEN SCRIPT Q0-Q7 + Limitations + Numbers (conversational English)
   - GLOSSARY (wall time, AI, ridge, coalescing, device-resident, etc.)
   - CORE NUMBERS (6-digit mnemonic)
   - Formal Q&A, limitations, code pointers, don't-say/do-say
   Use README.md benchmark numbers (T4, dim=384, 10K/100K/1M).
   Block-per-vector reduction = NOT shipped. FAISS = CPU IndexFlatL2 only.
   ---

3. Open: Cmd+P → type NVIDIA_INTERVIEW_CANVAS

Do NOT git add the canvas file (.gitignore blocks it).
