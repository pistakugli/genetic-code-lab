# Gates: measured cubic oracle extraction

OWNS: tools/llm_reader/cubic_identification_probe.py, tests/test_cubic_oracle_recovery.py, docs/nn-decompiler-roadmap.md, .agents/notes/implemented/feature/*, GATES.md

Scope: expose measured-span SwiGLU gate recovery through an oracle-only callable and prove overcomplete atom-count inference without solver access to teacher metadata or weights.

- [x] G1: callable-only recovery passes its access and 10-of-8 overcomplete assertions
  CHECK: ./.gcl-python-gpu/bin/python tests/test_cubic_oracle_recovery.py
  EXPECT: cubic oracle-only recovery passed
  CWD: .
  EVIDENCE: automatic-evidence=v1; definition-sha256=0906ebc7b3cc4c313ad58e6d5725a8b0686bcc4b4941e8175bd4306e0848e677; exit=0; EXPECT=matched; output-sha256=20cb4100beb20f82e30822858b4185f2de2284a219782f89c7bd5521a05c2b57; output-bytes=443; shell=/bin/sh; cwd=/home/kugli/Downloads/Genetic-Code-Lab/genetic-code-lab; path=b2667be5e3a2/24 entries

- [x] G2: live cubic access invariants remain intact
  CHECK: npm run test:cubic-live-access && echo cubic-live-access-invariants-passed
  EXPECT: cubic-live-access-invariants-passed
  CWD: .
  EVIDENCE: automatic-evidence=v1; definition-sha256=57ae89c4fceeeb15efb2c3251e1596b1a83a34e136d80eaa2115d96a2296e9bb; exit=0; EXPECT=matched; output-sha256=6a039f21e370c9878feef07a6011d8a15f39cf288e23aa2723ddda0b0c38a472; output-bytes=1172; shell=/bin/sh; cwd=/home/kugli/Downloads/Genetic-Code-Lab/genetic-code-lab; path=b2667be5e3a2/24 entries

- [x] G3: existing synthetic oracle lane still recovers all 10 gates from measured rank
  CHECK: ./.gcl-python-gpu/bin/python tools/llm_reader/cubic_identification_probe.py --device cpu --oracle-recover 10 --input-dim 8 --reach-width 8 --base-count 3 --readers 8 --steps 0.02 0.04 --span-tolerance 1e-5 --rank-tolerance 1e-6
  EXPECT: "gatesMatchSpanRank": true
  CWD: .
  EVIDENCE: automatic-evidence=v1; definition-sha256=2f358e3342f69413559859e3cc6958499ee791eb4d0b370be7e279a4f60bb099; exit=0; EXPECT=matched; output-sha256=c359a15abd7d55177aa24c18b60db9cf3d3a56909b787af0897ba25b58825715; output-bytes=662; shell=/bin/sh; cwd=/home/kugli/Downloads/Genetic-Code-Lab/genetic-code-lab; path=b2667be5e3a2/24 entries
