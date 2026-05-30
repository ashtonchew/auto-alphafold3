# Project Framing

This project is a NanoFold-style AlphaFold3-lite autoresearch scaffold for the Modal Autoresearch Hackathon.

We are not modifying, reproducing, training, improving, or beating Google DeepMind's real AlphaFold3. We are using NanoFold as an AlphaFold3-lite research sandbox.

The implementation target is `ogchen/nanofold`: a small, monomer-focused, backbone-oriented folding system inspired by AlphaFold papers. It is useful because it is small enough to inspect, patch, train, and evaluate under a locked benchmark.

The honest claim is narrow:

- run autoresearch over a small protein-folding model
- use cached feature data and fixed train/validation manifests
- evaluate with locked C-alpha lDDT
- route hypotheses with Fold Cartographer diagnostics
- execute fixed-budget trials through a Modal control plane

Do not describe this project as training real AlphaFold3, reproducing AlphaFold3, improving AlphaFold3, beating AlphaFold3, or using Google DeepMind AlphaFold3 parameters.
