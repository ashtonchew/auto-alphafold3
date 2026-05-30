"""Demo UI renderer.

Turns real run artifacts (or an illustrative sample) into the Modal-styled
evidence board. The same renderer powers the committed design mockups
(``--sample``) and the live demo (``--runs runs``); see ``build.py``.

No web framework, no server: components are plain functions that return HTML
and reuse the shared design system in ``docs/spec/ui/assets/modal.css``.
"""

from autoalphafold3.ui.state import UiState, load_state, sample_state

__all__ = ["UiState", "load_state", "sample_state"]
