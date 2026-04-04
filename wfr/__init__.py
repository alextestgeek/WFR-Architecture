"""
WFR-Architecture — публичное API исследовательского репозитория.

- :mod:`wfr.core` — WPE, резонанс, спайки, ``WFRNetwork``
- :mod:`wfr.losses` — композитный лосс Phase 1 (CE + RC + energy)

Корневые модули ``wfr_lm``, ``wfr_rfp`` (пакет + ``pip install -e .``) — LM и RFP.

Рекомендуется из корня репозитория: ``pip install -e .`` — тогда ``import wfr.core`` всегда актуален.
"""

from __future__ import annotations

from . import core
from . import losses

__all__ = ["core", "losses", "WFRNetwork"]

WFRNetwork = core.WFRNetwork
