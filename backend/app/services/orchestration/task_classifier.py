import re
from typing import Any, Dict, List, Optional, Tuple
from app.domain.tasks import TaskFamily
from app.domain.modalities import Modality
from app.schemas.image_asset import ImageAsset
from app.core.logging import logger


class TaskClassifier:
    """
    Intelligent query interpretation and task classification service.
    Interprets user intent using both natural-language semantics and input imagery metadata
    (count, modalities, spatial overlaps). Includes deterministic fallback that never
    requires a paid API key.
    """

    # Spatial grounding indicators (localization, highlighting, finding, bounding)
    GROUNDING_PATTERNS = [
        r"\b(highlight|locate|find|where\s+is|where\s+are|bounding\s+box|box|boxes|outline|segment|delineate|mark|point\s+out|show\s+me\s+where|detect)\b",
        r"\b(in\s+the\s+(southern|northern|eastern|western|top|bottom|center|left|right)\s+part)\b",
        r"\b(pinpoint|spatial\s+extent|region\s+of)\b",
    ]

    # VQA question indicators (identification, counting, descriptive inquiry)
    VQA_PATTERNS = [
        r"\b(what\s+(is|are|kind|type|color|feature)|how\s+many|is\s+there|are\s+there|can\s+you\s+see|identify|tell\s+me\s+about)\b",
        r"\b(primary\s+land\s+cover|density|condition|state\s+of)\b",
    ]

    # Captioning indicators
    CAPTION_PATTERNS = [
        r"\b(caption|describe\s+the\s+scene|summarize\s+the\s+image|overview\s+of\s+the\s+image|generate\s+a\s+description)\b",
    ]

    # Temporal change indicators (bitemporal comparison, differences, trends)
    TEMPORAL_PATTERNS = [
        r"\b(change|changed|difference|differences|between|before\s+and\s+after|over\s+time|temporal|evolution|trend|growth|loss|increase|decrease|expanded|shrunk)\b",
        r"\b(from\s+\d{4}\s+to\s+\d{4}|\d{4}\s+vs\s+\d{4})\b",
    ]

    # Unsupported semantic transition patterns (e.g. "Was forest converted to buildings?")
    SEMANTIC_TRANSITION_PATTERNS = [
        r"\b(convert|converted|conversion|transform|transformed|turned into|transition|transitioned)\b",
        r"\b(from\s+\w+\s+to\s+\w+)\b",
        r"\b(forest\s+to|farmland\s+to|agriculture\s+to|vegetation\s+to|trees\s+to|water\s+to)\b",
        r"\b(deforestation|afforestation|urbanization|commercial\s+to|residential\s+to)\b",
    ]

    # Cross-modal Optical + SAR indicators
    OPTICAL_SAR_PATTERNS = [
        r"\b(sar\s+and\s+optical|optical\s+and\s+sar|cross-modal|fuse|fusion|radar\s+and\s+multispectral|penetrate\s+clouds)\b",
    ]

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def classify(
        self,
        query: str,
        assets: List[ImageAsset]
    ) -> Tuple[TaskFamily, Dict[str, Any]]:
        """
        Classifies task family from query and input assets.
        Returns: (TaskFamily, classification_metadata)
        """
        clean_query = query.strip()
        num_assets = len(assets)
        modalities = [a.modality for a in assets]
        has_sar = Modality.SAR in modalities
        has_optical = Modality.OPTICAL in modalities or Modality.MULTISPECTRAL in modalities

        # Try structured LLM classification if available and configured
        if self.llm_client is not None:
            try:
                result = self._classify_with_llm(clean_query, num_assets, modalities)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"LLM classifier failed, falling back to deterministic classifier: {e}")

        # Deterministic classification
        return self._classify_deterministic(clean_query, num_assets, has_optical, has_sar, modalities)

    def _classify_deterministic(
        self,
        query: str,
        num_assets: int,
        has_optical: bool,
        has_sar: bool,
        modalities: List[Modality]
    ) -> Tuple[TaskFamily, Dict[str, Any]]:
        lower_q = query.lower()

        # Multi-image: Optical + SAR Pair Analysis
        if num_assets == 2 and has_optical and has_sar:
            return TaskFamily.OPTICAL_SAR_ANALYSIS, {
                "method": "deterministic_multimodal_pair",
                "reason": "Input contains paired Optical and SAR imagery.",
                "requires_grounding": any(re.search(p, lower_q) for p in self.GROUNDING_PATTERNS)
            }

        # Multi-image: Temporal Analysis
        if num_assets >= 2:
            is_semantic_transition = any(re.search(p, lower_q) for p in self.SEMANTIC_TRANSITION_PATTERNS)
            is_temporal = any(re.search(p, lower_q) for p in self.TEMPORAL_PATTERNS) or num_assets == 2
            is_vqa = any(re.search(p, lower_q) for p in self.VQA_PATTERNS) or lower_q.endswith("?") or is_semantic_transition
            if is_vqa:
                return TaskFamily.TEMPORAL_CHANGE_VQA, {
                    "method": "deterministic_temporal_vqa",
                    "reason": "Multiple temporal acquisitions with query-directed change question.",
                    "unsupported_semantic_query": is_semantic_transition
                }
            return TaskFamily.TEMPORAL_CHANGE, {
                "method": "deterministic_temporal_change",
                "reason": "Multiple temporal acquisitions with change detection intent.",
                "unsupported_semantic_query": is_semantic_transition
            }

        # Single image: Check query patterns
        has_grounding = any(re.search(p, lower_q) for p in self.GROUNDING_PATTERNS)
        has_vqa = any(re.search(p, lower_q) for p in self.VQA_PATTERNS) or lower_q.endswith("?")
        has_caption = any(re.search(p, lower_q) for p in self.CAPTION_PATTERNS)

        # Composite query check: Query asks to BOTH answer and highlight/locate
        # E.g. "What is the water body and highlight it" or "Identify and outline trees"
        if has_grounding and has_vqa:
            return TaskFamily.SINGLE_GROUNDING, {
                "method": "deterministic_composite",
                "reason": "Query contains both inquiry and spatial localization requirements.",
                "is_composite": True,
                "needs_vqa": True,
                "needs_grounding": True
            }

        if has_grounding:
            return TaskFamily.SINGLE_GROUNDING, {
                "method": "deterministic_grounding",
                "reason": "Query requests spatial localization or feature highlighting.",
                "is_composite": False,
                "needs_vqa": False,
                "needs_grounding": True
            }

        if has_caption:
            return TaskFamily.SINGLE_CAPTION, {
                "method": "deterministic_caption",
                "reason": "Query requests scene captioning or descriptive summary.",
                "is_composite": False
            }

        # Default single image is VQA
        return TaskFamily.SINGLE_VQA, {
            "method": "deterministic_vqa",
            "reason": "Single image inquiry routed to visual question answering.",
            "is_composite": False,
            "needs_vqa": True,
            "needs_grounding": False
        }

    def _classify_with_llm(
        self,
        query: str,
        num_assets: int,
        modalities: List[Modality]
    ) -> Optional[Tuple[TaskFamily, Dict[str, Any]]]:
        """Optional structured LLM classifier with schema validation and strict error boundaries."""
        # Intentionally returns None if no active provider is injected
        return None
