from abc import ABC, abstractmethod
from typing import Any, Dict, List
from app.domain.tasks import TaskFamily
from app.domain.modalities import Modality


class ModelAdapter(ABC):
    """
    Abstract specialist adapter interface for remote sensing AI backbones.
    Provides lazy loading, explicit resource management, health checks, and metadata.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Official canonical model repository or checkpoint identifier."""
        pass

    @property
    @abstractmethod
    def task_family(self) -> TaskFamily:
        """Primary task family handled by this adapter."""

        pass

    @property
    @abstractmethod
    def supported_modalities(self) -> List[Modality]:
        """List of supported raster modalities."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether model weights are loaded into memory."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load model weights and preprocessors onto target device."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model weights to release GPU/RAM memory."""
        pass

    @abstractmethod
    def predict(self, request: Any) -> Any:
        """Execute task inference."""
        pass

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Check adapter status and compute availability."""
        pass

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return provenance metadata, checkpoint versions, and licensing."""
        pass
