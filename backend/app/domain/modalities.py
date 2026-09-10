from enum import Enum


class Modality(str, Enum):
    OPTICAL = "OPTICAL"
    MULTISPECTRAL = "MULTISPECTRAL"
    SAR = "SAR"
    UNKNOWN = "UNKNOWN"


class ImageFormat(str, Enum):
    GEOTIFF = "GEOTIFF"
    TIFF = "TIFF"
    PNG = "PNG"
    JPEG = "JPEG"
    WEBP = "WEBP"
    UNKNOWN = "UNKNOWN"
