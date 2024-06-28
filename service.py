from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from PIL import Image as PILImage
import bentoml
from bentoml.io import Image, Multipart, NumpyNdarray, JSON
from datetime import datetime, timedelta, timezone
#from pydantic import BaseModel, Field


ktc_offset = timezone(timedelta(hours=9))

if TYPE_CHECKING:
    from numpy.typing import NDArray

gasdoor_runner = bentoml.models.get("gasdoor:latest").to_runner()
svc = bentoml.Service(name="gasdoor_demo", runners=[gasdoor_runner])

input_spec = Multipart(arr=Image(), annotat=JSON())


@svc.api(input=Image(), output=JSON())
#@svc.api(input=input_spec, output=NumpyNdarray())
#@svc.api(input=input_spec, output=JSON())
#async def predict_image(arr: PILImage.Image, annotat: dict[str, any], ctx: bentoml.Context) -> np.ndarray:
async def predict_image(image: PILImage.Image, ctx: bentoml.Context) -> np.ndarray:
    image = PILImage.merge("RGB", image.split()[:3]) if image.mode.endswith('A') else image
    image = np.array(image)
    print(f"Input image size: {image.shape}")
    out_lines, mask, line_img, patches, predictions, out_img, out_mask = await gasdoor_runner.async_run(image, eps=20, min_length=70)

    return {'hough': out_lines, 'image': out_img, 'mask': out_mask}




# class ImageMetadata(BaseModel):
#     eps: int = Field(description="Description of the image")
#     min_length: int = Field(description="Timestamp of when the image was captured")


# @bentoml.service
# class ImageProcessingService:

#     sample_model_runner = bentoml.depends(svc)

#     @bentoml.api
#     def predict_image(self, image: PILImage, metadata: ImageMetadata) -> dict:
#         image = PILImage.merge("RGB", image.split()[:3]) if image.mode.endswith('A') else image
#         image = np.array(image)
#         print(f"Input image size: {image.shape}")
#         out_lines, mask, line_img, patches, predictions, out_img, out_mask = self.predict.run(image)

#         return {'hough': out_lines, 'image': out_img, 'mask': out_mask}
