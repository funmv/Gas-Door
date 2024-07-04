from __future__ import annotations
import os
from typing import TYPE_CHECKING
import numpy as np
from PIL import Image as PILImage
import bentoml
from bentoml.io import Image, Multipart, NumpyNdarray, JSON
from datetime import datetime, timedelta, timezone


ktc_offset = timezone(timedelta(hours=9))

if TYPE_CHECKING:
    from numpy.typing import NDArray

gasdoor_runner = bentoml.models.get("gasdoor:latest").to_runner()
svc = bentoml.Service(name="gasdoor_demo", runners=[gasdoor_runner])


def timed_filename(time_str):
    utc_time = datetime.utcnow()
    ktc_time = utc_time.replace(tzinfo=timezone.utc).astimezone(ktc_offset)
    return ktc_time.strftime(time_str)
    
def save_img(output, label):
    filename = timed_filename("%Y%m%d_%H%M%S")+'_'+label+'.jpg'
    path_name = os.path.join(os.getcwd(),filename)
    print(path_name)
    image = PILImage.fromarray(output)
    image.save(path_name)
    return filename


@svc.api(input=Image(), output=JSON())
async def predict_image(image: PILImage.Image, ctx: bentoml.Context) -> dict:
    request_headers = ctx.request.headers
    client_ip = request_headers['host'].split(':')[0]
    print(request_headers['host'])
    if client_ip not in ["127.0.0.1", "::1", 'localhost']:
        return {}
        
    image = PILImage.merge("RGB", image.split()[:3]) if image.mode.endswith('A') else image
    image = np.array(image)
    print(f"Input image size: {image.shape}")
    out_lines, mask, line_img, patches, predictions, out_img, out_mask = await gasdoor_runner.async_run(image, eps=20, min_length=70)
    try:
        filenaem = save_img(line_img, 'im') 
        filenaem = save_img(mask, 'mask') 
    except Exception as e:
        print(f"Exception in saving Images: {e}")

    return {'hough': out_lines, 'image': out_img, 'mask': out_mask}
