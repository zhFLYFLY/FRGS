import json
import os
import time
import base64
import unicodedata

from openai import OpenAI
from openai import AzureOpenAI

import requests


class LLM:
    def __init__(self, api_key, model_name, max_tokens, cache_name='default', **kwargs):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.queried_tokens = 0

        cache_model_dir = os.path.join('llm', 'cache', self.model_name)
        os.makedirs(cache_model_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_model_dir, f'{cache_name}.json')
        self.cache = dict()

        if os.path.isfile(self.cache_file):
            with open(self.cache_file) as f:
                self.cache = json.load(f)

    def query_api(self, prompt):
        raise NotImplementedError

    def get_cache(self, prompt, instance_idx):
        sequences = self.cache.get(instance_idx, [])

        for sequence in sequences:
            if sequence.startswith(prompt) and len(sequence) > len(prompt)+1:
                return sequence
        return None

    def add_to_cache(self, sequence, instance_idx):
        if instance_idx not in self.cache:
            self.cache[instance_idx] = []
        sequences = self.cache[instance_idx]

        sequences.append(sequence)

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
        print('cache saved to: ' + self.cache_file)

    def get_sequence(self, prompt, instance_idx, read_cache=True):
        sequence = None
        if read_cache:
            sequence = self.get_cache(prompt, instance_idx)
        print('cached sequence')
        if sequence is None:
            print('query API')
            sequence = self.query_api(prompt)
            self.add_to_cache(sequence, instance_idx)
        return sequence


class OpenAI_LLM_v2(LLM):
    def __init__(self, model_name, api_key, client_type="openai", logit_bias=None, max_tokens=64, finish_reasons=None, **kwargs):

        if client_type == "openai":
            self.client = OpenAI(
                api_key=api_key,
                base_url = "https://ark.cn-beijing.volces.com/api/v3")

        self.logit_bias = logit_bias

        self.finish_reasons = finish_reasons
        if finish_reasons is None:
            self.finish_reasons = ['stop', 'length']

        super().__init__(api_key, model_name, max_tokens, **kwargs)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def query_viewpoint_api(self, prompt, image_paths=None, show_response=True):
        def query_func():
            content_block = []
            if image_paths is not None:
                for viewpoint, img_p in image_paths.items():
                    content_block.append({
                        "type": "text",
                        "text": f"{viewpoint} image: "
                    })
                    content_block.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self.encode_image(img_p)}"
                        }
                    })
            content_block.append({
                "type": "text",
                "text": f"{prompt}"
            })

            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": content_block
                    }
                ],
                model= self.model_name,
            )
            print("Using model:", self.model_name)

            message = completion.choices[0].message
            content = unicodedata.normalize('NFKC', message.content)

            return content

        try:
            response = query_func()
        except Exception as e:
            print(e)
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')

        return response

    def query_image_api(self, prompt, images=None, show_response=True):
        import io
        import base64
        import os

        def encode_image_from_data(image_data):
            if isinstance(image_data, str):
                if os.path.exists(image_data):
                    return self.encode_image(image_data)
                try:
                    base64.b64decode(image_data)
                    return image_data
                except:
                    raise ValueError(f"Invalid image data: {image_data[:50]}...")

            if isinstance(image_data, bytes):
                return base64.b64encode(image_data).decode('utf-8')

            try:
                import numpy as np
                if isinstance(image_data, np.ndarray):
                    from PIL import Image
                    if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                        image_data = image_data[:, :, ::-1]
                    pil_image = Image.fromarray(image_data)
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format='JPEG', quality=85)
                    return base64.b64encode(buffer.getvalue()).decode('utf-8')
            except ImportError:
                pass

            try:
                from PIL import Image
                if isinstance(image_data, Image.Image):
                    buffer = io.BytesIO()
                    if image_data.mode != 'RGB':
                        image_data = image_data.convert('RGB')
                    image_data.save(buffer, format='JPEG', quality=85)
                    return base64.b64encode(buffer.getvalue()).decode('utf-8')
            except ImportError:
                pass

            raise TypeError(f"Unsupported image type: {type(image_data)}")

        def query_func():
            content_block = []
            if images is not None:
                if isinstance(images, dict):
                    for viewpoint, img_data in images.items():
                        content_block.append({
                            "type": "text",
                            "text": f"{viewpoint} image: "
                        })
                        content_block.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image_from_data(img_data)}"
                            }
                        })
                elif isinstance(images, list):
                    for idx, img_data in enumerate(images):
                        content_block.append({
                            "type": "text",
                            "text": f"Image {idx + 1}: "
                        })
                        content_block.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image_from_data(img_data)}"
                            }
                        })
                else:
                    content_block.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_from_data(images)}"
                        }
                    })

            content_block.append({
                "type": "text",
                "text": f"{prompt}"
            })

            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": content_block
                    }
                ],
                model=self.model_name,
            )

            message = completion.choices[0].message
            content = unicodedata.normalize('NFKC', message.content)

            return content

        try:
            response = query_func()
        except Exception as e:
            print(e)
            time.sleep(10)
            print('try again')
            return self.query_image_api(prompt, images, show_response)

        if show_response:
            print('API Response:')
            print(response)
            print('')

        return response