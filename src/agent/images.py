"""Agent 双向附图：聊天图片和工作目录内的图片。"""

from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

from PIL import Image

import asyncio
import re

from src.common.data_models.message_component_data_model import EmojiComponent, ImageComponent, ReplyComponent

if TYPE_CHECKING:
    from .service import AgentService


async def resolve_images(
    service: "AgentService", caller: str, references: Sequence[str],
) -> Tuple[List[ImageComponent], List[Dict[str, str]]]:
    if isinstance(references, str) or not all(isinstance(ref, str) and ref for ref in references):
        raise ValueError("images 必须是非空图片引用组成的列表")
    images: List[ImageComponent] = []
    attachments: List[Dict[str, str]] = []
    if not references:
        return images, attachments

    from .messaging import image_sources, source_images

    available = dict(await source_images(caller)) if caller != service.agent_id else {}
    workspace = service.workspace

    def save(data: bytes) -> Tuple[ImageComponent, Dict[str, str]]:
        # 校验真实图片格式，原始字节保真保存；阻塞解码和文件操作在线程池进行。
        with Image.open(BytesIO(data)) as decoded:
            suffix = decoded.format.lower()
            decoded.verify()
        image = ImageComponent(binary_hash="", binary_data=data)
        folder = workspace / "relay_images"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{image.binary_hash}.{suffix}"
        path.write_bytes(data)
        image.content = f"[传话图片 hash={image.binary_hash}，附件文件={path}]"
        return image, {"hash": image.binary_hash, "path": str(path)}

    for reference in references:
        if reference not in available and caller != service.agent_id:
            # A quoted message can have fallen out of the live history. Query only the
            # current chat, and follow only its explicit quote, never the sender avatar.
            from src.services.message_service import get_message_by_id

            original = await asyncio.to_thread(get_message_by_id, reference, caller)
            if original is not None:
                sources = image_sources(original.raw_message.components)
                if not sources:
                    replies = [part.target_message_id for part in original.raw_message.components if isinstance(part, ReplyComponent)]
                    if len(replies) == 1:
                        quoted = await asyncio.to_thread(get_message_by_id, replies[0], caller)
                        if quoted is not None:
                            sources = image_sources(quoted.raw_message.components)
                if sources:
                    available[reference] = sources
        if reference in available:
            for original in available[reference]:
                image = original.clone()
                if isinstance(image, EmojiComponent):
                    await image.load_emoji_binary()
                else:
                    await image.load_image_binary()
                image, attachment = await asyncio.to_thread(save, image.binary_data)
                images.append(image)
                attachments.append(attachment)
            continue
        elif caller == service.agent_id and re.fullmatch(r"[a-f0-9]{64}", reference):
            matches = await asyncio.to_thread(lambda reference=reference: list((workspace / "relay_images").glob(reference + ".*")))
            if len(matches) != 1:
                raise ValueError(f"没有唯一的工作区图片：{reference}")
            data = await asyncio.to_thread(matches[0].read_bytes)
        elif caller == service.agent_id:
            path = await asyncio.to_thread(service.image_path, reference)
            data = await asyncio.to_thread(path.read_bytes)
        else:
            raise ValueError(f"当前聊天上下文中没有图片：{reference}；请使用带图消息 ID 或图片 hash")
        image, attachment = await asyncio.to_thread(save, data)
        images.append(image)
        attachments.append(attachment)
    return images, attachments
