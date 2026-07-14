import json
import os
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
import piexif
import piexif.helper
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from enum import Enum

import folder_paths

from .. import hook
from ..capture import Capture
from ..trace import Trace
from ..utils.log import print_warning


class OutputFormat(str, Enum):
    PNG = "png"
    PNG_JSON = "png_with_json"
    JPG = "jpg"
    JPG_JSON = "jpg_with_json"
    WEBP = "webp"
    WEBP_JSON = "webp_with_json"


class QualityOption(str, Enum):
    MAX = "max"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MetadataScope(str, Enum):
    FULL = "full"
    DEFAULT = "default"
    PARAMETERS_ONLY = "parameters_only"
    WORKFLOW_ONLY = "workflow_only"
    NONE = "none"


# refer. https://github.com/comfyanonymous/ComfyUI/blob/38b7ac6e269e6ecc5bdd6fefdfb2fb1185b09c9d/nodes.py#L1411
class SaveImageWithMetaData:
    OUTPUT_FORMATS = [e for e in OutputFormat]
    QUALITY_OPTIONS = [e for e in QualityOption]
    METADATA_OPTIONS = [e for e in MetadataScope]
    NEEDS_METADATA_KEYS = {"seed", "width", "height", "pprompt", "nprompt", "model"}

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the saved file. You can include formatting options like %date:yyyy-MM-dd% or %seed%, and combine them as needed, e.g., %date:hhmmss%_%seed%."}),
                "subdirectory_name": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Custom directory to save the images. Leave empty to use the default output "
                        "directory. You can include formatting options like %date:yyyy-MM-dd%."
                    ),
                }),
                "output_format": (s.OUTPUT_FORMATS, {
                    "tooltip": "The format in which the images will be saved."
                }),
            },
            "optional": {
                "extra_metadata": ("EXTRA_METADATA", {
                    "tooltip": "Additional key-value metadata to include in the image."
                }),
                # Keep this after the existing extra_metadata socket so old
                # serialized links retain their input-slot index.
                "context": ("RGTHREE_CONTEXT", {
                    "tooltip": (
                        "Optional rgthree Context/Context Big input. Non-empty "
                        "context values take priority over automatic graph detection."
                    )
                }),
                "quality": (s.QUALITY_OPTIONS, {
                    "tooltip": "Image quality:"
                            "\n'max' / 'lossless WebP' - 100"
                            "\n'high' - 80"
                            "\n'medium' - 60"
                            "\n'low' - 30"
                            "\n\nNote: Lower quality, smaller file size. PNG images ignore this setting."
                }),
                "metadata_scope": (s.METADATA_OPTIONS, {
                    "tooltip": "Choose the metadata to save: "
                            "\n'full' - default metadata with additional metadata, "
                            "\n'default' - same as SaveImage node, "
                            "\n'parameters_only' - only A1111-style metadata, "
                            "\n'workflow_only' - workflow metadata only, "
                            "\n'none' - no metadata."
                }),
                "include_batch_num": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include batch number in filename."
                }),
                "prefer_nearest": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Select inputs from closest nodes first if true."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves the input images with metadata to your ComfyUI output directory."
    CATEGORY = "Cyberdelia/Metadata"

    pattern_format = re.compile(r"(%[^%]+%)")

    def parse_output_format(self, output_format: str):
        fmt = OutputFormat(output_format)
        save_workflow_json = fmt.name.endswith("JSON")
        base_format = fmt.replace("_with_json", "")
        return base_format, save_workflow_json

    def get_quality_value(self, quality: str) -> int:
        return {
            QualityOption.MAX: 100,
            QualityOption.HIGH: 80,
            QualityOption.MEDIUM: 60,
            QualityOption.LOW: 30
        }.get(quality, 100)

    def find_next_available_filename(self, folder: str, name: str, ext: str):
        existing = {f.stem for f in Path(folder).glob(f"{name}_*.{ext}")}
        i = 1
        while f"{name}_{i:05d}" in existing:
            i += 1
        return i

    @classmethod
    def parse_filename_placeholders(cls, filename: str) -> list[str]:
        return re.findall(cls.pattern_format, filename) if "%" in filename else []

    def needs_pnginfo_in_filename(self, segments: list[str]) -> bool:
        for segment in segments:
            parts = segment.strip("%").split(":")
            if parts[0] in self.NEEDS_METADATA_KEYS:
                return True
        return False

    @staticmethod
    def _linked_prompt_input_source(prompt, save_node_id, input_name):
        if not isinstance(prompt, Mapping):
            return None
        if save_node_id not in prompt and str(save_node_id) in prompt:
            save_node_id = str(save_node_id)
        save_node = prompt.get(save_node_id)
        if not isinstance(save_node, Mapping):
            return None
        link = save_node.get("inputs", {}).get(input_name)
        if (
            not isinstance(link, (list, tuple))
            or len(link) != 2
            or not isinstance(link[0], (str, int))
            or not isinstance(link[1], int)
        ):
            return None
        source_id = link[0]
        if source_id in prompt:
            return source_id
        source_id = str(source_id)
        return source_id if source_id in prompt else None

    async def save_images(self, images, filename_prefix="ComfyUI", subdirectory_name="", prompt=None,
                    extra_pnginfo=None, extra_metadata=None, output_format="png",
                    quality="max", metadata_scope="full",
                    include_batch_num=True, prefer_nearest=True, pnginfo_dict=None,
                    context=None):

        metadata_scope = MetadataScope(metadata_scope)
        extra_metadata = extra_metadata or {}
        base_format, save_workflow_json = self.parse_output_format(output_format)

        filename_prefix = filename_prefix.strip()
        segments = self.parse_filename_placeholders(filename_prefix)

        needs_metadata = (
            metadata_scope in [MetadataScope.FULL, MetadataScope.PARAMETERS_ONLY]
            or self.needs_pnginfo_in_filename(segments)
        )

        images_length = len(images)
        is_list_batch = images_length > 1

        # For single images or when metadata isn't needed, we can resolve once.
        # For list-batches we defer per-image resolution so each image gets
        # the prompt string that was actually used to generate it.
        if needs_metadata and not is_list_batch:
            # batch_index=0 is always correct for a single image
            pnginfo_dict = pnginfo_dict or await self.gen_pnginfo(prompt, prefer_nearest, batch_index=0)

        # Use batch_index=0 for filename formatting (consistent across the batch)
        if pnginfo_dict is not None:
            fmt_pnginfo = self.apply_rgthree_context(
                pnginfo_dict, context, batch_index=0
            ) if needs_metadata else pnginfo_dict
            pnginfo_dict = fmt_pnginfo
        elif needs_metadata:
            fmt_pnginfo = self.apply_rgthree_context(
                await self.gen_pnginfo(prompt, prefer_nearest, batch_index=0),
                context,
                batch_index=0,
            )
        else:
            fmt_pnginfo = {}

        filename_prefix_fmt = self.format_filename(filename_prefix, fmt_pnginfo, segments) + self.prefix_append
        subdirectory_name = self.format_filename(subdirectory_name, fmt_pnginfo)

        image_shape = images[0].shape
        full_output_folder, filename, counter, subfolder, filename_prefix_fmt = folder_paths.get_save_image_path(
            filename_prefix_fmt, self.output_dir, image_shape[1], image_shape[0]
        )

        subdirectory_name = subdirectory_name.strip()
        if subdirectory_name:
            subdirectory_name = self.format_filename(subdirectory_name, fmt_pnginfo)
            full_output_folder = os.path.join(self.output_dir, subdirectory_name)
            filename = filename_prefix_fmt

        os.makedirs(full_output_folder, exist_ok=True)

        results = list()
        last_image_filename = None

        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # ── Per-image metadata resolution for list-batches ───────────────
            # When multiple images come from a list input, each was generated
            # with a different prompt string.  We resolve metadata fresh for
            # each image, passing batch_number so the correct list entry is
            # selected from the execution cache.
            if is_list_batch and needs_metadata:
                if batch_number == 0:
                    current_pnginfo_dict = fmt_pnginfo
                else:
                    current_pnginfo_dict = self.apply_rgthree_context(
                        await self.gen_pnginfo(
                            prompt, prefer_nearest, batch_index=batch_number
                        ),
                        context,
                        batch_index=batch_number,
                    )
            else:
                current_pnginfo_dict = pnginfo_dict
            # ────────────────────────────────────────────────────────────────

            parameters = self.get_parameters_text(
                current_pnginfo_dict,
                batch_number,
                images_length,
                extra_pnginfo,
                metadata_scope,
            )
            metadata = self.prepare_pnginfo(
                PngInfo(), current_pnginfo_dict, batch_number, images_length,
                prompt, extra_pnginfo, metadata_scope, parameters
            )
            # Extra key/value metadata belongs to the `full` scope. In
            # particular, `parameters_only` must never acquire additional
            # chunks after prepare_pnginfo() has applied its strict filter.
            if metadata is not None and metadata_scope == MetadataScope.FULL:
                for key, value in extra_metadata.items():
                    metadata.add_text(key, value)

            file = f"{filename}_{batch_number:05d}.{base_format}" if include_batch_num else f"{filename}.{base_format}"
            path = os.path.join(full_output_folder, file)

            if os.path.exists(path):
                count = self.find_next_available_filename(full_output_folder, filename, base_format)
                file = f"{filename}_{count:05d}.{base_format}"
                path = os.path.join(full_output_folder, file)

            last_image_filename = file
            quality_value = self.get_quality_value(quality)

            if base_format == "webp":
                img.save(path, "WEBP", lossless=(quality_value == 100), quality=quality_value)
            elif base_format == "png":
                img.save(path, pnginfo=metadata, compress_level=self.compress_level)
            else:
                img.save(path, optimize=True, quality=quality_value)

            if base_format in ["jpg", "webp"] and parameters:
                exif_bytes = piexif.dump({
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                            parameters, encoding="unicode"
                        )
                    }
                })
                piexif.insert(exif_bytes, path)

            results.append({"filename": file, "subfolder": full_output_folder, "type": self.type})

        if (save_workflow_json and images_length > 0 and last_image_filename
                and extra_pnginfo and "workflow" in extra_pnginfo):
            json_filename = Path(last_image_filename).with_suffix(".json").name
            batch_json_file = os.path.join(full_output_folder, json_filename)
            with open(batch_json_file, "w", encoding="utf-8") as f:
                json.dump(extra_pnginfo["workflow"], f)

        return {"ui": {"images": results}, "result": (images,)}

    @staticmethod
    def _context_scalar(value, batch_index=0):
        """Return a simple value from a scalar or per-image context value."""
        if isinstance(value, (list, tuple)):
            if not value or not all(
                item is None or isinstance(item, (str, int, float, bool))
                for item in value
            ):
                return None
            value = value[min(max(batch_index, 0), len(value) - 1)]

        if value is None or not isinstance(value, (str, int, float, bool)):
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @classmethod
    def _context_mapping(cls, context, batch_index=0):
        if isinstance(context, Mapping):
            return context
        if isinstance(context, (list, tuple)) and context:
            selected = context[min(max(batch_index, 0), len(context) - 1)]
            if isinstance(selected, Mapping):
                return selected
        return None

    @classmethod
    def _context_prompt(cls, context, global_key, local_key, batch_index=0):
        parts = []
        for key in (global_key, local_key):
            value = cls._context_scalar(context.get(key), batch_index)
            if isinstance(value, str):
                value = value.strip()
                if value and value not in parts:
                    parts.append(value)
        return "\n".join(parts) if parts else None

    @staticmethod
    def _normalise_number(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return str(int(number)) if number.is_integer() else str(number)

    @staticmethod
    def _model_display_name(value):
        return os.path.splitext(os.path.basename(str(value)))[0]

    @classmethod
    def apply_rgthree_context(cls, pnginfo_dict, context, batch_index=0):
        """Overlay safe metadata fields from an rgthree CONTEXT dictionary.

        Context values are authoritative when present. Model objects, CLIP,
        VAE, latent/image data and conditioning tensors are deliberately not
        copied into metadata.
        """
        merged = dict(pnginfo_dict or {})
        context = cls._context_mapping(context, batch_index)
        if context is None:
            return merged

        for context_key, metadata_key in {
            "seed": "Seed",
            "steps": "Steps",
            "cfg": "CFG scale",
        }.items():
            value = cls._context_scalar(context.get(context_key), batch_index)
            if value is not None:
                merged[metadata_key] = cls._normalise_number(value)

        sampler = cls._context_scalar(context.get("sampler"), batch_index)
        if sampler is not None:
            pretty_sampler = getattr(Capture, "_pretty_sampler", None)
            merged["Sampler"] = (
                pretty_sampler(sampler) if pretty_sampler else str(sampler)
            )

        scheduler = cls._context_scalar(context.get("scheduler"), batch_index)
        if scheduler is not None:
            pretty_scheduler = getattr(Capture, "_pretty_scheduler", None)
            merged["Schedule type"] = (
                pretty_scheduler(scheduler) if pretty_scheduler else str(scheduler)
            )

        model = cls._context_scalar(context.get("ckpt_name"), batch_index)
        if model is not None:
            display_name = cls._model_display_name(model)
            merged["Model"] = display_name
            merged.pop("Model hash", None)
            try:
                from ..defs.formatters import calc_model_hash
                model_hash = calc_model_hash(str(model))
            except Exception:
                model_hash = ""
            if model_hash:
                merged["Model hash"] = model_hash

        positive = cls._context_prompt(
            context, "text_pos_g", "text_pos_l", batch_index
        )
        if positive is not None and not merged.get("Positive prompt"):
            merged["Positive prompt"] = positive

        negative = cls._context_prompt(
            context, "text_neg_g", "text_neg_l", batch_index
        )
        if negative is not None and not merged.get("Negative prompt"):
            merged["Negative prompt"] = negative

        # CLIP target dimensions are useful when graph extraction has no
        # latent size, but they are not guaranteed to equal an already known
        # generation/image size.
        if not merged.get("Size"):
            width = cls._context_scalar(context.get("clip_width"), batch_index)
            height = cls._context_scalar(context.get("clip_height"), batch_index)
            try:
                width = int(float(width))
                height = int(float(height))
            except (TypeError, ValueError):
                width = height = None
            if width and height and width > 0 and height > 0:
                merged["Size"] = f"{width}x{height}"

        return merged

    @staticmethod
    def get_parameters_text(
        pnginfo_dict, batch_number, total_images, extra_pnginfo, metadata_scope
    ):
        metadata_scope = MetadataScope(metadata_scope)
        if metadata_scope not in {
            MetadataScope.FULL,
            MetadataScope.PARAMETERS_ONLY,
        }:
            return ""

        if pnginfo_dict:
            pnginfo_copy = pnginfo_dict.copy()
            if total_images > 1:
                pnginfo_copy["Batch index"] = batch_number
                pnginfo_copy["Batch size"] = total_images
            generated = Capture.gen_parameters_str(pnginfo_copy)
            if generated and "Steps" in generated:
                return generated

        fallback = extra_pnginfo.get("parameters") if extra_pnginfo else None
        return fallback if isinstance(fallback, str) and fallback.strip() else ""

    def prepare_pnginfo(
        self, metadata, pnginfo_dict, batch_number, total_images, prompt,
        extra_pnginfo, metadata_scope, parameters=None
    ):
        metadata_scope = MetadataScope(metadata_scope)
        if metadata_scope == MetadataScope.NONE:
            return None

        if parameters is None:
            parameters = self.get_parameters_text(
                pnginfo_dict,
                batch_number,
                total_images,
                extra_pnginfo,
                metadata_scope,
            )

        # This scope is exclusive even when our own graph extraction is empty.
        # Falling through here previously embedded the complete prompt and
        # workflow. If another extension already supplied an A1111 parameters
        # string through extra_pnginfo, retain only that safe fallback.
        if metadata_scope == MetadataScope.PARAMETERS_ONLY:
            if parameters:
                metadata.add_text("parameters", parameters)
            return metadata

        if metadata_scope == MetadataScope.FULL and parameters:
            metadata.add_text("parameters", parameters)

        if metadata_scope in [MetadataScope.FULL, MetadataScope.DEFAULT]:
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    # Parameters are always stored as raw A1111 infotext in
                    # full mode, never as a JSON-quoted duplicate.
                    if metadata_scope == MetadataScope.FULL and x == "parameters":
                        continue
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
        elif metadata_scope == MetadataScope.WORKFLOW_ONLY:
            workflow = extra_pnginfo.get("workflow") if extra_pnginfo else None
            if workflow is not None:
                metadata.add_text("workflow", json.dumps(workflow))

        return metadata

    @classmethod
    async def gen_pnginfo(cls, prompt, prefer_nearest, batch_index=0):
        inputs = await Capture.get_inputs(batch_index=batch_index)
        save_node_id = hook.current_save_image_node_id
        context_node_id = cls._linked_prompt_input_source(
            prompt, save_node_id, "context"
        )
        image_node_id = cls._linked_prompt_input_source(
            prompt, save_node_id, "images"
        )

        # Connecting metadata context adds a second ancestry branch to the
        # Save node. Keep automatic model/LoRA/VAE/hires discovery anchored to
        # the actual image branch; context values are merged explicitly below.
        trace_start_node_id = (
            image_node_id
            if context_node_id is not None and image_node_id is not None
            else save_node_id
        )
        trace_tree_from_this_node = Trace.trace(trace_start_node_id, prompt)
        inputs_before_this_node = Trace.filter_inputs_by_trace_tree(inputs, trace_tree_from_this_node, prefer_nearest)

        sampler_node_id = Trace.find_sampler_node_id(trace_tree_from_this_node)
        if sampler_node_id:
            trace_tree_from_sampler_node = Trace.trace(sampler_node_id, prompt)
            inputs_before_sampler_node = Trace.filter_inputs_by_trace_tree(inputs, trace_tree_from_sampler_node, prefer_nearest)
        else:
            inputs_before_sampler_node = {}

        context_prompts = None
        if context_node_id is not None:
            resolver = getattr(Capture, "resolve_context_prompts", None)
            if resolver is not None:
                context_prompts = resolver(
                    prompt, context_node_id, batch_index=batch_index
                )

        pnginfo = Capture.gen_pnginfo_dict(
            inputs_before_sampler_node, inputs_before_this_node, prompt,
            batch_index=batch_index, sampler_node_id=sampler_node_id,
            active_trace_tree=trace_tree_from_this_node,
            prompt_overrides=context_prompts,
        )
        return pnginfo

    @classmethod
    def format_filename(cls, filename, pnginfo_dict, segments=None):
        if "%" not in filename:
            return filename

        segments = segments or re.findall(cls.pattern_format, filename)
        now = datetime.now()
        date_table = {
            "yyyy": f"{now.year}",
            "MM": f"{now.month:02d}",
            "dd": f"{now.day:02d}",
            "hh": f"{now.hour:02d}",
            "mm": f"{now.minute:02d}",
            "ss": f"{now.second:02d}",
        }

        for segment in segments:
            parts = segment.strip("%").split(":")
            key = parts[0]

            if key == "seed":
                seed = pnginfo_dict.get("Seed")
                if seed is None:
                    print_warning("Seed not found in pnginfo_dict!")
                filename = filename.replace(segment, str(seed or ""))

            elif key in {"width", "height"}:
                size = pnginfo_dict.get("Size", "x").split("x")
                if "Size" not in pnginfo_dict:
                    print_warning("Size not found in pnginfo_dict!")
                value = size[0] if key == "width" else size[1]
                filename = filename.replace(segment, value)

            elif key in {"pprompt", "nprompt"}:
                prompt_key = "Positive prompt" if key == "pprompt" else "Negative prompt"
                prompt_val = pnginfo_dict.get(prompt_key, "")
                if not prompt_val:
                    print_warning(f"{prompt_key} not found in pnginfo_dict!")
                prompt_val = prompt_val.replace("\n", " ")
                length = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                filename = filename.replace(segment, prompt_val[:length].strip() if length else prompt_val.strip())

            elif key == "model":
                model = pnginfo_dict.get("Model", "")
                if not model:
                    print_warning("Model not found in pnginfo_dict!")
                model = os.path.splitext(os.path.basename(model))[0]
                length = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                filename = filename.replace(segment, model[:length] if length else model)

            elif key == "date":
                date_format = parts[1] if len(parts) > 1 else "yyyyMMddhhmmss"
                for k, v in date_table.items():
                    date_format = date_format.replace(k, v)
                filename = filename.replace(segment, date_format)

        return filename


class CreateExtraMetaData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "extra_metadata": ("EXTRA_METADATA", {"forceInput": True}),
                **{
                    f"{type}{i}": ("STRING", {"default": "", "multiline": False})
                    for i in range(1, 5)
                    for type in ["key", "value"]
                },
            }
        }

    RETURN_TYPES = ("EXTRA_METADATA",)
    FUNCTION = "create_extra_metadata"
    DESCRIPTION = "Creates custom extra metadata by adding key-value pairs. Empty values are allowed, but unpaired values are not."
    CATEGORY = "Cyberdelia/Metadata"

    def create_extra_metadata(self, extra_metadata=None, **keys_values):
        if extra_metadata is None:
            extra_metadata = {}

        for i in range(1, 5):
            key = keys_values.get(f"key{i}", "").strip()
            value = keys_values.get(f"value{i}", "").strip()

            if key:
                extra_metadata[key] = value
            elif value:
                raise ValueError(f"Value provided for 'value{i}' without corresponding 'key{i}'.")

        return (extra_metadata,)
