import re
import folder_paths
import numpy as np
import os.path
import json, os, math, io, random, struct

from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont
from comfy.cli_args import args
from comfy.utils import load_torch_file
from comfy.sd import load_lora_for_models
from PIL.PngImagePlugin import PngInfo
from .grid_types import Annotation
# Optional EXIF writer (JPEG/WEBP)
try:
    import piexif, piexif.helper
except Exception:
    piexif = None



static_x = 1
static_y = 1

reset_registry = {}

class GridFloats:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": {
                "index": ( "INT", {"default": 1, "min": 1 } ),
                "decimal_places": ("INT", {"default": 3, "min": 1}),
                "float1": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "float2": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "float3": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "float4": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "float5": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "float6": ("FLOAT", {"default": 1.0, "step": 0.01}),
            },
             }

    RETURN_TYPES = ("FLOAT","FLOAT_LIST","STRING_LIST")
    RETURN_NAMES = ("current value", "list", "list text")
    FUNCTION = "ReturnFloat"
    CATEGORY = "EasyGrids"

    def ReturnFloat( self, index: int, decimal_places: int, float1 : float, float2 : float, float3 : float, float4 : float, float5 : float, float6: float ) -> tuple[float, list[float], list[str]]:
        #TODO: probably a more pythonic way to do this
        ret_list = [float1, float2, float3, float4, float5, float6]
        ret_val = 0.0
        if ( index > len(ret_list) ):
            ret_val = ret_list[-1]
        else:
            ret_val = ret_list[ index - 1 ]
        return (ret_val, ret_list, ["{:.{}f}".format(val, decimal_places) for val in ret_list] )

class GridFloatList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ( "INT", {"default": 1, "min": 1, "max": 100 } ),
                "decimal_places": ("INT", {"default": 3, "min": 1}),
                "float_list": ("STRING", {"multiline": True}),
        }}

    RETURN_TYPES = ("FLOAT","FLOAT_LIST","STRING_LIST")
    RETURN_NAMES = ("current value", "list", "list text")
    FUNCTION = "ParseAndReturnFloat"
    CATEGORY = "EasyGrids"

    def __init__(self):
        self.static_text = "" 
        self.static_out_arr = []

    def ParseAndReturnFloat( self, index: int, decimal_places: int, float_list: str ) -> tuple[float, list[float], list[str]]:
        if float_list != self.static_text:
            split_str = re.split( ",|;|\s|:", float_list )
            out_arr = []
            for val in split_str:
                # let the exception happen if invalid
                out_arr.append(float(val))
            self.static_text = float_list
            self.static_out_arr = deepcopy( out_arr )
        ret_val = 0.0
        if ( index > len(self.static_out_arr) ):
            ret_val = self.static_out_arr[-1]
        else:
            ret_val = self.static_out_arr[ index - 1 ]
        return (ret_val,self.static_out_arr, ["{:.{}f}".format(val, decimal_places) for val in self.static_out_arr])

class GridInts:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ( "INT", {"default": 1, "min": 1 }),
                "int1": ("INT", {"default": 1, "step": 1}),
                "int2": ("INT", {"default": 1, "step": 1}),
                "int3": ("INT", {"default": 1, "step": 1}),
                "int4": ("INT", {"default": 1, "step": 1}),
                "int5": ("INT", {"default": 1, "step": 1}),
                "int6": ("INT", {"default": 1, "step": 1}),
            },
             }

    RETURN_TYPES = ("INT","INT_LIST","STRING_LIST")
    RETURN_NAMES = ("current value","list", "list text")
    FUNCTION = "ReturnInt"
    CATEGORY = "EasyGrids"

    def ReturnInt( self, index: int, int1 : int, int2 : int, int3 : int, int4 : int, int5 : int, int6: int )-> tuple[int, list[int], list[str]]:
        ret_list = [int1, int2, int3, int4, int5, int6]
        ret_val = 0
        if ( index > len(ret_list) ):
            ret_val = ret_list[-1]
        else:
            ret_val = ret_list[ index - 1 ]
        return (ret_val, ret_list, [str(val) for val in ret_list])

class GridIntList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ( "INT", {"default": 1, "min": 1, "max": 100 } ),
                "int_list": ("STRING", {"multiline": True}),
        }}

    RETURN_TYPES = ("INT","INT_LIST","STRING_LIST")
    RETURN_NAMES = ("current value","list","list text")
    FUNCTION = "ParseAndReturnInt"
    CATEGORY = "EasyGrids"

    def __init__(self):
        self.static_text = "" 
        self.static_out_arr = []

    def ParseAndReturnInt( self, index: int, int_list: str ) -> tuple[int, list[int], list[str]]:
        if int_list != self.static_text:
            split_str = re.split( ",|;|\s|:", int_list )
            out_arr = []
            for val in split_str:
                # let the exception happen if invalid
                out_arr.append(int(val))
            self.static_text = int_list
            self.static_out_arr = deepcopy( out_arr )
        ret_val = 0
        if ( index > len(self.static_out_arr) ):
            ret_val = self.static_out_arr[-1]
        else:
            ret_val = self.static_out_arr[ index - 1 ]
        return (ret_val, self.static_out_arr, [str(val) for val in self.static_out_arr])

class GridStrings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ( "INT", {"default": 1, "min": 1 }),
                "string1": ("STRING", {"default": ""}),
                "string2": ("STRING", {"default": ""}),
                "string3": ("STRING", {"default": ""}),
                "string4": ("STRING", {"default": ""}),
                "string5": ("STRING", {"default": ""}),
                "string6": ("STRING", {"default": ""}),
            },
             }

    RETURN_TYPES = ("STRING","STRING_LIST")
    RETURN_NAMES = ("current value","list")
    FUNCTION = "ReturnString"
    CATEGORY = "EasyGrids"

    def ReturnString( self, index: int, string1 : str, string2 : str, string3 : str, string4 : str, string5 : str, string6: str ) -> tuple[str, list[str]]:
        ret_list = [string1, string2, string3, string4, string5, string6]
        if ( index > len(ret_list) ):
            return ( ret_list[len(ret_list) - 1], ret_list)
        return (ret_list[ index - 1 ], ret_list)

class GridStringList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ( "INT", {"default": 1, "min": 1, "max": 100 } ),
                "string_list": ("STRING", {"multiline": True}), 
            }
        }

    RETURN_TYPES = ("STRING","STRING_LIST")
    RETURN_NAMES = ("current value","list")
    FUNCTION = "SplitAndReturnStrings"
    CATEGORY = "EasyGrids"

    def __init__(self):
        self.static_text = "" 
        self.static_out_arr = []

    def SplitAndReturnStrings( self, index: int, string_list: str ) -> tuple[str, list[str]]:
        if string_list != self.static_text:
            # unlike the numeric list nodes, we only want to split on newlines
            # TODO: support manual delimiter specification?
            split_str = re.split( "\n", string_list )
            self.static_text = string_list
            self.static_out_arr = deepcopy( split_str )
        if ( index > len(self.static_out_arr) ):
            return ( self.static_out_arr[len(self.static_out_arr) - 1], self.static_out_arr)
        return (self.static_out_arr[ index - 1 ], self.static_out_arr)

class GridLoras:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        lora_list = folder_paths.get_filename_list("loras")
        lora_list.insert(0, "None")
        return {
            "required": {
                "index": ( "INT", {"default": 1, "min": 1 }),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora1": (lora_list, ),
                "lora2": (lora_list, ),
                "lora3": (lora_list, ),
                "lora4": (lora_list, ),
                "lora5": (lora_list, ),
                "lora6": (lora_list, ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }}
    
    RETURN_TYPES = ("MODEL","CLIP","STRING_LIST")
    RETURN_NAMES = ("current model","current clip","lora name list")
    FUNCTION = "ReturnLora"
    CATEGORY = "EasyGrids"

    def ReturnLora( self, index: int, model, clip, lora1 : str, lora2 : str, lora3 : str, lora4 : str, lora5 : str, lora6: str, strength_model: float, strength_clip: float):
        ret_list = [lora1, lora2, lora3, lora4, lora5, lora6]
        target_name = ""
        if ( index > len(ret_list) ):
            target_name = ret_list[-1]
        else:
            target_name = ret_list[ index - 1 ]
        if target_name == "None" or ( strength_model == 0.0 and strength_clip == 0.0 ):
            return (model, clip, ret_list)
        
        lora_path = folder_paths.get_full_path("loras", target_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp
        if lora is None:
            lora = load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)                
        model_lora, clip_lora = load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora, ret_list)

                

class ImageGridCommander:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "x_count": ("INT", {"default": 1, "min": 1, "step": 1}),
                          "y_count": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT","INT","INT","INT",)
    RETURN_NAMES = ("x_index", "y_index", "x_size", "y_size",)
    FUNCTION = "queue_batch"
    CATEGORY = "EasyGrids"

    def __init__(self):
        self.curr_x_index = 1
        self.curr_y_index = 1
        self.last_x_count = 0
        self.last_y_count = 0
        self.unique_id = None
    
    def __del__(self):
        if self.unique_id is not None and self is reset_registry.get(self.unique_id, None):
            reset_registry.pop(self.unique_id, None)

    def queue_batch(self, x_count, y_count, unique_id )-> tuple[int, int, int, int]:
        #wish we could do this on init but there doesn't seem to be a way to get the unique_id at that point
        #there shouldn't be any need to reset before the first run in any case
        if unique_id is not None:
            if unique_id != self.unique_id:
                if self.unique_id is not None and self is reset_registry.get(self.unique_id, None):
                    reset_registry.pop(self.unique_id, None)
                self.unique_id = unique_id
        if self.unique_id not in reset_registry and self.unique_id is not None:
            reset_registry[unique_id] = self
        if x_count != self.last_x_count or y_count != self.last_y_count:
            self.last_x_count = x_count
            self.last_y_count = y_count
            self.curr_x_index = 1
            self.curr_y_index = 1
        last_x_index = self.curr_x_index
        last_y_index = self.curr_y_index
        self.curr_x_index += 1
        if self.curr_x_index > x_count:
            self.curr_x_index = 1
            self.curr_y_index += 1
            if self.curr_y_index > y_count:
                self.curr_y_index = 1
        return (last_x_index, last_y_index, x_count, y_count,)

    # This node will always be run
    @classmethod
    def IS_CHANGED( s, x_count, y_count ):
        return float("NaN")

    def reset(self):
        self.curr_x_index = 1
        self.curr_y_index = 1
        self.last_x_count = 0
        self.last_y_count = 0

class TextConcatenator:
    @classmethod
    def INPUT_TYPES(s):
        return { "required" : { "text_1": ("STRING", {"multiline": True}) },
                 "optional" : { "text_2": ("STRING", {"multiline": True}),
                                "text_3": ("STRING", {"multiline": True}),
                                "text_4": ("STRING", {"multiline": True}) } }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "concat_text"
    CATEGORY = "EasyGrids"

    def concat_text( self, text_1, text_2, text_3, text_4 )-> tuple[str]:
        #simple as!
        ret_accum = text_1
        if text_2 is not None:
            ret_accum += text_2
        if text_3 is not None:
            ret_accum += text_3
        if text_4 is not None:
            ret_accum += text_4
        return (ret_accum, )

class FloatToText:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "float_input": ("FLOAT", {"default": 1.0, "step": 0.01}), 
                               "decimal_places": ("INT", {"default": 3, "min": 1, "max": 10 }), }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert_to_str"
    CATEGORY = "EasyGrids"

    def convert_to_str(self, float_input : float, decimal_places : int) -> tuple[str]:
        # if this doesn't work, blame Copilot
        formatted_float = "{:.{}f}".format(float_input, decimal_places)
        return (formatted_float,)

class IntToText:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "int_input": ("INT", {"default": 1}), }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert_to_str"
    CATEGORY = "EasyGrids"

    def convert_to_str(self, int_input : int) -> tuple[str]:
        return (str(int_input),)

class SaveImageGrid:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.image_grid = []
        self.curr_x_size = 1
        self.curr_x_idx = 0
        self.curr_y_size = 1
        self.curr_y_idx = 0
        self.done_flag = False
        self.unique_id = None
    
    def __del__(self):
        if self.unique_id is not None and self is reset_registry.get(self.unique_id, None):
            reset_registry.pop(self.unique_id, None)

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE",),
                     "x_size": ("INT", {"default": 1, "min": 1, "step": 1}),
                     "y_size": ("INT", {"default": 1, "min": 1, "step": 1}),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                     # New save options
                     "file_format": (["PNG", "WEBP", "JPEG"], {"default": "WEBP"}),
                     "quality": ("INT", {"default": 92, "min": 1, "max": 100}),
                     "webp_lossless": ("BOOLEAN", {"default": False}),
                     "optimize": ("BOOLEAN", {"default": True})},
                "optional" : { "column_labels": ("STRING_LIST", {"default": None}),
                               "row_labels": ("STRING_LIST", {"default": None }),
                               "images_grid_annotation": ("GRID_ANNOTATION",)}, 
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID" },
                }

    RETURN_TYPES = ()
    FUNCTION = "accumulate_images"

    OUTPUT_NODE = True

    def accumulate_images(self, images, x_size, y_size, filename_prefix="ComfyUI",
                          file_format="WEBP", quality=92, webp_lossless=False, optimize=True,
                          prompt=None, extra_pnginfo=None,
                          column_labels=None, row_labels=None, images_grid_annotation=None, unique_id=None):
        if unique_id is not None:
            if unique_id != self.unique_id:
                if self.unique_id is not None and self is reset_registry.get(self.unique_id, None):
                    reset_registry.pop(self.unique_id, None)
                self.unique_id = unique_id
        if self.unique_id not in reset_registry and self.unique_id is not None:
            reset_registry[unique_id] = self
        filename_prefix += self.prefix_append
        if x_size != self.curr_x_size or y_size != self.curr_y_size or self.done_flag:
            self.curr_x_size = x_size
            self.curr_y_size = y_size
            self.reset()
        for image in images:
            self.image_grid.append(image)
            self.curr_x_idx += 1
            if self.curr_x_idx >= self.curr_x_size:
                self.curr_y_idx += 1
                self.curr_x_idx = 0
        
                
        if len(self.image_grid) >= self.curr_x_size * self.curr_y_size:
            #complete grid
            if images_grid_annotation is not None:
                column_labels = images_grid_annotation.column_texts
                row_labels = images_grid_annotation.row_texts
            grid_image = self.assemble_grid( column_labels, row_labels )
            return self.save_grid(grid_image, filename_prefix,
                                  file_format=file_format, quality=quality, webp_lossless=webp_lossless, optimize=optimize,
                                  prompt=prompt, extra_pnginfo=extra_pnginfo)
        return { "ui": { "images": [] } }

    def assemble_grid(self, column_labels: list[str] | None = None, row_labels: list[str] | None = None) -> Image:
        space_height = max( [ len(image) for image in self.image_grid ] )
        space_width = max( [ len(image[0]) for image in self.image_grid ] )
        total_width = space_width * self.curr_x_size
        total_height = space_height * self.curr_y_size
        width_padding = 0
        height_padding = 0
        #TODO: font size input, font type input, etc.? Also smarter path detection
        font_path = os.path.join(os.path.dirname(__file__), "fonts/Roboto-Regular.ttf")
        label_font = ImageFont.truetype(str(font_path), size=50)
        if column_labels is not None and len(column_labels) > 0:
            height_padding = int(max( [ ( label_font.getbbox( text )[3] - label_font.getbbox( text )[1] ) for text in column_labels ] )  * 1.5)
        if row_labels is not None and len(row_labels) > 0:
            width_padding = int(max( [ label_font.getlength( text ) for text in row_labels ] ) * 1.5)
        total_width += width_padding

        total_height += height_padding
        grid_canvas = Image.new("RGB", (total_width, total_height), color="#ffffff")
        draw = ImageDraw.Draw(grid_canvas)
        for y_idx in range(self.curr_y_size):
            if row_labels is not None and y_idx < len(row_labels):
                row_x_anchor = width_padding / 2
                row_y_anchor = height_padding + space_height * y_idx + (space_height / 2)
                draw.text((row_x_anchor, row_y_anchor), row_labels[y_idx], anchor="mm", font=label_font, fill="#000000")
            for x_idx in range(self.curr_x_size):
                if y_idx == 0 and column_labels is not None and x_idx < len(column_labels):
                    col_x_anchor = width_padding + space_width * x_idx + (space_width / 2)
                    col_y_anchor = height_padding / 2
                    draw.text((col_x_anchor, col_y_anchor), column_labels[x_idx], anchor="mm", font=label_font, fill="#000000")
                pil_image = Image.fromarray(np.clip((self.image_grid[(y_idx * self.curr_x_size) + x_idx].cpu().numpy() * 255.), 0, 255).astype(np.uint8))
                grid_canvas.paste(pil_image, ((x_idx * space_width) + width_padding, (y_idx * space_height) + height_padding))
        return grid_canvas

    def save_grid(self, grid_image, filename_prefix,
                  file_format="WEBP", quality=92, webp_lossless=False, optimize=True,
                  prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, grid_image.width, grid_image.height)
        results = list()

        fmt = str(file_format).upper() if file_format is not None else "PNG"
        if fmt not in ("PNG", "WEBP", "JPEG"):
            fmt = "PNG"
        ext = "png" if fmt == "PNG" else ("webp" if fmt == "WEBP" else "jpg")

        # Always compute output paths and prepare EXIF holder
        file = f"{filename}_{counter:05}_.{ext}"
        out_path = os.path.join(full_output_folder, file)
        exif_bytes = None

        # Prepare PNG metadata if enabled
        png_metadata = None
        if fmt == "PNG" and not args.disable_metadata:
            png_metadata = PngInfo()
            if prompt is not None:
                png_metadata.add_text("prompt", json.dumps(prompt))
            if isinstance(extra_pnginfo, dict):
                for k, v in extra_pnginfo.items():
                    png_metadata.add_text(k, json.dumps(v))

        # Build EXIF (UserComment) JSON payload for WEBP/JPEG
        # Stored in EXIF tag 0x9286 (UserComment). If Pillow drops it, we try piexif.insert().
        if fmt in ("WEBP", "JPEG") and not args.disable_metadata and piexif is not None:
            try:
                payload = {}
                if prompt is not None:
                    payload["prompt"] = prompt
                if isinstance(extra_pnginfo, dict):
                    if "workflow" in extra_pnginfo:
                        payload["workflow"] = extra_pnginfo.get("workflow")
                    # carry any other fields too
                    for k, v in extra_pnginfo.items():
                        if k != "workflow":
                            payload[k] = v
                user_comment = json.dumps(payload, ensure_ascii=False)
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
                exif_dict["0th"][piexif.ImageIFD.Software] = "ComfyUI Save Image Grid"
                exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(
                    user_comment, encoding="unicode"
                )
                exif_bytes = piexif.dump(exif_dict)
            except Exception:
                exif_bytes = None



        if fmt == "PNG":
            grid_image.save(out_path, pnginfo=png_metadata, compress_level=4)
        elif fmt == "WEBP":
            # grid is RGB already; WEBP supports quality/lossless
            save_kwargs = {"format": "WEBP", "quality": int(quality), "lossless": bool(webp_lossless), "method": 6}
            if exif_bytes is not None:
                save_kwargs["exif"] = exif_bytes
            grid_image.save(out_path, **save_kwargs)
            # If Pillow dropped EXIF for WEBP, try forced insert
            if exif_bytes is not None and piexif is not None:
                try:
                    piexif.insert(exif_bytes, out_path)
                except Exception:
                    pass
        else:
            # JPEG must be 3-channel, no alpha; optional optimize/progressive
            if grid_image.mode not in ("RGB", "L"):
                grid_image = grid_image.convert("RGB")
            save_kwargs = {"format": "JPEG", "quality": int(quality), "optimize": bool(optimize), "progressive": True, "subsampling": 1}
            if exif_bytes is not None:
                save_kwargs["exif"] = exif_bytes
            try:
                grid_image.save(out_path, **save_kwargs)
            except ValueError as e:
                # Typical Pillow error: ValueError("EXIF data is too long")
                if "EXIF data is too long" in str(e):
                    save_kwargs.pop("exif", None)
                    grid_image.save(out_path, **save_kwargs)
                else:
                    raise

            # Always embed the full metadata inside JPEG using COM segments (readable text).
            # We store UTF-8 JSON across multiple COM segments labeled COMFYMDv1 utf8.
            if not args.disable_metadata and (prompt is not None or extra_pnginfo is not None):
                try:
                    payload = {"prompt": prompt}
                    if isinstance(extra_pnginfo, dict):
                        for k, v in extra_pnginfo.items():
                            payload[k] = v
                    text = json.dumps(payload, ensure_ascii=False)
                    self._jpeg_insert_com_segments_text(out_path, label="comfy", text=text, chunk_size=60000)

                except Exception:
                    # Non-fatal; image already saved
                    pass

        # Keep sidecar for maximum compatibility (apps may strip EXIF on share)
        if fmt != "PNG" and not args.disable_metadata and (prompt is not None or extra_pnginfo is not None):
            sidecar = {
                "prompt": prompt if prompt is not None else None,
                "extra_pnginfo": extra_pnginfo if extra_pnginfo is not None else None
            }
            try:
                with open(os.path.join(full_output_folder, f"{filename}_{counter:05}_.json"), "w", encoding="utf-8") as f:
                    json.dump(sidecar, f, ensure_ascii=False, indent=2)
            except Exception:
                # Non-fatal: image already saved
                pass

        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        self.done_flag = True
        return { "ui": { "images": results } }

    def reset( self ):
        self.curr_x_idx = 0
        self.curr_y_idx = 0
        self.image_grid = []
        self.done_flag = False

    @staticmethod
    def _jpeg_insert_com_segments_text(jpeg_path: str, label: str, text: str, chunk_size: int = 60000) -> None:
        """
        Insert one or more JPEG COM segments carrying plain UTF-8 JSON metadata.
        Segment format (human readable in exiftool):
          "COMFYMDv1 utf8 {label} {idx}/{total}\n" + json_chunk
        """
        with open(jpeg_path, "rb") as f:
            buf = f.read()
        if len(buf) < 4 or buf[0:2] != b"\xFF\xD8":
            return  # not a JPEG

        # Find insertion point before SOS (0xFFDA)
        i = 2
        insert_at = 2
        n = len(buf)
        while i + 4 <= n and buf[i] == 0xFF:
            marker = buf[i + 1]
            i += 2
            # Standalone markers are not expected in header; treat as abnormal and stop
            if marker == 0xDA:  # SOS - start of scan
                insert_at = i - 2
                break
            if i + 2 > n:
                break
            seglen = struct.unpack(">H", buf[i:i + 2])[0]
            if seglen < 2 or i + seglen > n:
                break
            i += seglen
            insert_at = i  # keep sliding until SOS

        # Prepare chunks (UTF-8 bytes of JSON)
        raw = text.encode("utf-8")
        total = (len(raw) + chunk_size - 1) // chunk_size if len(raw) else 1
        segments = []
        for idx in range(total):
            chunk = raw[idx * chunk_size:(idx + 1) * chunk_size]
            header = f"COMFYMDv1 utf8 {label} {idx + 1}/{total}\n".encode("ascii")
            payload = header + chunk
            seg = b"\xFF\xFE" + struct.pack(">H", len(payload) + 2) + payload
            segments.append(seg)

        patched = buf[:insert_at] + b"".join(segments) + buf[insert_at:]
        with open(jpeg_path, "wb") as f:
            f.write(patched)


NODE_CLASS_MAPPINGS = {
    "ImageGridCommander": ImageGridCommander,
    "GridFloats": GridFloats,
    "GridFloatList": GridFloatList,
    "GridInts": GridInts,
    "GridIntList": GridIntList,
    "GridStrings": GridStrings,
    "GridStringList": GridStringList,
    "GridLoras": GridLoras,
    "TextConcatenator": TextConcatenator,
    "FloatToText": FloatToText,
    "IntToText": IntToText,
    "SaveImageGrid": SaveImageGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageGridCommander": "Create Image Grid",
    "GridFloats" : "Float List",
    "GridFloatList": "Float List fom Text Field",
    "GridInts": "Int List",
    "GridIntList": "Int List from Text Field",
    "GridStrings": "String List",
    "GridStringList": "String List from Text Field",
    "GridLoras": "Lora List",
    "TextConcatenator": "Text Concatenator",
    "FloatToText": "Float to Text",
    "IntToText": "Int to Text",
    "SaveImageGrid": "Save Image Grid",
}