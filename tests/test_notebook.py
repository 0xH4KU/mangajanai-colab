import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

import nbformat
import numpy as np
import pymupdf
import torch
from PIL import Image, PngImagePlugin


NOTEBOOK = Path(__file__).resolve().parents[1] / "MangaJaNai_Colab.ipynb"
torch.set_num_threads(1)


class SmallModel:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self, smooth=False):
        self.calls = 0
        self.smooth = smooth

    def __call__(self, tensor):
        self.calls += 1
        if self.smooth:
            kernel = torch.ones(3, 1, 3, 3) / 9
            tensor = torch.nn.functional.conv2d(tensor, kernel, padding=1, groups=3)
        return torch.nn.functional.interpolate(tensor, scale_factor=2, mode="nearest")


class TestPool:
    def __init__(self, namespace, model, requests):
        self.ns = namespace
        self.selection = namespace["MODEL_SELECTION"]
        self.illustration_selection = namespace["ILLUSTRATION_SELECTION"]
        self.model_path = namespace["MODEL_PATH"].strip()
        self.cache = Path(namespace["OUTPUT_DIR"]) / "models"
        self.model = model
        self.requests = requests

    def get(self, height):
        self.requests.append(height)
        return self.model


class TestDescriptor:
    scale = 2
    supports_half = True
    device = torch.device("cpu")

    def eval(self):
        return self

    half = cuda = eval


def image_bytes(value=80, size=(19, 27), color=False, format="PNG"):
    stream = io.BytesIO()
    Image.new("RGB" if color else "L", size, value).save(stream, format=format)
    return stream.getvalue()


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.input = self.root / "input"
        self.output = self.root / "output"
        self.input.mkdir()
        notebook = json.loads(NOTEBOOK.read_text())
        self.ns = {}
        for cell in notebook["cells"]:
            if cell["id"] in {"config", "functions"}:
                exec(cell["source"], self.ns)
        self.ns.update(INPUT_DIR=str(self.input), OUTPUT_DIR=str(self.output))
        self.model = SmallModel()
        self.requests = []

    def run_batch(self, limit=None):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            self.ns["run_batch"](TestPool(self.ns, self.model, self.requests), limit)
        self.last_output = output.getvalue()

    def test_notebook_structure_and_python(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, f"cell:{cell.id}", "exec")

    def test_tiles_match_whole_page_at_odd_edges(self):
        pixels = np.random.default_rng(1).integers(0, 256, size=(73, 91), dtype=np.uint8)
        image = Image.fromarray(pixels).convert("RGB")
        model = SmallModel(smooth=True)
        whole = self.ns["infer"](image, model, 0)
        tiled = self.ns["infer"](image, model, 16)
        self.assertEqual(tiled.size, (182, 146))
        np.testing.assert_array_equal(np.asarray(whole), np.asarray(tiled))

    def test_oom_retries_then_stops_at_minimum(self):
        sizes = []

        def run(image, model, tile):
            sizes.append(tile)
            if tile != 128:
                raise torch.cuda.OutOfMemoryError("test")
            return image

        image = Image.new("RGB", (3, 3))
        with patch.dict(self.ns, infer=run), patch("torch.cuda.empty_cache"), contextlib.redirect_stdout(io.StringIO()):
            result, tile = self.ns["upscale"](image, self.model, 0)
        self.assertIs(result, image)
        self.assertEqual(sizes, [0, 512, 256, 128])
        self.assertEqual(tile, 128)
        with patch.dict(self.ns, infer=Mock(side_effect=torch.cuda.OutOfMemoryError("test"))):
            with self.assertRaisesRegex(RuntimeError, "128"):
                self.ns["upscale"](image, self.model, 128)

    def test_nan_never_becomes_a_saved_black_page(self):
        model = Mock(device=torch.device("cpu"), dtype=torch.float32)
        model.side_effect = lambda tensor: torch.full(
            (1, 3, tensor.shape[2] * 2, tensor.shape[3] * 2), float("nan")
        )
        with self.assertRaises(FloatingPointError):
            self.ns["infer"](Image.new("RGB", (5, 7)), model, 0)

    def test_unsafe_archive_names_and_duplicates_are_rejected(self):
        for name in ("../escape.png", "/absolute.png", "C:\\escape.png", "a/../../x", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.ns["safe_member"](name)
        self.assertEqual(self.ns["safe_member"]("chapter\\001.png"), "chapter/001.png")
        with ZipFile(self.input / "bad.cbz", "w") as archive:
            archive.writestr("../escape.png", image_bytes())
        with self.assertRaises(ValueError):
            self.run_batch()
        self.assertFalse((self.root / "escape.png").exists())
        (self.input / "bad.cbz").unlink()
        with ZipFile(self.input / "duplicate.cbz", "w") as archive:
            archive.writestr("chapter/001.png", image_bytes())
            archive.writestr("chapter\\001.png", image_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_batch()

    def test_archive_resume_metadata_color_and_name_collisions(self):
        source = self.input / "Book.cbz"
        with ZipFile(source, "w") as archive:
            archive.writestr("chapter/001.jpg", image_bytes(format="JPEG"))
            archive.writestr("chapter/001.png", image_bytes(value=120))
            archive.writestr("chapter/002.png", image_bytes((230, 30, 10), color=True))
            archive.writestr("ComicInfo.xml", "<ComicInfo><Title>Test</Title></ComicInfo>")
            archive.writestr("__MACOSX/._001.jpg", b"not a page")
        original = source.read_bytes()
        self.run_batch(limit=1)
        self.assertFalse((self.output / "books" / "Book.cbz.2x.zip").exists())
        self.assertEqual(self.model.calls, 1)
        self.run_batch()
        self.assertEqual(self.model.calls, 3)
        with ZipFile(self.output / "books" / "Book.cbz.2x.zip") as archive:
            self.assertEqual(set(archive.namelist()), {
                "chapter/001.jpg.png", "chapter/001.png.png", "chapter/002.png.png", "ComicInfo.xml"
            })
            self.assertEqual(archive.read("ComicInfo.xml"), b"<ComicInfo><Title>Test</Title></ComicInfo>")
            with Image.open(io.BytesIO(archive.read("chapter/001.jpg.png"))) as image:
                self.assertEqual(image.size, (38, 54))
                self.assertEqual(image.mode, "L")
            with Image.open(io.BytesIO(archive.read("chapter/002.png.png"))) as image:
                self.assertEqual(image.size, (38, 54))
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.getpixel((0, 0)), (230, 30, 10))
        self.run_batch()
        self.assertEqual(self.model.calls, 3)
        self.assertEqual(source.read_bytes(), original)

    def test_changed_source_settings_corruption_and_partial_resume(self):
        source = self.input / "chapter" / "001.png"
        source.parent.mkdir()
        source.write_bytes(image_bytes())
        output = self.output / "pages" / "_images" / "chapter" / "001.png.png"
        output.parent.mkdir(parents=True)
        output.with_name(output.name + ".part").write_bytes(b"interrupted")
        self.run_batch()
        self.run_batch()
        self.assertEqual(self.model.calls, 1)
        source.write_bytes(image_bytes(value=160))
        self.run_batch()
        self.assertEqual(self.model.calls, 2)
        with Image.open(output) as image:
            self.assertEqual(image.getpixel((0, 0)), 160)
        self.ns["TILE_SIZE"] = 512
        self.run_batch()
        self.assertEqual(self.model.calls, 3)
        output.write_bytes(output.read_bytes()[:100])
        self.run_batch()
        self.assertEqual(self.model.calls, 3)
        self.ns["completion_path"](self.output / "books" / "images.2x.zip").unlink()
        self.run_batch()
        self.assertEqual(self.model.calls, 4)

    def test_failure_keeps_previous_complete_book(self):
        source = self.input / "Book.cbz"
        with ZipFile(source, "w") as archive:
            archive.writestr("001.png", image_bytes())
        self.run_batch()
        book = self.output / "books" / "Book.cbz.2x.zip"
        previous = book.read_bytes()
        with ZipFile(source, "w") as archive:
            archive.writestr("001.png", image_bytes())
            archive.writestr("002.png", b"broken image")
        with self.assertRaises(OSError):
            self.run_batch()
        self.assertEqual(book.read_bytes(), previous)

    def test_input_output_overlap_and_empty_input(self):
        with self.assertRaisesRegex(ValueError, "No "):
            self.run_batch()
        self.ns["OUTPUT_DIR"] = str(self.input / "results")
        with self.assertRaisesRegex(ValueError, "separate"):
            self.run_batch()
        self.ns["OUTPUT_DIR"] = str(self.root)
        with self.assertRaisesRegex(ValueError, "separate"):
            self.run_batch()

    def test_transparency_is_white_and_natural_page_order(self):
        stream = io.BytesIO()
        Image.new("RGBA", (3, 5), (0, 0, 0, 0)).save(stream, format="PNG")
        image = self.ns["read_page"](stream.getvalue())
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        self.assertFalse(self.ns["is_color"](image))
        self.assertEqual(sorted(["10.png", "2.png", "1.png"], key=self.ns["natural_key"]),
                         ["1.png", "2.png", "10.png"])
        self.assertEqual(self.ns["natural_key"]("²"), ["²"])

    def test_high_bit_depth_is_rejected_instead_of_clipped(self):
        stream = io.BytesIO()
        Image.fromarray(np.full((3, 5), 32768, dtype=np.uint16)).save(stream, format="PNG")
        with self.assertRaisesRegex(ValueError, "8-bit"):
            self.ns["read_page"](stream.getvalue())

    def test_pdf_native_pages_zip_resume_and_source_changes(self):
        source = self.input / "Book.pdf"
        with pymupdf.open() as document:
            for value, color in [(80, False), (120, False), ((230, 30, 10), True)]:
                page = document.new_page(width=200, height=300)
                page.insert_image(page.rect, stream=image_bytes(value, (100, 150), color))
            document.save(source)
        original = source.read_bytes()
        self.run_batch(limit=1)
        book = self.output / "books" / "Book.pdf.2x.zip"
        self.assertFalse(book.exists())
        self.assertEqual(self.model.calls, 1)
        self.run_batch()
        self.assertEqual(self.model.calls, 3)
        with ZipFile(book) as archive:
            self.assertEqual(archive.namelist(), ["00001.png", "00002.png", "00003.png"])
            with Image.open(io.BytesIO(archive.read("00001.png"))) as image:
                self.assertEqual(image.size, (200, 300))
            with Image.open(io.BytesIO(archive.read("00003.png"))) as image:
                self.assertEqual(image.size, (200, 300))
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.getpixel((50, 75)), (230, 30, 10))
        with patch.dict(self.ns, pdf_page_bytes=Mock(side_effect=AssertionError("must skip rendering")),
                        pack_zip=Mock(side_effect=AssertionError("must skip packing"))):
            self.run_batch()
        self.assertEqual(self.model.calls, 3)
        self.assertEqual(source.read_bytes(), original)
        with pymupdf.open(source) as document:
            document[0].draw_rect(pymupdf.Rect(20, 20, 60, 60), color=(0, 0, 0), fill=(0, 0, 0))
            changed = document.tobytes()
        source.write_bytes(changed)
        self.run_batch()
        self.assertEqual(self.model.calls, 4)
        with ZipFile(book) as archive, Image.open(io.BytesIO(archive.read("00001.png"))) as image:
            self.assertEqual(image.getpixel((40, 40)), 0)

    def test_pdf_rotation_crop_and_annotations_are_preserved(self):
        pixels = np.full((150, 100), 180, dtype=np.uint8)
        pixels[:, :50] = 80
        pixels[75:, :] = 240
        image = Image.fromarray(pixels)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        with pymupdf.open() as document:
            page = document.new_page(width=200, height=300)
            page.insert_image(page.rect, stream=stream.getvalue())
            page.set_cropbox(pymupdf.Rect(40, 0, 200, 300))
            page.set_rotation(90)
            before = self.ns["pdf_page_bytes"](page)
            with Image.open(io.BytesIO(before)) as rendered:
                expected = image.crop((20, 0, 100, 150)).transpose(Image.Transpose.ROTATE_270)
                self.assertEqual(rendered.size, (150, 80))
                np.testing.assert_array_equal(np.asarray(rendered.convert("L")), np.asarray(expected))
            annotation = page.add_rect_annot(pymupdf.Rect(60, 60, 120, 120))
            annotation.set_colors(stroke=(0, 0, 0))
            annotation.set_border(width=8)
            annotation.update()
            page = document.reload_page(page)
            after = self.ns["pdf_page_bytes"](page)
            self.assertNotEqual(before, after)
            with Image.open(io.BytesIO(after)) as rendered:
                self.assertEqual(rendered.size, (150, 80))
                self.assertEqual(rendered.convert("L").getextrema()[0], 0)

    def test_pdf_vector_and_partial_image_page_uses_full_page_render(self):
        source = self.input / "Vector.pdf"
        with pymupdf.open() as document:
            page = document.new_page(width=100, height=150)
            page.insert_image(pymupdf.Rect(10, 100, 30, 130), stream=image_bytes(80, (100, 150)))
            page.insert_text((10, 20), "Text and graphics", fontsize=8)
            page.draw_rect(pymupdf.Rect(10, 30, 90, 80), color=(0, 0, 0), width=2)
            document.save(source)
        self.run_batch()
        with ZipFile(self.output / "books" / "Vector.pdf.2x.zip") as archive:
            with Image.open(io.BytesIO(archive.read("00001.png"))) as image:
                self.assertEqual(image.size, (2560, 3840))
                self.assertEqual(image.getextrema(), (0, 255))
                self.assertEqual(image.getpixel((500, 1400)), 255)

    def test_password_protected_pdf_fails_without_publishing_zip(self):
        source = self.input / "Locked.pdf"
        with pymupdf.open() as document:
            document.new_page()
            source.write_bytes(document.tobytes(
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="owner-fixture", user_pw="reader-fixture",
            ))
        with self.assertRaisesRegex(ValueError, "password"):
            self.run_batch()
        self.assertFalse((self.output / "books" / "Locked.pdf.2x.zip").exists())

    def test_image_folders_produce_zip_only_after_complete_run(self):
        (self.input / "001.png").write_bytes(image_bytes())
        chapter = self.input / "Book" / "chapter"
        chapter.mkdir(parents=True)
        (chapter / "002.png").write_bytes(image_bytes(120))
        book = self.output / "books" / "images.2x.zip"
        self.run_batch(limit=1)
        self.assertFalse(book.exists())
        self.run_batch()
        self.assertEqual(self.model.calls, 2)
        with ZipFile(book) as archive:
            self.assertEqual(archive.namelist(), ["001.png.png", "Book/chapter/002.png.png"])
        self.assertFalse(list(self.output.rglob("*.cbz")))

    def test_auto_model_boundaries_match_official_height_ranges(self):
        for actual, expected in [(1, 1200), (1250, 1200), (1251, 1300), (1350, 1300),
                                 (1351, 1400), (1450, 1400), (1451, 1500), (1550, 1500),
                                 (1551, 1600), (1760, 1600), (1761, 1920), (1984, 1920),
                                 (1985, 2048), (5000, 2048)]:
            with self.subTest(height=actual):
                self.assertEqual(self.ns["choose_model"]("Auto", actual), expected)
        self.assertEqual(self.ns["choose_model"]("1920p", 1300), 1920)

    def test_auto_selection_uses_full_image_height_and_manual_change_recomputes(self):
        self.ns.update(MODEL_SELECTION="Auto", TILE_SIZE=256)
        for index, height in enumerate((1260, 1580, 2010), 1):
            (self.input / f"{index}.png").write_bytes(image_bytes(size=(20, height)))
        self.run_batch()
        self.assertEqual(self.requests, [1300, 1600, 2048])
        first_calls = self.model.calls
        self.ns["MODEL_SELECTION"] = "1920p"
        self.run_batch()
        self.assertEqual(self.requests[-3:], [1920, 1920, 1920])
        self.assertGreater(self.model.calls, first_calls)

    def test_legacy_1920p_checkpoint_is_reused(self):
        raw = image_bytes()
        (self.input / "001.png").write_bytes(raw)
        output = self.output / "pages" / "_images" / "001.png.png"
        output.parent.mkdir(parents=True)
        original_settings = json.dumps({"version": 1, "model": self.ns["MODELS"][1920][1], "tile": 0, "pad": 32})
        signature = self.ns["hashlib"].sha256(original_settings.encode() + raw).hexdigest()
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("mangajanai_signature", signature)
        Image.new("L", (38, 54), 80).save(output, pnginfo=metadata)
        self.run_batch()
        self.assertEqual(self.model.calls, 0)
        self.assertTrue((self.output / "books" / "images.2x.zip").is_file())

    def test_changed_zip_content_is_rebuilt_from_checkpoints(self):
        (self.input / "001.png").write_bytes(image_bytes())
        self.run_batch()
        book = self.output / "books" / "images.2x.zip"
        damaged = bytearray(book.read_bytes())
        damaged[len(damaged) // 2] ^= 1
        book.write_bytes(damaged)
        pack = Mock(wraps=self.ns["pack_zip"])
        with patch.dict(self.ns, pack_zip=pack):
            self.run_batch()
        pack.assert_called_once()
        self.assertEqual(self.model.calls, 1)
        with ZipFile(book) as archive:
            self.assertIsNone(archive.testzip())

    def test_cleanup_only_after_verified_completion_preserves_other_files(self):
        source = self.input / "001.png"
        raw = image_bytes()
        source.write_bytes(raw)
        self.run_batch()
        pages = self.output / "pages" / "_images"
        note = pages / "note.txt"
        note.write_text("user note")
        untouched = pages / "unrelated.png"
        untouched.write_bytes(image_bytes(40))
        self.ns["KEEP_PAGE_CHECKPOINTS"] = False
        self.run_batch()
        self.assertFalse((pages / "001.png.png").exists())
        self.assertEqual(note.read_text(), "user note")
        self.assertTrue(untouched.is_file())
        self.assertEqual(source.read_bytes(), raw)
        self.assertEqual(self.model.calls, 1)
        self.run_batch()
        self.assertEqual(self.model.calls, 1)

    def test_bad_upload_preserves_previous_zip_and_checkpoint(self):
        source = self.input / "001.png"
        source.write_bytes(image_bytes())
        self.run_batch()
        book = self.output / "books" / "images.2x.zip"
        previous = book.read_bytes()
        source.write_bytes(image_bytes(160))
        self.ns["KEEP_PAGE_CHECKPOINTS"] = False
        real_copy = self.ns["shutil"].copyfile

        def corrupt_upload(src, dst, *args, **kwargs):
            result = real_copy(src, dst, *args, **kwargs)
            if Path(dst).name.endswith(".zip.part"):
                Path(dst).write_bytes(b"bad upload")
            return result

        with patch("shutil.copyfile", side_effect=corrupt_upload):
            with self.assertRaisesRegex(OSError, "upload verification"):
                self.run_batch()
        self.assertEqual(book.read_bytes(), previous)
        checkpoint = self.output / "pages" / "_images" / "001.png.png"
        self.assertTrue(checkpoint.is_file())
        with Image.open(checkpoint) as image:
            self.assertEqual(image.getpixel((0, 0)), 160)

    def test_model_package_is_verified_cached_and_active_model_reused(self):
        payloads = {1600: b"1600p fixture weights", 1920: b"1920p fixture weights"}
        specs = {height: (f"fixture_{height}.pth", self.ns["hashlib"].sha256(raw).hexdigest())
                 for height, raw in payloads.items()}
        stream = io.BytesIO()
        with ZipFile(stream, "w") as archive:
            for height, raw in payloads.items():
                archive.writestr(specs[height][0], raw)
        download = Mock(return_value=io.BytesIO(stream.getvalue()))
        loader = Mock()
        loader.load_from_file.side_effect = lambda path: TestDescriptor()
        with patch.dict(self.ns, MODELS=specs, ImageModelDescriptor=TestDescriptor, ModelLoader=Mock(return_value=loader)), \
                patch("urllib.request.urlopen", download), patch("torch.cuda.empty_cache"), \
                contextlib.redirect_stdout(io.StringIO()):
            pool = self.ns["ModelPool"]()
            first = pool.get(1920)
            self.assertIs(pool.get(1920), first)
            second = pool.get(1600)
            self.assertIsNot(first, second)
            self.assertEqual(pool.active, 1600)
            fresh_pool = self.ns["ModelPool"]()
            fresh_pool.get(1920)
        download.assert_called_once()
        self.assertEqual(loader.load_from_file.call_count, 3)
        for height, raw in payloads.items():
            self.assertEqual((self.output / "models" / specs[height][0]).read_bytes(), raw)

    def test_corrupt_model_is_rejected_before_loading(self):
        path = self.output / "models" / self.ns["MODELS"][1920][0]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not verified weights")
        loader = Mock()
        with patch.dict(self.ns, ModelLoader=loader):
            with self.assertRaisesRegex(ValueError, "Model checksum"):
                self.ns["ModelPool"]().get(1920)
        loader.assert_not_called()

    def test_model_path_auto_conflict_and_stale_pool_are_rejected(self):
        self.ns.update(MODEL_SELECTION="Auto", MODEL_PATH="any.pth")
        with self.assertRaisesRegex(ValueError, "empty"):
            self.ns["ModelPool"]()
        self.ns.update(MODEL_SELECTION="1920p", MODEL_PATH="")
        pool = TestPool(self.ns, self.model, self.requests)
        self.ns["MODEL_SELECTION"] = "1600p"
        with self.assertRaisesRegex(ValueError, "step 4"):
            self.ns["run_batch"](pool)

    def test_zip_packing_reads_local_pages_and_reports_stage_times(self):
        (self.input / "001.png").write_bytes(image_bytes())
        real_pack = self.ns["pack_zip"]

        def check_local(book, pages, stats, archive=None):
            self.assertTrue(all(not page.is_relative_to(self.output) for page, _ in pages))
            self.assertTrue(all(page.is_file() for page, _ in pages))
            return real_pack(book, pages, stats, archive)

        with patch.dict(self.ns, pack_zip=check_local):
            self.run_batch()
        for label in ("PDF rendering", "GPU inference", "PNG encoding", "Drive checkpoints", "ZIP upload and verification"):
            self.assertIn(label, self.last_output)

    def test_small_trial_keeps_checkpoints_even_if_cleanup_is_enabled(self):
        self.ns["KEEP_PAGE_CHECKPOINTS"] = False
        (self.input / "001.png").write_bytes(image_bytes())
        self.run_batch(limit=3)
        checkpoint = self.output / "pages" / "_images" / "001.png.png"
        self.assertTrue(checkpoint.is_file())
        self.run_batch()
        self.assertFalse(checkpoint.exists())

    def test_invalid_completion_json_is_rebuilt_without_redoing_inference(self):
        (self.input / "001.png").write_bytes(image_bytes())
        self.run_batch()
        marker = self.ns["completion_path"](self.output / "books" / "images.2x.zip")
        for raw in ("null", "[]", "{}", "invalid json"):
            with self.subTest(marker=raw):
                marker.write_text(raw)
                self.run_batch()
                self.assertEqual(self.model.calls, 1)
                self.assertIsInstance(json.loads(marker.read_text()), dict)

    def test_cleanup_refuses_symlinks_and_parent_paths(self):
        source = self.input / "001.png"
        raw = image_bytes()
        source.write_bytes(raw)
        page_root = self.output / "pages" / "_images"
        page_root.mkdir(parents=True)
        (page_root / "link.png").symlink_to(source)
        for name in ("link.png", "../../../input/001.png"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.ns["clean_checkpoints"](page_root, [name], self.output.resolve())
        self.assertEqual(source.read_bytes(), raw)

    def test_rgb_inference_preserves_channels_and_tiled_edges(self):
        pixels = np.random.default_rng(3).integers(0, 256, size=(67, 91, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        whole = self.ns["infer"](image, self.model, 0)
        tiled = self.ns["infer"](image, self.model, 16)
        self.assertEqual(tiled.mode, "RGB")
        expected = pixels.repeat(2, axis=0).repeat(2, axis=1)
        np.testing.assert_array_equal(np.asarray(whole), expected)
        np.testing.assert_array_equal(np.asarray(tiled), expected)

    def test_illustration_change_recomputes_color_only_and_keep_original_is_available(self):
        (self.input / "001.png").write_bytes(image_bytes())
        color_raw = image_bytes((220, 40, 10), color=True)
        (self.input / "002.png").write_bytes(color_raw)
        self.run_batch()
        self.assertEqual(self.requests, [1920, "V3denoise FDAT"])
        self.ns["ILLUSTRATION_SELECTION"] = "V3detail SPAN"
        self.run_batch()
        self.assertEqual(self.requests, [1920, "V3denoise FDAT", "V3detail SPAN"])
        self.assertEqual(self.model.calls, 3)
        self.ns["ILLUSTRATION_SELECTION"] = "Keep original"
        self.run_batch()
        self.assertEqual(self.model.calls, 3)
        with Image.open(self.output / "pages" / "_images" / "002.png.png") as image:
            self.assertEqual(image.size, (19, 27))
            self.assertEqual(image.getpixel((0, 0)), (220, 40, 10))

    def test_color_tile_fallback_does_not_limit_black_and_white_model(self):
        (self.input / "001.png").write_bytes(image_bytes((220, 40, 10), color=True))
        (self.input / "002.png").write_bytes(image_bytes())
        (self.input / "003.png").write_bytes(image_bytes((180, 30, 10), color=True))
        attempts = []

        def downshift(image, model, tile):
            attempts.append((image.mode, tile))
            size = (image.width * 2, image.height * 2)
            return image.resize(size), 256 if image.mode == "RGB" else tile

        with patch.dict(self.ns, upscale=downshift):
            self.run_batch()
        self.assertEqual(attempts, [("RGB", 0), ("L", 0), ("RGB", 256)])

    def test_illustration_package_is_verified_and_caches_only_two_2x_models(self):
        self.ns["ILLUSTRATION_SELECTION"] = "V3detail FDAT"
        payloads = {"V3detail FDAT": b"color FDAT fixture", "V3detail SPAN": b"color SPAN fixture"}
        specs = {key: (key.replace(" ", "_") + ".safetensors", self.ns["hashlib"].sha256(raw).hexdigest(), "detail")
                 for key, raw in payloads.items()}
        stream = io.BytesIO()
        with ZipFile(stream, "w") as archive:
            for key, raw in payloads.items():
                archive.writestr(specs[key][0], raw)
            archive.writestr("unused_4x.safetensors", b"unused weights")
        package = stream.getvalue()
        packages = {"detail": ("https://fixture.invalid/detail.zip", self.ns["hashlib"].sha256(package).hexdigest())}
        loader = Mock()
        loader.load_from_file.side_effect = lambda path: TestDescriptor()
        download = Mock(return_value=io.BytesIO(package))
        with patch.dict(self.ns, ILLUSTRATION_MODELS=specs, ILLUSTRATION_PACKAGES=packages,
                        ImageModelDescriptor=TestDescriptor, ModelLoader=Mock(return_value=loader)), \
                patch("urllib.request.urlopen", download), patch("torch.cuda.empty_cache"), \
                contextlib.redirect_stdout(io.StringIO()):
            pool = self.ns["ModelPool"]()
            pool.get("V3detail FDAT")
            pool.get("V3detail SPAN")
        download.assert_called_once()
        self.assertEqual({p.name for p in (self.output / "models").iterdir()}, {s[0] for s in specs.values()})
        (self.output / "models" / specs["V3detail FDAT"][0]).unlink()
        bad_packages = {"detail": (packages["detail"][0], "0" * 64)}
        reject_loader = Mock()
        with patch.dict(self.ns, ILLUSTRATION_MODELS=specs, ILLUSTRATION_PACKAGES=bad_packages, ModelLoader=reject_loader), \
                patch("urllib.request.urlopen", return_value=io.BytesIO(package)), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "package checksum"):
                self.ns["ModelPool"]().get("V3detail FDAT")
        reject_loader.assert_not_called()

    def test_black_model_path_does_not_override_illustration_model(self):
        black = self.input / "black.pth"
        black.write_bytes(b"black fixture")
        color_name, _, variant = self.ns["ILLUSTRATION_MODELS"]["V3denoise FDAT"]
        color = self.output / "models" / color_name
        color.parent.mkdir(parents=True)
        color.write_bytes(b"color fixture")
        self.ns["MODEL_PATH"] = str(black)
        manga = {1920: ("black.pth", self.ns["file_hash"](black))}
        illustration = {"V3denoise FDAT": (color_name, self.ns["file_hash"](color), variant)}
        loader = Mock()
        loader.load_from_file.return_value = TestDescriptor()
        with patch.dict(self.ns, MODELS=manga, ILLUSTRATION_MODELS=illustration,
                        ImageModelDescriptor=TestDescriptor, ModelLoader=Mock(return_value=loader)), \
                patch("torch.cuda.empty_cache"), contextlib.redirect_stdout(io.StringIO()):
            self.ns["ModelPool"]().get("V3denoise FDAT")
        loader.load_from_file.assert_called_once_with(color)

    def test_illustration_selection_change_requires_pool_refresh(self):
        pool = TestPool(self.ns, self.model, self.requests)
        self.ns["ILLUSTRATION_SELECTION"] = "V3detail SPAN"
        with self.assertRaisesRegex(ValueError, "step 4"):
            self.ns["run_batch"](pool)


if __name__ == "__main__":
    unittest.main()
