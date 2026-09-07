import io
import zipfile

from plate_analyzer import scrape_faa_dtpp_zip


def test_dtpp_pdf_processing_tasks_iterator_filters_files(tmp_path):
    zip_path = tmp_path / "DDTPPA_250123.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("A.PDF", b"a")
        zf.writestr("B.PDF", b"b")
        zf.writestr("d-TPP_Metafile.xml", b"<meta/>")

    tasks = list(
        scrape_faa_dtpp_zip.dtpp_pdf_processing_tasks_iterator(
            folder_path=tmp_path,
            files_to_process={"B.PDF"},
        )
    )
    assert tasks == [(str(zip_path), "B.PDF")]


def test_process_single_dtpp_pdf_loads_pdf_from_zip(monkeypatch, tmp_path):
    zip_path = tmp_path / "DDTPPA_250123.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("B.PDF", b"fake-pdf-bytes")

    captured = {}

    def fake_open(*args, **kwargs):
        del args
        captured["filetype"] = kwargs["filetype"]
        captured["stream_type"] = type(kwargs["stream"])
        return "fake-pdf"

    def fake_extract(pdf, debug=False):
        del debug
        assert pdf == "fake-pdf"
        return "parsed-info"

    monkeypatch.setattr(scrape_faa_dtpp_zip.pymupdf, "open", fake_open)
    monkeypatch.setattr(
        scrape_faa_dtpp_zip, "extract_information_from_pdf", fake_extract
    )

    file_name, approach_info, exception_message, zip_file_name = (
        scrape_faa_dtpp_zip.process_single_dtpp_pdf((str(zip_path), "B.PDF"))
    )

    assert file_name == "B.PDF"
    assert approach_info == "parsed-info"
    assert exception_message is None
    assert zip_file_name == zip_path.name
    assert captured["filetype"] == "pdf"
    assert captured["stream_type"] is io.BytesIO


def test_process_single_dtpp_pdf_reports_exceptions_with_zip_name(
    monkeypatch, tmp_path
):
    zip_path = tmp_path / "DDTPPA_250123.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("B.PDF", b"fake-pdf-bytes")

    monkeypatch.setattr(scrape_faa_dtpp_zip.pymupdf, "open", lambda **_: "fake-pdf")

    def raise_error(pdf, debug=False):
        del pdf
        del debug
        raise ValueError("boom")

    monkeypatch.setattr(
        scrape_faa_dtpp_zip, "extract_information_from_pdf", raise_error
    )

    _, approach_info, exception_message, zip_file_name = (
        scrape_faa_dtpp_zip.process_single_dtpp_pdf((str(zip_path), "B.PDF"))
    )

    assert approach_info is None
    assert "ValueError('boom')" in exception_message
    assert zip_file_name == zip_path.name
