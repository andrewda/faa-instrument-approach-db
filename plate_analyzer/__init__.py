from . import segmentation, text_extraction

import pymupdf
import time


class PlateAnalyzerException(Exception):
    """
    Thrown when analysis of a plate fails
    """

    pass


class PlateNeedsOCRException(PlateAnalyzerException):
    pass


def extract_information_from_plate(plate_path, debug=False):
    pdf = pymupdf.open(plate_path, filetype="pdf")
    return extract_information_from_pdf(pdf, debug=debug)


def extract_information_from_pdf(pdf, debug=False):
    return extract_information_from_pdf_with_timing(pdf, debug=debug)[0]


def extract_information_from_pdf_with_timing(pdf, debug=False):
    plate = pdf[0]
    timings = {}
    overall_start = time.perf_counter()

    start = time.perf_counter()
    drawings = plate.get_cdrawings()
    timings["drawings_extraction"] = time.perf_counter() - start

    start = time.perf_counter()
    textpage = plate.get_textpage()
    timings["textpage_creation"] = time.perf_counter() - start

    # See if we need to run OCR on the page.
    start = time.perf_counter()
    text = textpage.extractText()
    timings["ocr_detection"] = time.perf_counter() - start
    if "CATEGORY" not in text:
        raise PlateNeedsOCRException("Plate requires OCR, no CATEGORY text")

    start = time.perf_counter()
    rectangles = segmentation.segment_plate_into_rectangles(
        plate, drawings, debug=debug
    )
    timings["segmentation"] = time.perf_counter() - start

    start = time.perf_counter()
    text_info = text_extraction.extract_text_from_segmented_plate(
        plate, drawings, textpage, rectangles, debug=debug
    )
    timings["text_extraction"] = time.perf_counter() - start
    timings["total"] = time.perf_counter() - overall_start

    if debug:
        print(
            "---- ", text_info.approach_name, " - ", text_info.airport_name, "--------"
        )
        print("Has ARC:", text_info.has_dme_arc)
        print("Has procedure turn:", text_info.has_procedure_turn)
        print("Has hold-in-lieu:", text_info.has_hold_in_lieu_of_procedure_turn)
        for appch in text_info.approach_minimums:
            print(appch)

        print("Vertical profile:")

        print("Vertical descent angle:", text_info.vda)
        print("Threshold crossing height:", text_info.tch)
        print("VGSI angle:", text_info.vgsi_angle)
        print("VGSI TCH:", text_info.vgsi_tch)
        print("VGSI VDA not coincident:", text_info.vgsi_vda_not_coincident)
        print("Timing:", timings)

    return text_info, timings
