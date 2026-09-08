import pymupdf

from . import drawing_extraction
from .segmentation import round_to_nearest

import collections
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict


@dataclass
class PlateComments:
    non_standard_takeoff_minimums: bool
    non_standard_alternative_requirements: bool
    comments: str


@dataclass
class ApproachMinimum:
    # e.g 3000 altitude 3/4 visibility
    altitude_msl: str
    altitude_agl: Optional[str]
    rvr: Optional[str]
    visibility: Optional[str]


@dataclass
class ApproachCategory:
    approach_type: str
    # Altitude, visibility for each category. If None, approach is not allowed.
    cat_a: Optional[ApproachMinimum]
    cat_b: Optional[ApproachMinimum]
    cat_c: Optional[ApproachMinimum]
    cat_d: Optional[ApproachMinimum]
    # Used if these minimums are valid based on a condition, such as being
    # able to identify a particular fix.
    condition: Optional[str]


@dataclass
class Waypoint:
    is_initial_approach_fix: bool
    is_intermediate_fix: bool
    is_final_approach_fix: bool

    def __init__(self):
        self.is_initial_approach_fix = False
        self.is_intermediate_fix = False
        self.is_final_approach_fix = False


@dataclass
class SegmentedPlate:
    approach_name: str
    airport_name: str
    approach_course: Tuple[pymupdf.Rect, str]

    has_dme_arc: bool
    has_procedure_turn: bool
    has_hold_in_lieu_of_procedure_turn: bool

    waypoints: Dict[str, Waypoint]

    required_equipment: Optional[Tuple[pymupdf.Rect, str]]
    comments: PlateComments
    missed_approach_instructions: Tuple[pymupdf.Rect, str]

    approach_minimums: List[ApproachCategory]
    vda: Optional[str]
    tch: Optional[str]
    vgsi_angle: Optional[str]
    vgsi_tch: Optional[str]
    vgsi_vda_not_coincident: bool = False


VGSI_PATTERN = re.compile(r"VGSI\s+Angle\s+(\d\.\d{2})/TCH\s+(\d+)", re.IGNORECASE)
VDA_NUM_PATTERN = re.compile(r"^(\d\.\d{2})$")
VDA_NUM_DEG_PATTERN = re.compile(r"^(\d\.\d{2})°$")


def rect_to_cache_key(rect: pymupdf.Rect) -> Tuple[float, float, float, float]:
    return (
        round(rect.x0, 1),
        round(rect.y0, 1),
        round(rect.x1, 1),
        round(rect.y1, 1),
    )


def build_rectangle_text_cache(
    rectangle_layout: List[List[pymupdf.Rect]],
    plate: pymupdf.Page,
    textpage,
    all_words=None,
) -> Dict[Tuple[float, float, float, float], str]:
    if all_words is None:
        all_words = plate.get_text("words", textpage=textpage, sort=True)

    cache = {}
    for row in rectangle_layout:
        for rect in row:
            words = filter_words_by_strict_overlap(all_words, rect)
            seen = set()
            lines = collections.OrderedDict()
            for word in words:
                key = (word[4], round(word[0], 1), round(word[1], 1))
                if key in seen:
                    continue
                seen.add(key)
                lines.setdefault((word[5], word[6]), []).append(word)
            cache[rect_to_cache_key(rect)] = "\n".join(
                " ".join(word[4].strip() for word in sorted(line, key=lambda w: w[0]))
                for line in lines.values()
            )
    return cache


def rect_contains_bbox(
    rect: pymupdf.Rect, bbox: Tuple[float, float, float, float], tolerance: float = 0.5
) -> bool:
    return (
        bbox[0] >= rect.x0 - tolerance
        and bbox[1] >= rect.y0 - tolerance
        and bbox[2] <= rect.x1 + tolerance
        and bbox[3] <= rect.y1 + tolerance
    )


def rect_overlaps_bbox(
    rect: pymupdf.Rect, bbox: Tuple[float, float, float, float], tolerance: float = 0.5
) -> bool:
    return not (
        bbox[2] < rect.x0 - tolerance
        or bbox[0] > rect.x1 + tolerance
        or bbox[3] < rect.y0 - tolerance
        or bbox[1] > rect.y1 + tolerance
    )


def filter_words_in_rect(words, rect: pymupdf.Rect):
    return [
        word
        for word in words
        if rect_overlaps_bbox(rect, (word[0], word[1], word[2], word[3]), tolerance=1.0)
    ]


def filter_words_by_strict_overlap(words, rect: pymupdf.Rect):
    """Match get_textbox's strict per-character rectangle overlap rule."""
    return [
        word
        for word in words
        if not (
            word[0] >= rect.x1
            or word[1] >= rect.y1
            or word[2] <= rect.x0
            or word[3] <= rect.y0
        )
    ]


def filter_segment_words(words, rect: pymupdf.Rect):
    # PyMuPDF keeps a word from a clipped extraction when at least half of
    # its area intersects the clip. Preserve the full extraction's order.
    filtered = []
    for word in words:
        word_rect = pymupdf.Rect(word[:4])
        intersection = word_rect & rect
        if intersection.get_area() >= word_rect.get_area() * 0.5:
            filtered.append(word)
    return filtered


def filter_chars_by_origin(chars, rect: pymupdf.Rect):
    return [
        char
        for char in chars
        if (rect.x0 - 0.5) <= char["origin"][0] <= (rect.x1 + 0.5)
        and (rect.y0 - 0.5) <= char["origin"][1] <= (rect.y1 + 0.5)
    ]


def words_to_text(words) -> str:
    return " ".join(word[4].strip() for word in words if word[4].strip())


def flatten_rawdict_chars(raw_text_dict) -> List[dict]:
    chars = []
    for block in raw_text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                chars.extend(span.get("chars", []))
    return chars


def build_point_spatial_index(points, cell_size=8):
    index = collections.defaultdict(list)
    for point in points:
        index[(int(point.x // cell_size), int(point.y // cell_size))].append(point)
    return index


def nearest_distance_from_spatial_index(
    point: pymupdf.Point,
    spatial_index,
    cell_size: int,
    max_distance: float,
) -> float:
    cell_x = int(point.x // cell_size)
    cell_y = int(point.y // cell_size)
    search_radius = int(max_distance // cell_size) + 1
    closest = float("inf")

    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            for candidate in spatial_index.get((cell_x + dx, cell_y + dy), []):
                distance = candidate.distance_to(point)
                if distance < closest:
                    closest = distance

    return closest


def extract_text_from_segmented_plate(
    plate: pymupdf.Page, drawings, textpage, rectangles: List[pymupdf.Rect], debug=False
) -> SegmentedPlate:
    # Put the rectangles into a sparse 2d array by their y and then x positions
    # on the page:
    #
    #     0  1  2  3  4
    #   ------------
    # 0 | r1 r2 r3
    # 1 | r4 r5 r6 r7 r8
    # 2 | r9 r10
    #
    # rectangle_layout[0] = [r1, r2, r3]
    # rectangle_layout[1] = [r4, r5, r6, r7, r8]
    # rectangle_layout[2] = [r9, r10]
    rectangles.sort(key=lambda rect: (rect.top_left.y, rect.top_left.x))
    rectangle_layout = []
    previous_y = -1
    for r in rectangles:
        rectangle_y = round(r.top_left.y, 1)
        if previous_y != rectangle_y:
            rectangle_layout.append([])
            previous_y = rectangle_y
        rectangle_layout[-1].append(r)

    # Reuse one full-page words extraction for every rectangular text query.
    all_words = plate.get_text("words", textpage=textpage, sort=True)
    all_chars = flatten_rawdict_chars(
        plate.get_text("rawdict", textpage=textpage, sort=True)
    )

    rectangle_text_cache = build_rectangle_text_cache(
        rectangle_layout=rectangle_layout,
        plate=plate,
        textpage=textpage,
        all_words=all_words,
    )

    def rectangle_text(rect: pymupdf.Rect, strip: bool = False):
        text = rectangle_text_cache[rect_to_cache_key(rect)]
        if strip:
            return text.strip()
        return text

    approach_course_box = rectangle_layout[0][1]
    # Some RNP approaches do not have a channel/ILS box on the top-left
    if len(rectangle_layout[0]) == 2:
        approach_course_box = rectangle_layout[0][0]
    # One legacy curved-text course box has source-word line order unlike
    # get_textbox; preserve that exact result without affecting the cache.
    approach_text = plate.get_textbox(approach_course_box, textpage=textpage)
    approach_text = approach_text.replace("APP CRS", "").strip()

    # Approach title will be on the right side of the page, after the approach
    # course and info boxes on the left.
    approach_title_area = pymupdf.Rect(
        rectangle_layout[0][-1].top_right + pymupdf.Point(30, 0),
        pymupdf.Point(plate.rect.width, rectangle_layout[0][-1].bottom_right.y),
    )
    approach_title = filter_segment_words(all_words, approach_title_area)
    approach_title = pymupdf_group_words_into_lines_based_on_vertical_position(
        approach_title
    )
    # Ignore lines with the FAA-approach identifier
    approach_title = [
        line
        for line in approach_title
        if ("(FAA)" not in line) and (not line.isdigit())
    ]
    # If there is an ILS category like `(CAT II)`, append it to the approach
    # name.
    if len(approach_title) == 3 and (
        approach_title[0].startswith("(CAT") or approach_title[0].startswith("(SA")
    ):
        ils_category = approach_title[0]
        approach_title = [f"{approach_title[1]} {ils_category}", approach_title[2]]
    # Another hack to deal with extra stuff being included in the approach title.
    # Get rid of anything that doesn't have some alphabets.
    approach_title = [line for line in approach_title if any(c.isalpha() for c in line)]
    # Finally if we still have too many elements, just do the best we can.
    if len(approach_title) > 2:
        approach_title = approach_title[:2]

    # First line is the approach title, then the airport name.
    approach_name, airport_name = approach_title

    # Get all the waypoints in the plan view.
    plan_view_box = find_plan_view_box(rectangle_layout, plate)
    plan_view_words = filter_segment_words(all_words, plan_view_box)
    plan_view_rawdict = filter_chars_by_origin(all_chars, plan_view_box)
    waypoints = extract_all_waypoints_from_plan_view(
        plan_view_box,
        plate,
        plan_view_words=plan_view_words,
    )
    (has_hold_in_lieu, has_procedure_turn) = (
        drawing_extraction.extract_approach_metadata(
            plan_view_box, plate, drawings, debug=debug
        )
    )
    has_dme_arc = has_dme_arc_in_plan_view(
        plan_view_box,
        plate,
        plan_view_rawdict=plan_view_rawdict,
    )

    # Single top-section classification pass for missed approach, comment box,
    # and required equipment candidates.
    missed_approach_rect = None
    comment_candidates = []
    required_equipment_candidates = []
    max_scan_row = min(len(rectangle_layout), 4)
    for i in range(0, max_scan_row):
        for rect in rectangle_layout[i]:
            rect_text = rectangle_text(rect)
            if i <= 2 and "MISSED" in rect_text:
                # HACK: RNAV 22 for FLP has a typo. (Reported to the FAA)
                if "APPROACH" in rect_text or "APROACH" in rect_text:
                    missed_approach_rect = rect

            if i in (1, 2, 3) and rect.width > (plate.rect.width * 0.3):
                comment_candidates.append(rect)
            if i in (0, 1, 2):
                required_equipment_candidates.append(rect)

    if missed_approach_rect is None:
        raise ValueError("Could not find missed approach instructions")

    # Comments box will be more than a third the width of the document, and
    # its bottom will line up with the missed approach box.
    comments_box = None
    for rect in comment_candidates:
        if rect == missed_approach_rect:
            continue
        if abs(rect.bottom_left.y - missed_approach_rect.bottom_left.y) < 3:
            if comments_box is None or rect.width > comments_box.width:
                comments_box = rect
    if comments_box is None:
        raise ValueError("Could not find comments box")

    # If there is a required equipment box, it will be a narrow one above the
    # comments.
    required_equipment = None
    for rect in required_equipment_candidates:
        if (
            rect != comments_box
            and int(rect.width - comments_box.width) == 0
            and rect.top_left.y < comments_box.top_left.y
        ):
            required_equipment = rect
            break

    # Missed approach has very strange ordering in the pdf, we often end up with
    # things like:
    #   'direct CELSY and hold.  \nMISSED APPROACH: Climb to 4200'.
    # Therefore, re-extract the text with sorting.
    # Keep the original clipped extraction for this irregularly ordered box;
    # unlike the other word regions, its fixture output depends on a fresh
    # clipped text page's ordering.
    missed_approach_text = plate.get_text(
        option="words", sort=True, clip=missed_approach_rect
    )
    missed_approach_text = pymupdf_extracted_words_to_string(missed_approach_text)

    # The left side of the comments box will have a "T" for non-standard takeoff
    # minimums and "A" for non-standard alternative requirements.
    non_standard_takeoff_minimums = False
    non_standard_alternative_requirements = False

    left_side_comments = pymupdf.Rect(
        comments_box.top_left, comments_box.bottom_left + pymupdf.Point(10, 0)
    )
    left_side_text = words_to_text(filter_segment_words(all_words, left_side_comments))
    if "A" in left_side_text:
        non_standard_takeoff_minimums = True
    if "T" in left_side_text:
        non_standard_alternative_requirements = True

    right_side_comments = pymupdf.Rect(
        left_side_comments.top_right, comments_box.bottom_right
    )
    # Same legacy ordering exception as the missed-approach box above.
    comments_text = plate.get_text(option="words", sort=True, clip=right_side_comments)
    comments_text = pymupdf_extracted_words_to_string(comments_text)
    # Remove solitary As and Ts from the start and end of comments. More than
    # likely just accidentally included the alternatives symbols.
    comments_text = re.sub(r"^\b(T|A)\b", "", comments_text, count=2).strip()
    comments_text = re.sub(r"\b(T|A)\b$", "", comments_text, count=2).strip()

    comments = PlateComments(
        non_standard_takeoff_minimums,
        non_standard_alternative_requirements,
        comments=comments_text,
    )

    if required_equipment:
        required_equipment_text = filter_segment_words(all_words, required_equipment)
        required_equipment = (
            required_equipment,
            pymupdf_extracted_words_to_string(required_equipment_text),
        )

    # Find CATEGORY rectangle once and reuse.
    category_rect = None
    for i in range(len(rectangle_layout) - 1, 0, -1):
        for rect in rectangle_layout[i]:
            rect_text = rectangle_text(rect, strip=True)
            if "CATEGORY" not in rect_text:
                continue
            category_rect = rect
            break
        if category_rect is not None:
            break

    try:
        minimums = extract_minimums(
            rectangle_layout,
            plate=plate,
            textpage=textpage,
            rectangle_text_getter=rectangle_text,
            category_rect=category_rect,
            preextracted_words=all_words,
            preextracted_chars=all_chars,
        )
    except ValueError:
        minimums = []

    # Try to find the profile view box - below plan view and above minimums.
    profile_view_box = None
    category_rect_top = plate.rect.height  # Default to bottom if minimums not foun

    if category_rect:
        category_rect_top = category_rect.y0

    if plan_view_box:
        profile_view_box = pymupdf.Rect(
            plan_view_box.x0,
            plan_view_box.y1 + 1,  # Start just below plan view
            plan_view_box.x1,
            category_rect_top - 1,  # End just above minimums category header
        )
        # Ensure the box has positive height
        if profile_view_box.y1 <= profile_view_box.y0:
            profile_view_box = None

    # Extract VDA and TCH from the profile view
    vda, tch, vgsi_angle, vgsi_tch, vgsi_vda_not_coincident = (
        extract_vertical_profile_info(
            plate,
            profile_view_box,
            words=(
                filter_segment_words(all_words, profile_view_box)
                if profile_view_box
                else None
            ),
        )
    )

    return SegmentedPlate(
        approach_name=approach_name,
        airport_name=airport_name,
        approach_course=(approach_course_box, approach_text),
        has_dme_arc=has_dme_arc,
        has_hold_in_lieu_of_procedure_turn=has_hold_in_lieu,
        has_procedure_turn=has_procedure_turn,
        waypoints=dict(waypoints),
        required_equipment=required_equipment,
        missed_approach_instructions=(missed_approach_rect, missed_approach_text),
        comments=comments,
        approach_minimums=minimums,
        vda=vda,
        tch=tch,
        vgsi_angle=vgsi_angle,
        vgsi_tch=vgsi_tch,
        vgsi_vda_not_coincident=vgsi_vda_not_coincident,
    )


CATEGORIES = "ABCD"


def collapse_duplicate_lines(text: str) -> str:
    """Some plates draw glyphs twice at identical positions, which makes
    get_textbox return doubled text such as 'A\\nA' for the category letter
    boxes. Collapse exactly two identical lines into one."""
    lines = text.split("\n")
    if len(lines) == 2 and lines[0] == lines[1]:
        return lines[0]
    return text


def extract_minimums(
    rectangle_layout,
    plate: pymupdf.Page,
    textpage,
    rectangle_text_getter=None,
    category_rect: Optional[pymupdf.Rect] = None,
    preextracted_words=None,
    preextracted_chars=None,
) -> List[ApproachCategory]:
    if rectangle_text_getter is None:
        rectangle_text_getter = lambda rect, strip=False: (
            plate.get_textbox(rect, textpage=textpage).strip()
            if strip
            else plate.get_textbox(rect, textpage=textpage)
        )

    # Locate the rectangle that says "CATEGORY"
    if category_rect is None:
        for i in range(len(rectangle_layout) - 1, 0, -1):
            for rect in rectangle_layout[i]:
                rect_text = rectangle_text_getter(rect, strip=True)
                if "CATEGORY" in rect_text:
                    category_rect = rect

    if category_rect is None:
        raise ValueError("Unable to find CATEGORY box")

    # Filter out any rectangles that are above or the left of the minimums.
    filtered_rectangles = []
    for row in rectangle_layout:
        filtered_row = []
        for rect in row:
            if (rect.top_left.x + 0.5) > category_rect.top_left.x and (
                rect.top_left.y + 0.5
            ) > category_rect.top_left.y:
                filtered_row.append(rect)
        if len(filtered_row) != 0:
            filtered_rectangles.append(filtered_row)

    rectangle_layout = filtered_rectangles

    if len(rectangle_layout[0]) < 4:
        raise ValueError("Not enough letter boxes after CATEGORY")

    # Verify that the boxes next to category are A, B, C, D like we expect.
    category_boxes = []
    for i, letter in enumerate(CATEGORIES):
        letter_rect = rectangle_layout[0][i + 1]
        letter_text = collapse_duplicate_lines(
            rectangle_text_getter(letter_rect)
        ).strip()
        if letter_text != letter:
            raise ValueError(
                f"letter {i} after CATEGORY should be {letter}, was {letter_text}"
            )
        category_boxes.append(letter_rect)
    categories_width = sum([cat_box.width for cat_box in category_boxes])

    # Filter out rectangles that are to the right of the last category box
    # and remove empty rows
    rightmost_x = category_boxes[-1].top_right.x
    rectangle_layout = [
        [rect for rect in row if rect.top_left.x <= rightmost_x + 0.5]
        for row in rectangle_layout
    ]
    rectangle_layout = [row for row in rectangle_layout if len(row) > 0]

    minimums_region = pymupdf.Rect(category_rect)
    for row in rectangle_layout:
        for rect in row:
            minimums_region.x0 = min(minimums_region.x0, rect.x0)
            minimums_region.y0 = min(minimums_region.y0, rect.y0)
            minimums_region.x1 = max(minimums_region.x1, rect.x1)
            minimums_region.y1 = max(minimums_region.y1, rect.y1)
    if preextracted_words is None:
        minimums_words = plate.get_text("words", clip=minimums_region, sort=True)
    else:
        minimums_words = filter_segment_words(preextracted_words, minimums_region)
        # A full-page sorted word list can place adjacent minimums lines in
        # source order rather than the clipped extractor's visual order.
        minimums_words.sort(key=lambda word: (word[1], word[0]))
    if preextracted_chars is None:
        minimums_rawdict = plate.get_text("rawdict", clip=minimums_region, sort=True)
        minimums_chars = flatten_rawdict_chars(minimums_rawdict)
    else:
        minimums_chars = filter_chars_by_origin(preextracted_chars, minimums_region)

    # Grab the first approach name.
    all_minimums = []
    # First set of minimums are the default, no conditions.
    condition = None

    for i in range(1, len(rectangle_layout)):
        approach_name_rect = rectangle_layout[i][0]
        # Should be the same size as the category cell and have some text.
        if int(approach_name_rect.width) != int(category_rect.width):
            break
        approach_name = collapse_duplicate_lines(
            rectangle_text_getter(approach_name_rect)
        )
        if len(approach_name.strip()) == 0:
            break
        # Remove the Decision Altitude/Minimum Descent Altitude suffix, and fix
        # LNAV/VNAV being split over two lines.
        approach_name = approach_name.replace("MDA", "").replace("DA", "").strip()
        if "LNAV" in approach_name and "VNAV" in approach_name:
            approach_name = "LNAV/VNAV"

        # If this is Circling with an additional C in the box, denote that it's
        # circling with extended protected area.
        if "CIRCLING" in approach_name and (
            "C" in approach_name.replace("CIRCLING", "")
        ):
            approach_name = "CIRCLING (Expanded Radius)"

        minimums_per_category = []
        # Now iterate through the minimums values, up to 4 boxes.
        num_minimums = 0
        j = 0
        while num_minimums < 4:
            minimums_box = rectangle_layout[i][j + 1]
            minimums = extract_minimums_from_text_box(
                minimums_box,
                approach_name,
                plate,
                preextracted_words=minimums_words,
                preextracted_chars=minimums_chars,
            )
            # Check the width of the minimums box to see how many categories it
            # covers.
            num_categories_covered = int(
                round(minimums_box.width / (categories_width / 4), 0)
            )
            for _ in range(num_categories_covered):
                minimums_per_category.append(minimums)
                num_minimums += 1
            j += 1

        # Some plates have special category E for very fast military planes.
        # Let's ignore those :)
        minimums_per_category = minimums_per_category[:4]

        cat_a, cat_b, cat_c, cat_d = minimums_per_category
        all_minimums.append(
            ApproachCategory(
                approach_name, cat_a, cat_b, cat_c, cat_d, condition=condition
            )
        )

    return all_minimums


MINIMUMS_TEXT_NEXT_LINE_THRESHOLD = 4
# At what percentage of a character's height is a number considered a small
# fraction character.
FRACTION_HEIGHT_PERCENTAGE = 0.8


def get_minimums_text_letters(box, plate, preextracted_chars=None):
    # Gets the letters from a minimums box.
    if preextracted_chars is None:
        raw_text = plate.get_text(option="rawdict", clip=box)
        letters = flatten_rawdict_chars(raw_text)
    else:
        letters = [
            char
            for char in preextracted_chars
            if (box.x0 - 0.5) <= char["origin"][0] <= (box.x1 + 0.5)
            and (box.y0 - 0.5) <= char["origin"][1] <= (box.y1 + 0.5)
        ]

    # Remove any characters that are very far apart vertically from the first line.
    if not letters:
        return []
    min_y = min(letter["origin"][1] for letter in letters)
    filtered_letters = []
    for letter in letters:
        if abs(letter["origin"][1] - min_y) < MINIMUMS_TEXT_NEXT_LINE_THRESHOLD:
            filtered_letters.append(letter)
    letters = filtered_letters

    # Remove spaces.
    letters = [l for l in letters if l["c"] != " "]
    # Sort by x-cordinate.
    letters.sort(key=lambda c: c["origin"][0])

    # HACK: occasionally, we will have dashes where the fraction that comes
    # after actually has a x-coordinate that is before the dash. For example:
    #   "1446-½"
    # '1': origin (145.69, 512.84)
    # '-': origin (145.92, 515.07)
    #
    # So if we detect a "small" letter right before a dash, swap them.
    for i, letter in enumerate(letters):
        if i <= 0 or letter["c"] != "-":
            continue
        # Okay we have a dash, check the letter before it.
        letter_before = letters[i - 1]
        # Check if they're close together.
        if letter["origin"][0] - letter_before["origin"][0] > 0.8:
            continue

        # See if it's a fraction compared to the dash.
        letter_before_height = letter_before["bbox"][3] - letter_before["bbox"][1]
        dash_height = letter["bbox"][3] - letter["bbox"][1]
        if letter_before_height < dash_height * FRACTION_HEIGHT_PERCENTAGE:
            # Swap the letters, this was likely just the dash being too close
            # to the fraction.
            letters[i] = letter_before
            letters[i - 1] = letter

    return letters


def extract_minimums_from_text_box(
    box,
    minimum_type,
    plate,
    preextracted_words=None,
    preextracted_chars=None,
) -> ApproachMinimum:
    # Check if the procedure is allowed for this category.
    if preextracted_words is None:
        text = plate.get_text(option="text", clip=box).strip()
    else:
        text = words_to_text(filter_words_in_rect(preextracted_words, box)).strip()
    # Empty category cells can occur when minimums are only defined for some
    # approach categories.
    if len(text) == 0:
        return None
    if "NA" in text:
        return None
    # If the text "CAT" appears in the box, this is a special ILS cat approach,
    # we don't handle that format of minimums yet.
    if "CAT" in text:
        return "Unknown"

    letters = get_minimums_text_letters(
        box,
        plate,
        preextracted_chars=preextracted_chars,
    )
    # Gets set to visibility or rvr depending on what we're expecting next.
    next_number = None
    altitude_msl = ""
    # Scan for the altitude first.
    for i, letter in enumerate(letters):
        # Dash separates altitude from visibility
        if letter["c"] == "-":
            next_number = "visibility"
            break
        # Slash separates rvr from visibility
        if letter["c"] == "/":
            next_number = "rvr"
            break
        altitude_msl += letter["c"]

    # Weird, no altitude or rvr seperator. something must have gone wrong.
    if next_number is None:
        print(minimum_type, letters)
        raise ValueError("No slash or dash in minimums box")

    rvr = None
    visibility = None

    if next_number == "visibility":
        # A visibility will either be a single digit like '1', a fraction like ½
        # or a mixed fraction like 1 ½.
        first_number = letters[i + 1]
        visibility = first_number["c"]
        # Check if the first number is a fraction numerator by checking its
        # size against the altitude number.
        first_number_bbox = pymupdf.Rect(first_number["bbox"])
        first_letter_bbox = pymupdf.Rect(letters[0]["bbox"])
        if (
            first_number_bbox.height
            < first_letter_bbox.height * FRACTION_HEIGHT_PERCENTAGE
        ):
            visibility = f"{visibility}/{letters[i + 2]['c']}"
        elif len(letters) > (i + 3):
            # First number was not a fraction, so this could be a single number
            # or a mixed fraction. Check if the next number is a fraction.
            second_number_bbox = pymupdf.Rect(letters[i + 2]["bbox"])
            if (
                second_number_bbox.height
                < first_letter_bbox.height * FRACTION_HEIGHT_PERCENTAGE
            ):
                # Okay, next should be a fraction since it's close to the first
                # number.
                visibility = f"{visibility} {letters[i + 2]['c']}/{letters[i + 3]['c']}"
    elif next_number == "rvr":
        # RVR could be up to two numbers
        rvr = f"{letters[i + 1]["c"]}{letters[i + 2]["c"]}"
    else:
        raise NotImplemented()

    altitude_agl = None
    agl_match = re.search(r"(\d{2,5})\s*\(", text)
    if agl_match is not None:
        altitude_agl = agl_match.group(1)

    return ApproachMinimum(
        altitude_msl=altitude_msl,
        altitude_agl=altitude_agl,
        rvr=rvr,
        visibility=visibility,
    )


def pymupdf_extracted_words_to_string(words):
    """Joins a list of extracted words from pymudpf which are a list of tuples
    of the form `(x0, y0, x1, y1, "word", block_no, line_no, word_no)`
    into a string of the words.
    """
    return " ".join([w[4].strip() for w in words])


def pymupdf_group_words_into_source_lines(words):
    """Join words using PyMuPDF's source block/line identity."""
    lines = collections.OrderedDict()
    for word in words:
        lines.setdefault((word[5], word[6]), []).append(word[4].strip())
    return [" ".join(line) for line in lines.values()]


def pymupdf_group_words_into_lines_based_on_vertical_position(words):
    """Joins a list of extracted words into lines as above but returns a list
    of lines, grouping them based on their y-coordinate."""
    # Sort by x only
    words.sort(key=lambda w: w[0])

    words_grouped_by_y = collections.defaultdict(list)
    for w in words:
        y = (w[1] + w[3]) / 2
        y_round = round_to_nearest(y, nearest=6)
        words_grouped_by_y[y_round].append(w[4].strip())

    lines = []
    for y in sorted(words_grouped_by_y.keys()):
        lines.append(" ".join(words_grouped_by_y[y]))
    return lines


# Distance threshold between the (IAF) text to WAYPOINT name text to consider
# it an IAF. Also used for FAF, IF.
FIX_TEXT_DISTANCE_THRESHOLD = 25


def is_waypoint_text_close_to_approach_type(waypoint_loc, approach_fixes):
    is_close = False
    for initial_appraoch_fix in approach_fixes:
        iaf_location = pymupdf.Point(initial_appraoch_fix[2], initial_appraoch_fix[3])
        distance = iaf_location.distance_to(waypoint_loc)
        if distance < FIX_TEXT_DISTANCE_THRESHOLD:
            is_close = True
    return is_close


def extract_all_waypoints_from_plan_view(plan_view_box, plate, plan_view_words=None):
    words = plan_view_words
    if words is None:
        words = plate.get_text(option="words", sort=True, clip=plan_view_box)

    initial_approach_fix_texts = []
    intermediate_fix_texts = []
    final_approach_fix_texts = []
    for w in words:
        word = w[4].strip()
        if (not word.startswith("(")) or (not word.endswith(")")):
            continue
        if "IAF" in word:
            initial_approach_fix_texts.append(w)
        if "IF" in word:
            intermediate_fix_texts.append(w)
        if "FAF" in word:
            final_approach_fix_texts.append(w)

    waypoints = collections.defaultdict(Waypoint)
    for w in words:
        word = w[4].strip()
        # Waypoints are generally 5 uppercase letters.
        if len(word) != 5 or (not word.isalpha()) or word.upper() != word:
            continue
        # See if this is an initial approach fix by looking for the text IAF
        # nearby.
        word_location = pymupdf.Point(w[0], w[1])

        is_initial_approach_fix = is_waypoint_text_close_to_approach_type(
            word_location, initial_approach_fix_texts
        )
        is_intermediate_fix = is_waypoint_text_close_to_approach_type(
            word_location, intermediate_fix_texts
        )
        is_final_approach_fix = is_waypoint_text_close_to_approach_type(
            word_location, final_approach_fix_texts
        )
        # Set if the fix is IAF/IF/FAF based on what we saw here, updating any
        # previous bools.
        waypoints[word].is_initial_approach_fix |= is_initial_approach_fix
        waypoints[word].is_intermediate_fix |= is_intermediate_fix
        waypoints[word].is_final_approach_fix |= is_final_approach_fix

    return waypoints


def has_dme_arc_in_plan_view(plan_view_box, plate, plan_view_rawdict=None):
    """Look for the words 'Arc' in the plan view, this is slightly complicated
    by the fact that the words can be curved. This means we can't just use
    pymupdf's word extaction directly to find it.
    """
    words = plan_view_rawdict
    if words is None:
        words = flatten_rawdict_chars(
            plate.get_text(option="rawdict", sort=True, clip=plan_view_box)
        )
    elif isinstance(words, dict):
        # Accept the legacy rawdict shape for direct callers as well.
        words = flatten_rawdict_chars(words)

    letter_locations = collections.defaultdict(list)

    for char in words:
        # Note the locations of all 'A', 'r' and 'c' characters.
        if char["c"] in ("A", "r", "c"):
            letter_locations[char["c"]].append(pymupdf.Point(char["origin"]))

    if len(letter_locations["r"]) == 0 or len(letter_locations["c"]) == 0:
        return False

    r_index = build_point_spatial_index(letter_locations["r"], cell_size=8)
    c_index = build_point_spatial_index(letter_locations["c"], cell_size=8)

    # Iterate through all the 'A' characters.
    for a_location in letter_locations["A"]:
        # Check distances to the closest 'r' character.
        closest_r = nearest_distance_from_spatial_index(
            point=a_location,
            spatial_index=r_index,
            cell_size=8,
            max_distance=6,
        )
        if closest_r > 6:
            continue
        closest_c = nearest_distance_from_spatial_index(
            point=a_location,
            spatial_index=c_index,
            cell_size=8,
            max_distance=8,
        )
        if closest_c > 8:
            continue
        return True

    return False


def extract_vertical_profile_info(
    plate: pymupdf.Page, profile_view_box: pymupdf.Rect, words=None
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[bool]]:
    """Extracts VDA, TCH, VGSI Angle, VGSI TCH strings and not coincident flag,
    using line info for VGSI and positional info for VDA/TCH.
    """
    vda = None
    tch = None
    vgsi_angle = None
    vgsi_tch = None
    vgsi_vda_not_coincident = False
    vda_bbox = None  # Store VDA bounding box

    if profile_view_box is None:
        return None, None, None, None, None

    if words is None:
        # Keep the direct-call behavior, including its clipped extraction path.
        profile_text_raw = plate.get_text(
            option="text", clip=profile_view_box, sort=True
        )
    else:
        profile_text_raw = pymupdf_extracted_words_to_string(words)
    if "not coincident".lower() in profile_text_raw.lower():
        vgsi_vda_not_coincident = True

    # --- VGSI Extraction (Line-based) ---
    if words is None:
        # Use get_text("dict") to get line info, though blocks might be safer if lines split
        profile_dict = plate.get_text("dict", clip=profile_view_box, sort=True)
        profile_lines = [
            "".join([span["text"] for span in line.get("spans", [])])
            for block in profile_dict.get("blocks", [])
            for line in block.get("lines", [])
        ]
    else:
        profile_lines = pymupdf_group_words_into_source_lines(words)
    for line_text in profile_lines:
        if "VGSI Angle".lower() in line_text.lower():
            vgsi_match = VGSI_PATTERN.search(line_text)
            if vgsi_match:
                vgsi_angle = vgsi_match.group(1).strip()
                vgsi_tch = vgsi_match.group(2).strip()
                break  # Assume only one such line

    # --- VDA Extraction (Positional Analysis) ---
    if words is None:
        words = plate.get_text("words", clip=profile_view_box, sort=True)
    horizontal_closeness_threshold = 5

    for i, word_info in enumerate(words):
        word_text = word_info[4]
        potential_vda_match = None
        num_word_info = None
        num_word_idx = -1

        # Case 1: Word is the number (e.g., "3.00") and the next word is "°"
        if (
            VDA_NUM_PATTERN.match(word_text)
            and i + 1 < len(words)
            and words[i + 1][4] == "°"
        ):
            potential_vda_match = VDA_NUM_PATTERN.match(word_text)
            num_word_info = word_info
            num_word_idx = i

        # Case 2: Word contains number and degree symbol (e.g., "3.00°")
        elif VDA_NUM_DEG_PATTERN.match(word_text):
            potential_vda_match = VDA_NUM_DEG_PATTERN.match(word_text)
            num_word_info = word_info
            num_word_idx = i

        # If a potential VDA number was found in either case:
        if potential_vda_match and num_word_info:
            # Check if this number is immediately preceded by another numeric word
            # (e.g., to avoid matching a course number like 042°)
            precedes_numeric = False
            if num_word_idx > 0:
                prev_word_info = words[num_word_idx - 1]
                prev_word_text = prev_word_info[4]
                prev_word_x1 = prev_word_info[2]
                num_word_x0 = num_word_info[0]
                # Check if previous word is numeric and close horizontally
                if (
                    prev_word_text.isdigit()
                    and (num_word_x0 - prev_word_x1) < horizontal_closeness_threshold
                ):
                    precedes_numeric = True

            # If it's not preceded by a number, we likely found the VDA
            if not precedes_numeric:
                vda = potential_vda_match.group(1).strip()
                # Store the bounding box of the number word itself
                vda_bbox = pymupdf.Rect(num_word_info[:4])
                break  # Found VDA, exit loop

    # --- Primary TCH Extraction (Positional based on VDA) ---
    if vda_bbox is not None:
        for word_info in words:
            if word_info[4].upper() != "TCH":
                continue

            tch_label_bbox = pymupdf.Rect(word_info[:4])
            if tch_label_bbox.y0 <= vda_bbox.y1:
                continue

            same_line_numeric_words = []
            for candidate_word_info in words:
                candidate_text = candidate_word_info[4].strip()
                if not candidate_text.isdigit():
                    continue

                candidate_bbox = pymupdf.Rect(candidate_word_info[:4])
                if candidate_bbox.x0 < tch_label_bbox.x1:
                    continue

                vertically_aligned = max(tch_label_bbox.y0, candidate_bbox.y0) <= min(
                    tch_label_bbox.y1, candidate_bbox.y1
                )
                if not vertically_aligned:
                    continue

                same_line_numeric_words.append((candidate_bbox.x0, candidate_text))

            if same_line_numeric_words:
                same_line_numeric_words.sort(key=lambda candidate: candidate[0])
                tch = same_line_numeric_words[0][1]
                break

    return vda, tch, vgsi_angle, vgsi_tch, vgsi_vda_not_coincident


def find_plan_view_box(rectangle_layout, plate) -> pymupdf.Rect:
    """Find the plan view part of the plate"""
    # Largest rectangle is probably the plan view.
    largest_rect = rectangle_layout[0][0]

    for row in rectangle_layout:
        for rect in row:
            if rect.get_area() > largest_rect.get_area():
                largest_rect = rect

    # Just assert that the rectangle is around the middle of the plate, that's
    # where we expect it to be.
    assert largest_rect.top_left.y < (plate.rect.height / 2)
    assert largest_rect.bottom_right.y > (plate.rect.height / 2)
    assert largest_rect.top_left.x < (plate.rect.width / 2)
    assert largest_rect.bottom_right.x > (plate.rect.width / 2)

    return largest_rect
