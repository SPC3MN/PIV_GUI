"""WCAG contrast floors for the theme's text tokens.

Added after a restyle dropped every structural label in the app to 2.2:1 by
reusing INK_FAINT -- the token whose job is making DISABLED things look
disabled -- for section titles. The mistake is easy to repeat and invisible in
a screenshot review, so it is asserted numerically.

WCAG 2.1 AA: 4.5:1 for normal text, 3:1 for large text (>=18pt, or >=14pt
bold). Disabled text is exempt by the spec and is checked separately only to
confirm it stays visibly weaker than live text.
"""

import pytest

from piv_suite_gui import theme


def _relative_luminance(hex_colour):
    h = hex_colour.lstrip("#")
    channels = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(a, b):
    la, lb = _relative_luminance(a), _relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def test_contrast_helper_matches_known_values():
    """Guard the guard: black on white is exactly 21:1, white on white 1:1."""
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0, abs=0.01)
    assert contrast_ratio("#FFFFFF", "#FFFFFF") == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("ground", ["BACKDROP", "PANEL", "PAPER"])
@pytest.mark.parametrize("ink", ["INK", "INK_SOFT"])
def test_body_and_secondary_text_meet_aa(ink, ground):
    ratio = contrast_ratio(getattr(theme, ink), getattr(theme, ground))
    assert ratio >= 4.5, f"{ink} on {ground} is {ratio:.2f}:1, below the AA floor of 4.5"


def test_accent_text_meets_aa_on_every_ground():
    """The accent is used for the active tab's label and for links."""
    for ground in ("BACKDROP", "PANEL", "PAPER"):
        ratio = contrast_ratio(theme.ACCENT, getattr(theme, ground))
        assert ratio >= 4.5, f"ACCENT on {ground} is {ratio:.2f}:1"


def test_text_on_the_accent_fill_meets_aa():
    """Primary buttons draw INK_ON_ACCENT over ACCENT."""
    assert contrast_ratio(theme.INK_ON_ACCENT, theme.ACCENT) >= 4.5


def test_disabled_text_is_weaker_than_secondary_but_still_perceptible():
    """INK_FAINT is exempt from AA (disabled), but it must stay clearly weaker
    than INK_SOFT -- that difference is the only signal that a control is off."""
    faint = contrast_ratio(theme.INK_FAINT, theme.BACKDROP)
    soft = contrast_ratio(theme.INK_SOFT, theme.BACKDROP)
    assert faint < soft
    assert faint > 1.8


def test_section_titles_do_not_use_the_disabled_token():
    """The specific regression this file exists for: INK_FAINT reused for
    structural labels put every section title in the app at 2.2:1."""
    for selector in ("QGroupBox::title", "QLabel#inlineSectionLabel",
                     "QLabel#plotPlaceholder"):
        start = theme.STYLESHEET.index(selector)
        body_open = theme.STYLESHEET.index("{", start)
        body_close = theme.STYLESHEET.index("}", body_open)
        block = theme.STYLESHEET[body_open:body_close]
        assert theme.INK_FAINT not in block, (
            f"{selector} uses INK_FAINT (the disabled token) -- "
            f"that is {contrast_ratio(theme.INK_FAINT, theme.BACKDROP):.2f}:1")
