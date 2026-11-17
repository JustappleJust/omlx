# SPDX-License-Identifier: Apache-2.0
"""Image playground page: route, template, entry points, i18n coverage."""

import json
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).parents[1]
WEB = ROOT / "omlx_web"
I18N = WEB / "i18n"
TEMPLATE = (WEB / "templates/images.html").read_text(encoding="utf-8")
SCRIPT = (WEB / "static/js/images.js").read_text(encoding="utf-8")

_KEY_RE = re.compile(r"""\bt\(\s*'((?:images|navbar|chat|modal|login)\.[a-z0-9_.]+)'""")


def _used_keys() -> set[str]:
    return set(_KEY_RE.findall(TEMPLATE)) | set(_KEY_RE.findall(SCRIPT))


def test_every_used_key_exists_in_every_locale():
    keys = _used_keys() | {"navbar.tab.images", "chat.image_playground"}
    assert len(keys) > 60
    for path in sorted(I18N.glob("*.json")):
        locale = json.loads(path.read_text(encoding="utf-8"))
        missing = sorted(k for k in keys if k not in locale)
        assert not missing, f"{path.name} lacks {missing}"


def test_translated_not_copied_in_cjk_locales():
    en = json.loads((I18N / "en.json").read_text(encoding="utf-8"))
    for lang in ("zh-TW", "zh", "ja", "ko"):
        locale = json.loads((I18N / f"{lang}.json").read_text(encoding="utf-8"))
        same = [k for k in ("images.title", "images.mode.edit", "images.gallery") if locale[k] == en[k]]
        assert not same, f"{lang} untranslated: {same}"


def test_template_renders():
    env = Environment(loader=FileSystemLoader(str(WEB / "templates")))
    env.globals.update(
        t=lambda k: k,
        static=lambda p: f"/admin/static/{p}",
        version="test",
        custom_theme=lambda: {},
        current_lang="en",
        locale_json="{}",
    )
    html = env.get_template("images.html").render(api_key="sk-test")
    # single-quoted attribute: tojson's double quotes must not end it
    assert """x-data='imagePlayground("sk-test")'""" in html
    assert "/admin/static/js/images.js" in html
    # both API endpoints and the progress feed are wired in the script
    for needle in ("/v1/images/models", "/v1/images/generations", "/v1/images/edits", "/admin/api/activity"):
        assert needle in SCRIPT


def test_controls_follow_capabilities():
    # edit mode, mask tools, strength and step cache only show when the model has them
    assert 'x-show="canEdit"' in TEMPLATE
    assert 'x-show="caps.mask"' in TEMPLATE
    assert 'x-show="caps.strength"' in TEMPLATE
    assert "includes('step_cache')" in TEMPLATE
    assert 'x-show="caps.negative_prompt"' in TEMPLATE


def test_edit_uses_multipart_and_generation_json():
    assert "fd.append('image[]'" in SCRIPT
    assert "fd.append('mask'" in SCRIPT
    assert "'Content-Type': 'application/json'" in SCRIPT
    assert "AbortController" in SCRIPT  # cancel aborts the request -> server cancels


def test_entry_points_link_to_playground():
    navbar = (WEB / "templates/dashboard/_navbar.html").read_text(encoding="utf-8")
    chat = (WEB / "templates/chat.html").read_text(encoding="utf-8")
    assert navbar.count('href="/admin/images"') == 2  # desktop + mobile
    assert 'href="/admin/images"' in chat


def test_route_registered():
    from omlx_web.routes import router

    paths = {getattr(r, "path", None) for r in router.routes}
    assert any(p and p.endswith("/images") for p in paths)


def test_image_backdrop_never_filters_the_image():
    # A CSS filter on the checker container also applies to the <img> inside
    # and dims the shown result (dark mode once used brightness(0.35)).
    style = TEMPLATE.split("<style>", 1)[1].split("</style>", 1)[0]
    rules = re.findall(r"([^{}]*\.img-checker[^{}]*)\{([^}]*)\}", style)
    assert rules
    for selector, body in rules:
        assert "filter" not in body, f"{selector.strip()} uses a filter"
