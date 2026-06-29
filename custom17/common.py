#!/usr/bin/env python3
"""Shared constants and helpers for the custom 17-class dataset."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence


CUSTOM17_CLASSES: Sequence[str] = (
    "person",
    "bottle",
    "wine glass",
    "cup",
    "bowl",
    "chair",
    "couch",
    "bed",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "book",
    "clock",
    "vase",
)

CLASS_TO_NEW_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CUSTOM17_CLASSES)}
EXPECTED_CATEGORIES: List[Dict[str, object]] = [
    {"id": idx, "name": name, "supercategory": "custom17"}
    for idx, name in enumerate(CUSTOM17_CLASSES)
]

DEFAULT_NAME_ALIASES: Mapping[str, Sequence[str]] = {
    "person": ("person",),
    "bottle": ("bottle",),
    "wine glass": ("wine glass", "wineglass"),
    "cup": ("cup",),
    "bowl": ("bowl",),
    "chair": ("chair",),
    "couch": ("couch", "sofa"),
    "bed": ("bed",),
    "tv": ("tv", "tvmonitor", "tv monitor", "television"),
    "laptop": ("laptop",),
    "mouse": ("mouse",),
    "remote": ("remote", "remote control"),
    "keyboard": ("keyboard",),
    "cell phone": ("cell phone", "cellphone", "mobile phone", "phone"),
    "book": ("book",),
    "clock": ("clock",),
    "vase": ("vase",),
}


def canonicalize_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def build_name_lookup(
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, str]:
    alias_source = aliases or DEFAULT_NAME_ALIASES
    lookup: Dict[str, str] = {}
    for canonical_name in CUSTOM17_CLASSES:
        candidates = alias_source.get(canonical_name, (canonical_name,))
        for candidate in candidates:
            lookup[canonicalize_name(candidate)] = canonical_name
    return lookup


def build_source_category_remap(
    source_categories: Iterable[Mapping[str, object]],
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> Dict[int, int]:
    lookup = build_name_lookup(aliases=aliases)
    remap: Dict[int, int] = {}
    for category in source_categories:
        source_name = str(category["name"])
        canonical_name = lookup.get(canonicalize_name(source_name))
        if canonical_name is None:
            continue
        remap[int(category["id"])] = CLASS_TO_NEW_ID[canonical_name]
    return remap


def categories_match_expected(categories: Sequence[Mapping[str, object]]) -> bool:
    if len(categories) != len(CUSTOM17_CLASSES):
        return False
    for expected, observed in zip(EXPECTED_CATEGORIES, categories):
        if int(observed["id"]) != int(expected["id"]):
            return False
        if str(observed["name"]) != str(expected["name"]):
            return False
    return True
