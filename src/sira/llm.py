# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared LLM client helpers for enrichment scripts."""

import asyncio
import json

import aiohttp


async def post_chat(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    max_retries: int = 3,
) -> dict:
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")
                return json.loads(body)
        except Exception:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2.0**attempt)
    raise RuntimeError("unreachable")


def parse_phrases(raw: str) -> list[str]:
    end_char = {"{": "}", "[": "]"}
    for start in ["{", "["]:
        idx = raw.find(start)
        if idx == -1:
            continue
        end = raw.rfind(end_char[start])
        if end <= idx:
            continue
        try:
            parsed = json.loads(raw[idx : end + 1])
            if isinstance(parsed, dict):
                parsed = parsed.get("keywords", [])
            if isinstance(parsed, list):
                return [p for p in parsed if isinstance(p, str) and p.strip()]
        except json.JSONDecodeError:
            continue
    return []
