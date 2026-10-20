#!/usr/bin/env python3
"""Convert an OpenCode v1 session store into funes turns files.

Reads the ``session`` / ``message`` / ``part`` tables of
``~/.local/share/opencode/opencode.db`` and writes one
``<session_id>.funes.jsonl`` per session, then index the directory with::

    funes index --yes <out-dir>

The turns format is documented at
https://github.com/huggingface/funes/blob/main/docs/funes-jsonl.md

Why not the community ``funes add opencode`` integration? It targets the
OpenCode v2 store (``session_v2`` + ``session_message``) and needs ``sqlite3``
and ``node`` on PATH; the v1 store this converter reads is not covered. This
script only uses the Python standard library.
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

DEFAULT_DB = os.path.expanduser("~/.local/share/opencode/opencode.db")
HARNESS = "opencode"
VALID_ROLES = ("user", "assistant", "tool", "system")


def rfc3339(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def truncate(text, cap):
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n...[truncated {len(text) - cap} chars]"


def blocks_from_parts(conn, message_id, tool_cap):
    blocks = []
    rows = conn.execute(
        "SELECT data FROM part WHERE message_id=? ORDER BY time_created, id",
        (message_id,),
    )
    for (part_json,) in rows:
        part = json.loads(part_json)
        kind = part.get("type")
        if kind == "text":
            text = part.get("text") or ""
            if text:
                blocks.append({"block_type": "text", "text": text})
        elif kind == "reasoning":
            text = part.get("text") or ""
            if text:
                blocks.append({"block_type": "thinking", "text": text})
        elif kind == "tool":
            name = part.get("tool")
            call_id = part.get("callID")
            state = part.get("state") or {}
            payload = state.get("input")
            try:
                call_text = (
                    json.dumps(payload, ensure_ascii=False)
                    if payload is not None
                    else ""
                )
            except TypeError:
                call_text = str(payload)
            blocks.append(
                {
                    "block_type": "tool_use",
                    "text": call_text,
                    "tool_name": name,
                    "tool_use_id": call_id,
                }
            )
            output = state.get("output")
            if output:
                blocks.append(
                    {
                        "block_type": "tool_result",
                        "text": truncate(str(output), tool_cap),
                        "tool_name": name,
                        "tool_use_id": call_id,
                    }
                )
        elif kind == "patch":
            files = part.get("files") or []
            text = "patched: " + ", ".join(str(f) for f in files) if files else "patch"
            blocks.append({"block_type": "text", "text": text})
    return blocks


def convert_session(conn, session, out_dir, tool_cap, write=True):
    sid, cwd = session["id"], session["directory"]
    if ":" in sid:
        raise ValueError(f"session id may not contain ':': {sid}")
    messages = list(
        conn.execute(
            "SELECT id, data, time_created FROM message "
            "WHERE session_id=? ORDER BY time_created, id",
            (sid,),
        )
    )
    turns = []
    seq = 0
    parent = None
    for mid, message_json, column_time in messages:
        if ":" in mid:
            raise ValueError(f"turn id may not contain ':': {mid}")
        message = json.loads(message_json)
        role = message.get("role")
        if role not in VALID_ROLES:
            role = "assistant"
        blocks = blocks_from_parts(conn, mid, tool_cap)
        if not blocks:
            continue
        created = (message.get("time") or {}).get("created") or column_time
        turn = {
            "format": 1,
            "session_id": sid,
            "turn_uuid": mid,
            "seq": seq,
            "ts": rfc3339(created),
            "role": role,
            "harness": HARNESS,
            "blocks": blocks,
        }
        if cwd:
            turn["cwd"] = cwd
        if parent:
            turn["parent_uuid"] = parent
        turns.append(turn)
        parent = mid
        seq += 1

    if not turns:
        return 0
    if not write:
        return len(turns)
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, f"{sid}.funes.jsonl")
    tmp = final + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for turn in turns:
            handle.write(json.dumps(turn, ensure_ascii=False) + "\n")
    os.replace(tmp, final)
    return len(turns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help="path to opencode.db")
    parser.add_argument("out_dir", help="directory to write .funes.jsonl files into")
    parser.add_argument(
        "--session", action="append", help="only this session id (repeatable)"
    )
    parser.add_argument(
        "--tool-cap",
        type=int,
        default=4000,
        help="max chars kept per tool_result block (0 = uncapped)",
    )
    parser.add_argument(
        "--check", action="store_true", help="validate ids and write nothing"
    )
    args = parser.parse_args()

    tool_cap = args.tool_cap if args.tool_cap > 0 else 10**12
    uri = f"file:{args.db}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    query = "SELECT id, directory FROM session ORDER BY time_created"
    sessions = list(conn.execute(query))
    if args.session:
        wanted = set(args.session)
        sessions = [s for s in sessions if s["id"] in wanted]

    total_turns = 0
    for session in sessions:
        count = convert_session(
            conn, session, args.out_dir, tool_cap, write=not args.check
        )
        total_turns += count
        print(f"{session['id']}  {count:5d} turns  {session['directory']}")
    print(f"\n{len(sessions)} sessions, {total_turns} turns -> {args.out_dir}")
    conn.close()


if __name__ == "__main__":
    sys.exit(main())
