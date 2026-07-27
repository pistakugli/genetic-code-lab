#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import socket
import struct
from typing import Any


HEADER = struct.Struct("<I")
MAX_HEADER_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 1024 * 1024 * 1024
INITIAL_TRANSCRIPT = "0" * 64


def canonical_header(header: dict[str, Any]) -> bytes:
    return json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def advance_transcript(previous: str, header: dict[str, Any], payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(previous))
    framed_header = dict(header)
    framed_header["payloadBytes"] = len(payload)
    encoded = canonical_header(framed_header)
    digest.update(HEADER.pack(len(encoded)))
    digest.update(encoded)
    digest.update(payload)
    return digest.hexdigest()


def recv_exact(connection: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError(f"socket closed with {remaining} bytes remaining")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(connection: socket.socket, header: dict[str, Any], payload: bytes = b"") -> None:
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise RuntimeError(f"oracle payload too large: {len(payload)}")
    framed_header = dict(header)
    framed_header["payloadBytes"] = len(payload)
    encoded = canonical_header(framed_header)
    if len(encoded) > MAX_HEADER_BYTES:
        raise RuntimeError(f"oracle header too large: {len(encoded)}")
    connection.sendall(HEADER.pack(len(encoded)) + encoded + payload)


def recv_frame(connection: socket.socket) -> tuple[dict[str, Any], bytes]:
    header_size = HEADER.unpack(recv_exact(connection, HEADER.size))[0]
    if header_size <= 0 or header_size > MAX_HEADER_BYTES:
        raise RuntimeError(f"invalid oracle header size: {header_size}")
    raw_header = recv_exact(connection, header_size)
    header = json.loads(raw_header.decode("ascii"))
    if not isinstance(header, dict):
        raise RuntimeError("oracle header must be an object")
    payload_size = int(header.get("payloadBytes", -1))
    if payload_size < 0 or payload_size > MAX_PAYLOAD_BYTES:
        raise RuntimeError(f"invalid oracle payload size: {payload_size}")
    payload = recv_exact(connection, payload_size)
    return header, payload
