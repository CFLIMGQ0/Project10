#!/usr/bin/env python3
"""Loopback-only CONNECT proxy that sends GDC traffic over direct IPv4."""

from __future__ import annotations

import argparse
import select
import socket
import socketserver


ALLOWED_HOST = "api.gdc.cancer.gov"
ALLOWED_PORT = 443
MAX_HEADER_BYTES = 64 * 1024


class GDCConnectHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(15)
        header = bytearray()
        while b"\r\n\r\n" not in header:
            block = self.request.recv(4096)
            if not block:
                return
            header.extend(block)
            if len(header) > MAX_HEADER_BYTES:
                self.request.sendall(b"HTTP/1.1 431 Request Header Too Large\r\n\r\n")
                return

        request_line = bytes(header).split(b"\r\n", 1)[0]
        fields = request_line.split()
        if len(fields) != 3 or fields[0].upper() != b"CONNECT":
            self.request.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            return

        try:
            host_bytes, port_bytes = fields[1].rsplit(b":", 1)
            host = host_bytes.decode("ascii").lower()
            port = int(port_bytes)
        except (UnicodeDecodeError, ValueError):
            self.request.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            return

        if host != ALLOWED_HOST or port != ALLOWED_PORT:
            self.request.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return

        addresses = socket.getaddrinfo(
            ALLOWED_HOST,
            ALLOWED_PORT,
            family=socket.AF_INET,
            type=socket.SOCK_STREAM,
        )
        upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        upstream.settimeout(15)
        try:
            upstream.connect(addresses[0][4])
            upstream.settimeout(None)
            self.request.settimeout(None)
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            self._relay(self.request, upstream)
        finally:
            upstream.close()

    @staticmethod
    def _relay(client: socket.socket, upstream: socket.socket) -> None:
        sockets = [client, upstream]
        while True:
            readable, _, exceptional = select.select(sockets, [], sockets, 60)
            if exceptional or not readable:
                return
            for source in readable:
                target = upstream if source is client else client
                block = source.recv(1024 * 1024)
                if not block:
                    return
                target.sendall(block)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    with ThreadedTCPServer((args.host, args.port), GDCConnectHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
