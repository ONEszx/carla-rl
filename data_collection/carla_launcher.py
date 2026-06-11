"""Helpers to launch a local CARLA server and wait until it is ready."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from typing import Optional

import carla

DEFAULT_CARLA_EXE = r"E:\carla\WindowsNoEditor\CarlaUE4.exe"


def wait_for_port(host: str, port: int, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            time.sleep(1.0)
    return False


def wait_for_carla_api(port: int, timeout_seconds: float, client_timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            client = carla.Client("localhost", port)
            client.set_timeout(client_timeout)
            world = client.get_world()
            _ = world.get_map().name
            return True
        except Exception:
            time.sleep(2.0)
    return False


def launch_carla_server(
    carla_exe: str = DEFAULT_CARLA_EXE,
    port: int = 2000,
    wait_seconds: float = 45.0,
    quality_level: str = "Low",
    windowed: bool = True,
    api_timeout_seconds: float = 90.0,
) -> subprocess.Popen:
    if not os.path.isfile(carla_exe):
        raise FileNotFoundError(f"CARLA executable not found: {carla_exe}")

    cmd = [carla_exe, f"-carla-port={port}"]
    if quality_level:
        cmd.append(f"-quality-level={quality_level}")
    if windowed:
        cmd.append("-windowed")

    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(cmd, cwd=os.path.dirname(carla_exe), creationflags=creationflags)

    if not wait_for_port("localhost", port, wait_seconds):
        raise TimeoutError(f"CARLA server on port {port} did not open within {wait_seconds} seconds.")

    if not wait_for_carla_api(port, api_timeout_seconds):
        raise TimeoutError(
            f"CARLA API on port {port} did not become ready within {api_timeout_seconds} seconds."
        )

    return process


def stop_carla_server(process: Optional[subprocess.Popen]) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=15)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
