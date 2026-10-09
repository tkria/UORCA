"""Paramiko-based SSH manager supporting two-hop (ProxyJump) connections."""

from __future__ import annotations

import subprocess
from typing import Optional

try:
    import paramiko
except ImportError as e:
    raise ImportError(
        "paramiko is required for HPC connectivity. "
        "Install it with: uv pip install 'uorca[hpc]'"
    ) from e


def _get_ssh_config(host_alias: str) -> dict[str, str]:
    """Run 'ssh -G <host_alias>' and return parsed key-value pairs."""
    result = subprocess.run(
        ["ssh", "-G", host_alias],
        capture_output=True,
        text=True,
        check=True,
    )
    config: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            config[parts[0].lower()] = parts[1]
    return config


def connect(host_alias: str, password: Optional[str] = None) -> paramiko.SSHClient:
    """Connect to *host_alias* via Paramiko, handling ProxyJump if configured.

    If the SSH config for *host_alias* contains a ``proxyjump`` directive the
    connection is made in two hops: first to the jump host, then through an
    open channel to the target host.

    The returned ``SSHClient`` has an extra ``_jump_client`` attribute that
    holds the jump client (or ``None`` when no jump was used) so that
    :func:`close` can clean it up.
    """
    config = _get_ssh_config(host_alias)

    hostname = config.get("hostname", host_alias)
    port = int(config.get("port", 22))
    username = config.get("user")
    identity_files = config.get("identityfile", "").split()

    def _make_pkey(identity_files: list[str]) -> Optional[paramiko.PKey]:
        for path in identity_files:
            path = path.replace("~", __import__("os").path.expanduser("~"))
            for cls in (
                paramiko.RSAKey,
                paramiko.Ed25519Key,
                paramiko.ECDSAKey,
                paramiko.DSSKey,
            ):
                try:
                    return cls.from_private_key_file(path)
                except (paramiko.SSHException, ValueError, OSError):
                    continue
        return None

    jump_raw = config.get("proxyjump", "").strip()
    jump_client: Optional[paramiko.SSHClient] = None

    if jump_raw and jump_raw not in ("none", ""):
        # --- First hop: connect to jump host ---
        jump_config = _get_ssh_config(jump_raw)
        jump_hostname = jump_config.get("hostname", jump_raw)
        jump_port = int(jump_config.get("port", 22))
        jump_user = jump_config.get("user")
        jump_identity = jump_config.get("identityfile", "").split()

        jump_client = paramiko.SSHClient()
        jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        jump_client.connect(
            hostname=jump_hostname,
            port=jump_port,
            username=jump_user,
            password=password,
            pkey=_make_pkey(jump_identity),
            look_for_keys=True,
            allow_agent=True,
        )

        # Open a direct-tcpip channel from the jump host to the target
        transport = jump_client.get_transport()
        assert transport is not None
        channel = transport.open_channel(
            "direct-tcpip",
            (hostname, port),
            ("127.0.0.1", 0),
        )

        # --- Second hop: connect through the channel ---
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=hostname,
            port=port,
            username=username,
            password=password,
            pkey=_make_pkey(identity_files),
            sock=channel,
            look_for_keys=True,
            allow_agent=True,
        )
    else:
        # Direct connection
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=hostname,
            port=port,
            username=username,
            password=password,
            pkey=_make_pkey(identity_files),
            look_for_keys=True,
            allow_agent=True,
        )

    # Stash jump client for cleanup
    client._jump_client = jump_client  # type: ignore[attr-defined]
    return client


def run_command(
    client: paramiko.SSHClient,
    cmd: str,
    timeout: int = 300,
) -> tuple[int, str, str]:
    """Execute *cmd* on the remote host and return ``(exit_code, stdout, stderr)``."""
    _, stdout_ch, stderr_ch = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout_ch.channel.recv_exit_status()
    stdout = stdout_ch.read().decode(errors="replace")
    stderr = stderr_ch.read().decode(errors="replace")
    return exit_code, stdout, stderr


def close(client: paramiko.SSHClient) -> None:
    """Close *client* and any associated jump client."""
    client.close()
    jump_client: Optional[paramiko.SSHClient] = getattr(client, "_jump_client", None)
    if jump_client is not None:
        jump_client.close()
