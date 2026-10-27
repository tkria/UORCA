"""``python -m uorca``: the same CLI as the ``uorca`` script, without the script shim.

On Windows the shim is ``.venv\\Scripts\\uorca.exe``. Some application-control policies block
``.exe`` files there but allow ``python.exe``, so ``uv run python -m uorca ...`` still works.
"""

from uorca.cli import main

if __name__ == "__main__":
    main()
