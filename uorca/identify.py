"""
UORCA Dataset Identification CLI Wrapper

``uorca identify`` (``uorca/cli.py``) loads ``.env``, checks the required environment
variables and builds ``sys.argv``, then calls :func:`main` here, which runs the
identification module.

This wrapper used to repeat the environment check and ``os.chdir`` into the package
directory before running. The chdir made a relative ``-o`` land inside the installed
package instead of the user's working directory, so it was removed; the CLI now also
passes an absolute output path.
"""


def main():
    """Run dataset identification with the arguments already in ``sys.argv``."""
    from uorca.identification.dataset_identification import main as identify_main

    identify_main()


__all__ = ["main"]

if __name__ == "__main__":
    main()
