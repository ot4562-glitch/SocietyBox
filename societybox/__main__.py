"""Society Box CLI, sharing the existing offline research engine."""

from econscope.__main__ import main as _engine_main


def main():
    return _engine_main(prog="societybox")


if __name__ == "__main__":
    raise SystemExit(main())
