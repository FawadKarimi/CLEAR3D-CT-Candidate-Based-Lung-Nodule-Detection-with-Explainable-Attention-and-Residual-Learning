"""Create an exact archive-hashed lock from binary wheels, without installing them."""
import argparse
import email
import hashlib
from pathlib import Path
import re
import zipfile

# A fresh venv has pip, but need not have standalone packaging installed yet.
try:
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.version import Version
except ModuleNotFoundError as exc:
    if exc.name not in {"packaging", "packaging.utils", "packaging.version"}:
        raise
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
    from pip._vendor.packaging.version import Version


def lock_lines(directory):
    wheels = sorted(Path(directory).glob("*.whl"))
    if not wheels:
        raise ValueError("Empty wheelhouse")
    seen = set()
    lines = []
    for wheel in wheels:
        filename_name, filename_version, _, _ = parse_wheel_filename(wheel.name)
        with zipfile.ZipFile(wheel) as archive:
            # Inspect ZipInfo entries, not a set/dict: duplicate entries must fail.
            members = [info for info in archive.infolist()
                       if len(info.filename.split("/")) == 2
                       and info.filename.split("/")[0].endswith(".dist-info")
                       and info.filename.split("/")[1] == "METADATA"]
            if len(members) != 1:
                raise ValueError("Exactly one top-level metadata record per wheel required")
            metadata = email.message_from_bytes(archive.read(members[0]))
        names = metadata.get_all("Name", [])
        versions = metadata.get_all("Version", [])
        if len(names) != 1 or len(versions) != 1:
            raise ValueError("Exactly one metadata Name and Version required")
        name, version = names[0], versions[0]
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", name):
            raise ValueError("Invalid wheel identity")
        normalized = canonicalize_name(name)
        parsed_version = Version(version)
        if normalized != filename_name or parsed_version != filename_version:
            raise ValueError("Wheel filename and metadata identity mismatch")
        directory_identity = members[0].filename.split("/")[0][:-len(".dist-info")]
        try:
            directory_name, directory_version = directory_identity.rsplit("-", 1)
            if (canonicalize_name(directory_name) != filename_name
                    or Version(directory_version) != filename_version):
                raise ValueError("Top-level metadata directory identity mismatch")
        except ValueError as exc:
            raise ValueError("Top-level metadata directory identity mismatch") from exc
        if normalized in seen:
            raise ValueError("Duplicate package wheel: " + normalized)
        seen.add(normalized)
        digest = hashlib.sha256()
        with wheel.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        lines.append(f"{normalized}=={parsed_version} --hash=sha256:{digest.hexdigest()}")
    return sorted(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lines = lock_lines(args.wheelhouse)
    with args.output.open("x") as stream:
        stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
