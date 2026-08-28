from pathlib import Path

from lspe.networks.mapping_closeout import verify_mapping_checksums


def test_mapping_checksum_verification_detects_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("stable")
    import hashlib

    digest = hashlib.sha256(b"stable").hexdigest()
    (tmp_path / "checksums.sha256").write_text(f"{digest}  artifact.txt\n")
    assert verify_mapping_checksums(tmp_path)
    artifact.write_text("changed")
    assert not verify_mapping_checksums(tmp_path)
