from src.loop.state_manifest import StateManifest, load_manifest, write_manifest


def test_state_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    original = StateManifest(
        state_name="M_R0",
        parent_state=None,
        visible_round="R0",
        data_cutoff="before-R0",
        input_hashes={"public": "abc"},
    )
    write_manifest(original, path)
    loaded = load_manifest(path)
    assert loaded.state_name == "M_R0"
    assert loaded.visible_round == "R0"
    assert loaded.input_hashes == {"public": "abc"}
