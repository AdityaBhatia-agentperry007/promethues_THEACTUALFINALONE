from fastapi.testclient import TestClient

from backend.app import app


client = TestClient(app)


def test_healthz_and_scenarios():
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    scenarios = client.get("/scenarios")
    assert scenarios.status_code == 200
    assert len(scenarios.json()["scenarios"]) == 16


def test_predict_endpoint():
    response = client.post("/predict", json={"mach_sonic": 1.5, "mach_alfvenic": 0.7, "steps": 4})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["frames"]) == 4
    assert 0.0 <= payload["risk"] <= 1.0


def test_pir_fetch_endpoint():
    response = client.post("/pir/fetch", json={"scenario_index": 4, "method": "dpf"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["reconstructed_equals_direct"] is True
    assert payload["index_bits_leaked_to_any_single_server"] == 0


def test_agent_endpoint():
    response = client.post(
        "/agent",
        json={"request_text": "confidential supersonic sub-Alfvenic prediction", "channel": "dashboard"},
    )
    assert response.status_code == 200
    assert response.json()["result"]["used_private_fetch"] is True


def test_simulate_endpoint_reports_checkpoint_state():
    response = client.post("/simulate", json={"task": "generate a black hole simulation", "steps": 4, "mode": "lensing"})
    assert response.status_code == 200
    payload = response.json()
    assert "checkpoint_loaded" in payload["meta"]
    assert payload["meta"]["dataset_hint"] in {"MHD_64", "post_neutron_star_merger", "supernova_explosion_64"}
    assert payload["meta"]["mode"] == "lensing"
    assert "supported_modes" in payload["meta"]
    if not payload["meta"]["checkpoint_loaded"]:
        assert len(payload["frames"]) == 4
        assert payload["meta"]["data_source_kind"] == "deterministic_physics_renderer"
        assert "not a trained The Well" in payload["warning"]
