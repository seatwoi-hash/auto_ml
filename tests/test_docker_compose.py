from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_compose():
    with (PROJECT_ROOT / "docker-compose.yml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_compose_contains_required_services():
    services = load_compose()["services"]

    assert set(services) == {"app", "nextcloud", "nextcloud-db"}
    assert services["app"]["depends_on"]["nextcloud"]["condition"] == "service_healthy"
    assert (
        services["nextcloud"]["depends_on"]["nextcloud-db"]["condition"]
        == "service_healthy"
    )


def test_app_uses_environment_names_expected_by_python_code():
    app = load_compose()["services"]["app"]
    environment = app["environment"]

    assert environment["NEXTCLOUD_URL"] == "http://nextcloud:80"
    assert "NEXTCLOUD_USER" in environment
    assert "NEXTCLOUD_PASSWORD" in environment
    assert "API_KEY" in environment
    assert "NEXTCLOUD_USERNAME" not in environment


def test_app_mounts_model_read_only():
    volumes = load_compose()["services"]["app"]["volumes"]

    assert "./model_checkpoint.pth:/app/model.pth:ro" in volumes
    assert "uploads_data:/app/uploads" in volumes


def test_stateful_services_have_persistent_storage_and_healthchecks():
    services = load_compose()["services"]

    assert "healthcheck" in services["nextcloud"]
    assert "healthcheck" in services["nextcloud-db"]
    assert "healthcheck" in services["app"]
    assert "nextcloud_data:/var/www/html/data" in services["nextcloud"]["volumes"]
    assert "nextcloud_config:/var/www/html/config" in services["nextcloud"]["volumes"]
    assert (
        "nextcloud_db_data:/var/lib/postgresql/data"
        in services["nextcloud-db"]["volumes"]
    )
