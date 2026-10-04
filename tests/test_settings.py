import pytest
from pydantic import ValidationError

from settings import Settings


def valid_settings(**overrides):
    values = {
        "nextcloud_url": "http://nextcloud:80",
        "nextcloud_user": "admin",
        "nextcloud_password": "secret",
        "api_key": "test-api-key-123456",
    }
    values.update(overrides)
    return Settings(**values)


def test_settings_accept_valid_configuration():
    settings = valid_settings(cors_origins=["https://photos.example.com"])

    assert settings.nextcloud_user == "admin"
    assert settings.max_file_size == 10 * 1024 * 1024


def test_settings_reject_short_api_key():
    with pytest.raises(ValidationError):
        valid_settings(api_key="short")


def test_settings_reject_wildcard_cors():
    with pytest.raises(ValidationError):
        valid_settings(cors_origins=["*"])
