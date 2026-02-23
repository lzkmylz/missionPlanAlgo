"""
Tests for Docker Compose security configuration.
Ensures no hardcoded secrets and proper environment variable usage.
"""

import re
import yaml
from pathlib import Path


class TestDockerSecurity:
    """Test suite for Docker security compliance."""

    def test_docker_compose_no_hardcoded_passwords(self):
        """
        CRITICAL: Verify no hardcoded passwords in docker-compose.yml.
        All passwords should use environment variable references.
        """
        compose_path = Path(__file__).parent.parent.parent / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml should exist"

        with open(compose_path, 'r') as f:
            content = f.read()

        # Parse YAML to check structure
        compose = yaml.safe_load(content)

        # Check MySQL service uses env_file or environment variable references
        if 'services' in compose and 'mysql' in compose['services']:
            mysql_service = compose['services']['mysql']

            # Should use env_file for secrets
            has_env_file = 'env_file' in mysql_service

            # Or should use variable references (not literal passwords)
            env_vars = mysql_service.get('environment', {})
            hardcoded_passwords = []

            for key, value in env_vars.items():
                if isinstance(value, str) and 'PASSWORD' in key.upper():
                    # Check if it's a literal password (not a variable reference)
                    if not value.startswith('${') and value != '':
                        hardcoded_passwords.append(f"{key}: {value}")

            assert len(hardcoded_passwords) == 0, (
                f"CRITICAL: Hardcoded passwords found in docker-compose.yml: "
                f"{hardcoded_passwords}. Use environment variables instead."
            )

    def test_env_file_exists(self):
        """Verify .env.example template file exists."""
        env_example_path = Path(__file__).parent.parent.parent / ".env.example"
        assert env_example_path.exists(), (
            ".env.example should exist as a template for environment variables"
        )

    def test_env_example_contains_required_variables(self):
        """Verify .env.example contains all required database variables."""
        env_example_path = Path(__file__).parent.parent.parent / ".env.example"

        with open(env_example_path, 'r') as f:
            content = f.read()

        required_vars = [
            'MYSQL_ROOT_PASSWORD',
            'MYSQL_PASSWORD',
            'MYSQL_DATABASE',
            'MYSQL_USER',
        ]

        for var in required_vars:
            assert var in content, (
                f".env.example should contain {var} environment variable"
            )

    def test_gitignore_includes_env(self):
        """Verify .gitignore includes .env file to prevent committing secrets."""
        gitignore_path = Path(__file__).parent.parent.parent / ".gitignore"

        with open(gitignore_path, 'r') as f:
            content = f.read()

        # Check for .env pattern (should be ignored but not .env.example)
        assert '.env' in content, (
            ".gitignore should include '.env' to prevent committing secrets"
        )

    def test_env_example_not_in_gitignore(self):
        """Verify .env.example is NOT in .gitignore (it's a safe template)."""
        gitignore_path = Path(__file__).parent.parent.parent / ".gitignore"

        with open(gitignore_path, 'r') as f:
            lines = f.readlines()

        # Check that .env.example is not ignored
        for line in lines:
            # Skip comments and empty lines
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # If line contains .env, it should not be .env.example
                if '.env' in stripped and '.env.example' in stripped:
                    assert False, (
                        ".env.example should NOT be in .gitignore - it's a safe template"
                    )

    def test_docker_compose_uses_env_file(self):
        """Verify docker-compose.yml uses env_file for MySQL service."""
        compose_path = Path(__file__).parent.parent.parent / "docker-compose.yml"

        with open(compose_path, 'r') as f:
            compose = yaml.safe_load(f)

        if 'services' in compose and 'mysql' in compose['services']:
            mysql_service = compose['services']['mysql']

            # Should reference env_file
            assert 'env_file' in mysql_service, (
                "MySQL service should use 'env_file' to load environment variables"
            )

            env_files = mysql_service['env_file']
            if isinstance(env_files, list):
                assert '.env' in env_files, (
                    "env_file should include '.env'"
                )
            else:
                assert env_files == '.env', (
                    "env_file should be '.env'"
                )
