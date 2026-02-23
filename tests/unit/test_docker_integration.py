"""Tests for Docker integration."""
import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestDockerfile:
    """Test Dockerfile exists and has correct structure."""

    def test_dockerfile_exists(self):
        """Should have Dockerfile in project root."""
        dockerfile = Path("Dockerfile")
        assert dockerfile.exists(), "Dockerfile should exist in project root"

    def test_dockerfile_uses_python_slim(self):
        """Should use python:3.10-slim base image."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert "python:3.10-slim" in content or "python:3.11-slim" in content, \
            "Should use python slim base image"

    def test_dockerfile_installs_requirements(self):
        """Should install requirements."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert "requirements.txt" in content, "Should reference requirements.txt"
        assert "pip install" in content, "Should pip install requirements"

    def test_dockerfile_installs_package(self):
        """Should install package in editable mode."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert "pip install -e ." in content, "Should install package in editable mode"

    def test_dockerfile_sets_entrypoint(self):
        """Should set sat-cli as entrypoint."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert "ENTRYPOINT" in content, "Should have ENTRYPOINT"
        assert "sat-cli" in content, "Should reference sat-cli"


class TestDockerCompose:
    """Test docker-compose.yml configuration."""

    def test_docker_compose_exists(self):
        """Should have docker-compose.yml in project root."""
        compose = Path("docker-compose.yml")
        assert compose.exists(), "docker-compose.yml should exist"

    def test_docker_compose_has_scenario_tools_service(self):
        """Should have scenario-tools service."""
        compose = Path("docker-compose.yml")
        content = compose.read_text()
        assert "scenario-tools:" in content, "Should have scenario-tools service"

    def test_docker_compose_has_volume_mounts(self):
        """Should mount entity_lib, scenarios, and results."""
        compose = Path("docker-compose.yml")
        content = compose.read_text()
        assert "entity_lib" in content, "Should mount entity_lib"
        assert "scenarios" in content, "Should mount scenarios"
        assert "results" in content, "Should mount results"

    def test_docker_compose_sets_environment(self):
        """Should set environment variables."""
        compose = Path("docker-compose.yml")
        content = compose.read_text()
        assert "ENTITY_LIB_PATH" in content, "Should set ENTITY_LIB_PATH"
        assert "SCENARIOS_PATH" in content, "Should set SCENARIOS_PATH"


class TestDockerIgnore:
    """Test .dockerignore file."""

    def test_dockerignore_exists(self):
        """Should have .dockerignore file."""
        dockerignore = Path(".dockerignore")
        assert dockerignore.exists(), ".dockerignore should exist"

    def test_dockerignore_excludes_git(self):
        """Should exclude .git directory."""
        dockerignore = Path(".dockerignore")
        content = dockerignore.read_text()
        assert ".git" in content, "Should exclude .git"

    def test_dockerignore_excludes_pycache(self):
        """Should exclude __pycache__."""
        dockerignore = Path(".dockerignore")
        content = dockerignore.read_text()
        assert "__pycache__" in content or "*.pyc" in content, \
            "Should exclude Python cache"

    def test_dockerignore_excludes_venv(self):
        """Should exclude virtual environments."""
        dockerignore = Path(".dockerignore")
        content = dockerignore.read_text()
        assert "venv" in content or "env/" in content, \
            "Should exclude virtual environments"


class TestGitHubActions:
    """Test GitHub Actions CI/CD workflow."""

    def test_github_workflows_directory_exists(self):
        """Should have .github/workflows directory."""
        workflows_dir = Path(".github/workflows")
        assert workflows_dir.exists(), ".github/workflows directory should exist"

    def test_scenario_validation_workflow_exists(self):
        """Should have scenario validation workflow."""
        workflow = Path(".github/workflows/scenario-validation.yml")
        assert workflow.exists(), "scenario-validation.yml workflow should exist"

    def test_workflow_triggers_on_scenario_changes(self):
        """Should trigger on scenario file changes."""
        workflow = Path(".github/workflows/scenario-validation.yml")
        content = workflow.read_text()
        assert "scenarios/" in content, "Should trigger on scenarios/ changes"

    def test_workflow_builds_docker_image(self):
        """Should build Docker image."""
        workflow = Path(".github/workflows/scenario-validation.yml")
        content = workflow.read_text()
        assert "docker build" in content, "Should build Docker image"

    def test_workflow_runs_validate(self):
        """Should run validate command."""
        workflow = Path(".github/workflows/scenario-validation.yml")
        content = workflow.read_text()
        assert "validate" in content, "Should run validate command"
