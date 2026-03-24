# GitHub Pages Deployment Design

**Date:** 2026-03-24
**Status:** Approved

## Problem

The Python nSTAT GitHub Pages site (cajigaslab.github.io/nSTAT-python/) is frozen at a Feb 28, 2026 commit. The `docs-build` CI job verifies Sphinx compilation but never deploys the output. Over 20 merges to main have had no effect on the live site.

## Solution

A dedicated `deploy-docs.yml` workflow that builds Sphinx HTML with the Read the Docs theme and deploys it to GitHub Pages on every push to main.

## Components

### 1. `.github/workflows/deploy-docs.yml`

New workflow triggered by push to main and workflow_dispatch. Uses GitHub's artifact-based Pages deployment (`actions/upload-pages-artifact` + `actions/deploy-pages`). Requires `pages: write` and `id-token: write` permissions. Concurrency group prevents overlapping deployments.

### 2. `docs/conf.py` updates

Switch from Alabaster (default) to `sphinx_rtd_theme`. Keep `myst_parser` extension. Set project metadata.

### 3. `pyproject.toml` updates

Add `sphinx-rtd-theme` to dev dependencies if not already present.

### 4. `.nojekyll` marker

Created during build to prevent Jekyll from ignoring `_static/` directories.

## What Gets Deployed

The existing docs/index.rst toctree: help home, paper overview, class definitions, examples, API reference, paper examples gallery with figure thumbnails.

## What Doesn't Change

The existing `docs-build` job in ci.yml remains as a PR build-verification step.
