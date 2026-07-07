# Deployment Guide

## Overview

This document describes the deployment process for the authentication service.
All engineers are expected to follow this process when shipping changes to production.

## Prerequisites

Before deploying, ensure the following:

- All unit tests pass locally with `pytest`
- The pull request has been reviewed and approved by at least one senior engineer
- The staging environment has been tested for at least 24 hours
- The on-call engineer has been notified via PagerDuty

## Deployment Steps

### Step 1: Merge the Pull Request

Once approved, merge the pull request into the main branch.
The CI pipeline will trigger automatically and run the full test suite.
Do not proceed until the CI pipeline shows a green checkmark.

### Step 2: Tag the Release

Create a Git tag for the release version:

```bash
git tag -a v2.1.4 -m "Release version 2.1.4"
git push origin v2.1.4
```

### Step 3: Deploy to Production

Run the deployment script from the repository root:

```bash
./scripts/deploy.sh v2.1.4
```

This script pulls the tagged Docker image, runs database migrations,
updates the load balancer, and performs a health check.
The process takes approximately 8 minutes.

## Rollback Procedures

If the deployment causes issues, follow these steps immediately.

### When to Roll Back

Roll back if any of the following occur within 30 minutes of deployment:

- Error rate exceeds 1% for more than 2 consecutive minutes
- P99 latency exceeds 2000ms
- Any critical alerts fire in PagerDuty
- The on-call engineer makes the call

### How to Roll Back

Run the revert script with the previous version number:

```bash
./scripts/revert.sh v2.1.3
```

This restores the previous Docker image and updates DNS automatically.
The rollback process takes approximately 3 minutes.

After rolling back, notify the engineering channel in Slack immediately.
Create an incident report within 24 hours using the template in Notion.

## Monitoring

After a successful deployment, monitor the following dashboards for 30 minutes:

- Error rate dashboard: grafana.internal/d/errors
- Latency dashboard: grafana.internal/d/latency
- Infrastructure dashboard: grafana.internal/d/infra

If all metrics are stable after 30 minutes, the deployment is considered complete.