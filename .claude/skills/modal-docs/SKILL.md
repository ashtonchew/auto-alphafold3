---
name: modal-docs
description: >
  Use when a task depends on current Modal Python APIs, migration guidance,
  deployed object lookup, or docs-backed runtime behavior.
---

# Modal Docs

Use this skill when a task depends on current Modal APIs or when existing
code may rely on stale pre-1.0 patterns. Open the smallest relevant doc set
first instead of crawling the full corpus.

## Workflow

1. Pick the closest domain below before changing code.
2. Read the linked guide/reference pages and prefer current APIs over memory.
3. If the task smells like a migration, open the migration/deprecation docs
   before choosing a fix.
4. Expand to adjacent references only when the first page does not answer the
   task.

## Reference Groups

### Images and build-time artifacts
Use for image construction, dependency installation, and stale build-time patterns that now need image-based replacements.

- [Defining Images](references/images.md)
- [modal.Image](references/modal-image.md)
- [Using existing container images](references/existing-images.md)

### Local files and Python source
Use for add_local_*, add_local_python_source, file staging, and Mount-to-image migrations.

- [Passing local data](references/local-data.md)
- [File access](references/sandbox-files.md)
- [Modal 1.0 migration guide](references/modal-1-0-migration.md)

### Deployed objects and cross-app lookup
Use for app wiring, deployed object discovery, and looking up functions or classes across apps.

- [Apps, Functions, and entrypoints](references/apps.md)
- [Managing deployments](references/managing-deployments.md)
- [Invoking deployed functions](references/trigger-deployed-functions.md)
- [modal.App](references/modal-app.md)
- [modal.Function](references/modal-function.md)

### Volumes and persistence
Use for persisted files, cache state, Volume attachment semantics, and commit/reload behavior.

- [Volumes](references/volumes.md)
- [modal.Volume](references/modal-volume.md)

### Web endpoints and app wiring
Use for endpoint decorators, request handling, and wiring deployed apps to HTTP entrypoints.

- [Web Functions](references/webhooks.md)
- [modal.fastapi_endpoint](references/modal-fastapi-endpoint.md)
- [modal.App](references/modal-app.md)

### Migration and deprecations
Use when snippets mention deprecated Modal APIs such as Mount, copy_local_*, older lifecycle hooks, or outdated lookup patterns.

- [Modal 1.0 migration guide](references/modal-1-0-migration.md)
- [modal.Error](references/modal-error.md)
- [Container lifecycle hooks](references/lifecycle-functions.md)
