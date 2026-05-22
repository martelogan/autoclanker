# Site Operations

Generated sites are host-neutral. Persistence is optional and adapter-based.

Adapter methods may include:

- `load({ appId })`
- `save({ appId, registry })`
- `listSnapshots({ appId })`
- `saveSnapshot({ appId, registry, name })`
- `loadSnapshot({ appId, snapshotId })`
- `writeArtifacts({ appId, artifacts })`
- `fetchIssue({ url, issue })`

If an adapter is absent, the site uses browser localStorage. A host-specific
adapter should live beside the generated site or in the host project, not inside
the generic package.

Generated files are disposable. Update the generic generator when changing
layout, accessibility, artifact versioning, validation, or adapter contracts;
update only the host registry/adapter when changing portfolio content or
private persistence.
