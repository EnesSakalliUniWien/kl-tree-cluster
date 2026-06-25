# Sync

Synchronize the real memory system first:

1. Read `wiki/index.md`.
2. Read the relevant wiki pages and cited source files.
3. Update wiki synthesis only from local evidence.
4. Append durable changes to `wiki/log.md`.
5. Run `make wiki-lint`.

Use mex checks as a secondary compatibility signal only:

```bash
npx mex-agent check --json
```

If mex reports scaffold drift, repair `.mex/` so it continues to point to the
wiki rather than copying project facts into a second memory system.
