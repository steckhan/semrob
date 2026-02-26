# AGENTS.md
# Repository guidance for coding agents
# Scope: repository root and all subdirectories

## Overview
- Stack: Next.js 14 App Router + React 18 + TypeScript (strict mode).
- Purpose: local web app for ComfyUI inpainting workflows.
- Runtime profile: server routes interact with local filesystem and ComfyUI HTTP API.
- Package manager: npm (`package-lock.json` present).
- Path alias: `@/*` maps to repository root via `tsconfig.json`.
- Main code lives in `app/` and `lib/`.

## Build, Lint, and Test Commands

### Install and Run
- Install dependencies: `npm install`
- Run development server: `npm run dev`
- Build production bundle: `npm run build`
- Start production server: `npm run start`
- Lint project: `npm run lint`

### Tests (Current State)
- No test framework is currently configured in `package.json`.
- There is no `test` script, no test runner config, and no project test files.
- `npm test` will fail unless a test script is added.

### Single-Test Execution (Current + Recommended)
- Current single-test command: not available (no test runner configured).
- Recommended when adding Vitest:
  - Add script: `"test": "vitest run"`
  - Run all tests: `npm test`
  - Run single file: `npm test -- app/api/jobs/route.test.ts`
  - Run single test case: `npm test -- app/api/jobs/route.test.ts -t "returns 400 without image"`
- Recommended when adding Jest instead:
  - Add script: `"test": "jest"`
  - Run single file: `npm test -- app/api/jobs/route.test.ts`
  - Run single test case: `npm test -- app/api/jobs/route.test.ts -t "returns 400 without image"`

### Useful Maintenance Commands
- Type-check only: `npx tsc --noEmit`
- Remove Next build cache if needed: `rm -rf .next`

## Repository Structure
- `app/`: Next.js App Router pages, API routes, and UI components.
- `lib/`: workflow patching, job orchestration, storage helpers, shared types/constants.
- `workflows/`: ComfyUI workflow JSON files + workflow mapping JSON.
- `data/`: local runtime data (`uploads/`, `outputs/`, `jobs/`).
- `.next/`: generated build/dev output (do not edit manually).

## Agent Operating Rules
- Do not commit or edit generated/runtime artifacts in `.next/` or `data/` unless explicitly requested.
- Prefer changes in `app/`, `lib/`, `workflows/workflow-mapping.json`, and docs.
- Keep this app local-first: avoid adding cloud dependencies unless requested.
- Preserve current environment-driven behavior for paths and ComfyUI connectivity.

## Code Style Guidelines

### General Principles
- Prefer explicit, readable code over clever abstractions.
- Keep functions focused and avoid deep nesting.
- Use guard clauses and early returns for invalid states.
- Keep side effects localized (filesystem, network, timers).
- Favor deterministic behavior and idempotent helpers where practical.

### Imports and Module Structure
- Use ESM import syntax consistently.
- Order imports: built-in modules, third-party modules, then local modules.
- In local imports, prefer `@/` alias in `app/*`; relative imports are fine in `lib/*` when clear.
- Use `import type` for type-only imports.
- Remove unused imports and dead exports.

### TypeScript and Types
- Respect `strict: true`; do not weaken compiler settings.
- Add explicit return types to exported functions.
- Model domain entities with named type aliases (`JobRecord`, `WorkflowMapping`, etc.).
- Avoid `any`; if unavoidable, constrain and narrow immediately.
- Validate untrusted input at boundaries (`request.json()`, `formData`, file IO payloads).

### React and Next.js Conventions
- Use `"use client"` only where client hooks/browser APIs are required.
- Keep server logic in route handlers or `lib/` utilities.
- Keep UI state updates immutable and predictable.
- Prefer derived state with `useMemo` only when it improves clarity/perf.
- Clean up side effects in `useEffect` (timers, object URLs, event ownership).

### Formatting
- Match existing style:
  - 2-space indentation.
  - semicolons enabled.
  - double quotes for strings.
  - trailing commas in multiline literals/calls.
- Prefer multiline formatting over very long lines.
- Keep JSX readable; extract helper functions/types when blocks become large.

### Naming Conventions
- `camelCase`: variables, functions, local helpers.
- `PascalCase`: React components and type names.
- `SCREAMING_SNAKE_CASE`: true constants (`DEFAULT_PARAMS`, `DATA_ROOT`).
- Use descriptive boolean names (`isReady`, `isSubmitting`, `hasPaintedRef`).
- Name files by feature or primary export.

### Error Handling and Logging
- Throw errors with actionable context.
- In API routes, return `NextResponse.json` with clear `error` messages and proper status codes.
- Preserve `ENOENT` special-casing where used for optional files.
- Avoid swallowing failures from filesystem/network calls.
- Prefer failing fast over silently continuing on malformed state.

### Filesystem and Path Safety
- Use `path.join`/`path.basename`; avoid manual path concatenation.
- Create required directories with `fs.mkdir(..., { recursive: true })`.
- Keep Windows/WSL path translation behavior intact unless intentionally refactoring.
- Do not assume working directory stability; rely on constants/root resolution helpers.

### Concurrency and Async Patterns
- Use bounded concurrency for workflow fan-out (`runWithConcurrency`).
- Keep polling loops bounded by clear completion conditions.
- Ensure asynchronous background jobs persist status transitions (`queued` -> `running` -> terminal state).
- Prefer `Promise.all` for independent async operations.

### Testing Guidance (When Added)
- Place tests near code or under a consistent `__tests__/` layout.
- Prioritize unit tests for `lib/*` (workflow mapping/patching, job store, concurrency).
- Add API route tests for validation and status-code behavior.
- Keep tests deterministic; mock filesystem/network interactions.

### Documentation and Maintenance
- Update `README.md` when setup, commands, or runtime behavior changes.
- Update this file when adding test tooling, lint config, or formatter config.
- Record exact single-test and single-test-case commands once a runner is installed.

## Cursor and Copilot Rules
- No Cursor rules detected in `.cursor/rules/`.
- No `.cursorrules` file detected.
- No Copilot instructions detected in `.github/copilot-instructions.md`.

## Change Management for Agents
- Keep changes minimal and scoped to the user request.
- Do not refactor unrelated modules opportunistically.
- If touching runtime paths/constants, explain impact in PR/commit notes.
- Prefer adding small typed helpers over inlining repeated logic.
