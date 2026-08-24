import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync } from "node:child_process";
import { after, test } from "node:test";

const worker = fileURLToPath(new URL("../bin/ast-soleaux-typescript-project.mjs", import.meta.url));
const roots = [];

function project(
  source = 'import { b } from "./b.js";\nimport { a } from "./a.js";\nexport const value = a + b;\n',
) {
  const root = mkdtempSync(join(tmpdir(), "ast-soleaux-typescript-"));
  roots.push(root);
  mkdirSync(join(root, "src"));
  writeFileSync(
    join(root, "tsconfig.json"),
    JSON.stringify({
      compilerOptions: {
        target: "ES2022",
        module: "NodeNext",
        moduleResolution: "NodeNext",
        rootDir: "src",
        outDir: "dist",
      },
      include: ["src/**/*.ts"],
    }),
  );
  writeFileSync(join(root, "src", "a.ts"), "export const a = 1;\n");
  writeFileSync(join(root, "src", "b.ts"), "export const b = 2;\n");
  writeFileSync(join(root, "src", "index.ts"), source);
  return root;
}

function run(root, extra = {}) {
  return spawnSync(process.execPath, [worker], {
    input: JSON.stringify({
      project_root: root,
      tsconfig: "tsconfig.json",
      include_emit: true,
      ...extra,
    }),
    encoding: "utf8",
  });
}

after(() => {
  for (const root of roots) rmSync(root, { recursive: true, force: true });
});

test("returns diagnostics, module resolution, symbols, inferred types, emit, and code actions", () => {
  const root = project();
  const completed = run(root);
  assert.equal(completed.status, 0, completed.stderr);
  const result = JSON.parse(completed.stdout);
  assert.equal(result.typescript_version, "6.0.2");
  assert.deepEqual(result.root_files.sort(), ["src/a.ts", "src/b.ts", "src/index.ts"]);
  assert.equal(result.diagnostics.length, 0);
  assert.ok(
    result.modules
      .find((module) => module.file === "src/index.ts")
      .imports.every((item) => item.resolution === "resolved"),
  );
  assert.ok(result.symbols.some((symbol) => symbol.name === "value" && symbol.type === "number"));
  assert.ok(result.emit.some((artifact) => artifact.file.endsWith("index.js")));
  assert.ok(result.code_actions.some((action) => action.kind === "organize_imports"));
  assert.match(result.source_digest, /^[0-9a-f]{64}$/);
  assert.equal(result.cache_hit, false);
});

test("reads bounded dependency declarations larger than the project source limit", () => {
  const root = project(
    'import { dependencyValue } from "large-types";\nexport const value = dependencyValue;\n',
  );
  const dependencyRoot = join(root, "node_modules", "large-types");
  mkdirSync(dependencyRoot, { recursive: true });
  writeFileSync(
    join(dependencyRoot, "package.json"),
    JSON.stringify({ name: "large-types", version: "1.0.0", types: "index.d.ts" }),
  );
  writeFileSync(
    join(dependencyRoot, "index.d.ts"),
    `export declare const dependencyValue: number;\n${" ".repeat(2 * 1024 * 1024)}`,
  );

  const completed = run(root, { include_emit: false, include_code_actions: false });
  assert.equal(completed.status, 0, completed.stderr);
  const result = JSON.parse(completed.stdout);
  assert.equal(result.diagnostics.length, 0);
});

test("reports compiler diagnostics and rejects malformed configuration", () => {
  const root = project("export const broken: string = 1;\n");
  const completed = run(root, { include_emit: false, include_code_actions: true });
  assert.equal(completed.status, 0, completed.stderr);
  const result = JSON.parse(completed.stdout);
  assert.ok(result.diagnostics.some((diagnostic) => diagnostic.code === 2322));
  assert.ok(result.code_actions.length >= 0);

  writeFileSync(join(root, "tsconfig.json"), "{ invalid json }");
  const malformed = run(root);
  assert.notEqual(malformed.status, 0);
  assert.match(malformed.stderr, /property name|JSON|Unexpected|expected/i);
});

test("rejects source roots outside project", () => {
  const root = project();
  const outside = join(dirname(root), `${basename(root)}-outside.ts`);
  writeFileSync(outside, "export const outside = true;\n");
  roots.push(outside);
  const completed = run(root, { paths: [`../${basename(outside)}`] });
  assert.notEqual(completed.status, 0);
  assert.match(completed.stderr, /outside project_root/);
});

test("persistent worker caches identical projects and invalidates on source changes", async () => {
  const root = project();
  const child = spawn(process.execPath, [worker, "--serve"], { stdio: ["pipe", "pipe", "pipe"] });
  const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
  const iterator = lines[Symbol.asyncIterator]();
  const request = JSON.stringify({
    project_root: root,
    tsconfig: "tsconfig.json",
    include_emit: false,
  });

  child.stdin.write(`${request}\n`);
  const first = JSON.parse((await iterator.next()).value);
  child.stdin.write(`${request}\n`);
  const second = JSON.parse((await iterator.next()).value);
  assert.equal(first.cache_hit, false);
  assert.equal(second.cache_hit, true);
  assert.equal(first.source_digest, second.source_digest);

  writeFileSync(join(root, "src", "index.ts"), "export const value = 99;\n");
  child.stdin.write(`${request}\n`);
  const third = JSON.parse((await iterator.next()).value);
  assert.equal(third.cache_hit, false);
  assert.notEqual(third.source_digest, first.source_digest);

  child.stdin.end();
  await new Promise((resolve) => child.once("close", resolve));
  assert.equal(child.exitCode, 0);
});
