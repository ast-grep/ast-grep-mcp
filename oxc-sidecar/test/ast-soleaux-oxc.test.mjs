import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { after, test } from "node:test";
import { spawn, spawnSync } from "node:child_process";

const sidecar = fileURLToPath(new URL("../bin/ast-soleaux-oxc.mjs", import.meta.url));
const temporaryRoots = [];

after(() => {
  for (const root of temporaryRoots) {
    rmSync(root, { force: true, recursive: true });
  }
});

function temporaryProject() {
  const root = mkdtempSync(join(tmpdir(), "ast-soleaux-oxc-"));
  temporaryRoots.push(root);
  return root;
}

function runSidecar(request, ...argumentsList) {
  return spawnSync(process.execPath, [sidecar, ...argumentsList], {
    encoding: "utf8",
    input: request === null ? undefined : JSON.stringify(request),
  });
}

function runServerRequests(requests) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(process.execPath, [sidecar, "--serve"], {
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.on("error", rejectPromise);
    child.on("close", (code) => {
      if (code !== 0) return rejectPromise(new Error(`worker exited ${code}: ${stderr}`));
      resolvePromise(
        stdout
          .trim()
          .split("\n")
          .map((line) => JSON.parse(line)),
      );
    });
    child.stdin.end(`${requests.map((request) => JSON.stringify(request)).join("\n")}\n`);
  });
}

test("reports the pinned helper and Oxc dependency versions", () => {
  const completed = runSidecar(null, "--version-json");
  assert.equal(completed.status, 0);
  assert.deepEqual(JSON.parse(completed.stdout), {
    helper: "0.1.0",
    parser: "0.147.0",
    resolver: "11.24.2",
  });
});

test("returns contained static module edges, dynamic facts, and parser diagnostics", () => {
  const root = temporaryProject();
  mkdirSync(join(root, "src"));
  writeFileSync(join(root, "package.json"), '{"type":"module"}\n');
  writeFileSync(join(root, "src", "dep.js"), "export const value = 1;\n");
  writeFileSync(
    join(root, "src", "common.cjs"),
    'const dep = require("./dep.js");\nmodule.exports = dep;\n',
  );
  writeFileSync(
    join(root, "src", "entry.ts"),
    [
      "const π = 1;",
      'import { value } from "./dep.js";',
      'export { value } from "./dep.js";',
      'import fs from "node:fs";',
      'import missing from "./missing.js";',
      'import("./lazy.js");',
      "console.log(value, fs, missing, import.meta.url);",
      "",
    ].join("\n"),
  );

  const completed = runSidecar({
    project_root: root,
    files: ["src/entry.ts", "src/common.cjs"],
    include_dynamic: true,
  });
  assert.equal(completed.status, 0, completed.stderr);
  const result = JSON.parse(completed.stdout);
  assert.deepEqual(result.versions, {
    helper: "0.1.0",
    parser: "0.147.0",
    resolver: "11.24.2",
  });
  assert.equal(result.graph_version, 1);
  assert.equal(result.modules.length, 2);
  assert.equal(result.graph.nodes.length, 2);
  const module = result.modules.find((item) => item.file === "src/entry.ts");
  assert.ok(module);
  assert.equal(module.file, "src/entry.ts");
  assert.equal(module.has_module_syntax, true);
  assert.equal(module.import_meta_spans.length, 1);
  assert.equal(module.diagnostics.length, 0);
  assert.deepEqual(
    module.edges.map((edge) => edge.kind),
    ["import", "reexport", "import", "import", "dynamic"],
  );

  const resolved = module.edges[0];
  assert.equal(resolved.start, 35);
  assert.equal(resolved.resolution, "resolved");
  assert.equal(resolved.resolved_path, realpathSync(join(root, "src", "dep.js")));
  assert.equal(resolved.package_json_path, realpathSync(join(root, "package.json")));

  const builtin = module.edges.find((edge) => edge.specifier === "node:fs");
  assert.equal(builtin.resolution, "external");
  assert.equal(builtin.resolved_path, null);

  const missing = module.edges.find((edge) => edge.specifier === "./missing.js");
  assert.equal(missing.resolution, "unresolved");
  assert.match(missing.resolution_error, /Cannot find module/);

  const dynamic = module.edges.at(-1);
  assert.equal(dynamic.resolution, "dynamic");
  assert.equal(dynamic.expression, '"./lazy.js"');

  const commonjs = result.modules.find((item) => item.file === "src/common.cjs");
  assert.ok(commonjs);
  assert.equal(commonjs.edges[0].module_system, "commonjs");
  assert.equal(commonjs.edges[0].specifier, "./dep.js");
  assert.equal(commonjs.commonjs_exports[0].text, "module.exports");
  assert.equal(commonjs.package.path, "package.json");
  assert.ok(
    result.graph.edges.some(
      (edge) => edge.module_system === "commonjs" && edge.target === "src/dep.js",
    ),
  );
});

test("persistent worker caches module graphs and recovers from invalid requests", async () => {
  const root = temporaryProject();
  writeFileSync(join(root, "package.json"), '{"type":"module"}\n');
  writeFileSync(join(root, "entry.js"), 'import "./dep.js";\n');
  writeFileSync(join(root, "dep.js"), "export {};\n");
  const request = { project_root: root, files: ["entry.js"], include_dynamic: false };
  const responses = await runServerRequests([request, request, {}, request]);
  assert.equal(responses.length, 4);
  assert.equal(responses[0].cache_hit, false);
  assert.equal(responses[1].cache_hit, true);
  assert.match(responses[2].error, /project_root/);
  assert.equal(responses[3].modules[0].file, "entry.js");
});

test("rejects the removed formatter operation", () => {
  const completed = runSidecar({
    operation: "format",
    filename: "sample.ts",
    code: "const x={a:1}",
    options: {},
  });
  assert.equal(completed.status, 1);
  assert.match(completed.stderr, /Unsupported operation: format/);
});

test("transforms and minifies through curated operations", () => {
  const transformed = runSidecar({
    operation: "transform",
    filename: "sample.ts",
    code: "const x: number = 1",
    options: { lang: "ts" },
  });
  assert.equal(transformed.status, 0, transformed.stderr);
  assert.equal(JSON.parse(transformed.stdout).result.code, "const x = 1;\n");

  const minified = runSidecar({
    operation: "minify",
    filename: "sample.js",
    code: "function add(a, b) { return a + b; }",
    options: {},
  });
  assert.equal(minified.status, 0, minified.stderr);
  assert.match(JSON.parse(minified.stdout).result.code, /function add/);
});

test("rejects paths outside the requested project", () => {
  const root = temporaryProject();
  const outside = join(dirname(root), `${basename(root)}-outside.js`);
  writeFileSync(outside, "export {};\n");
  temporaryRoots.push(outside);
  const completed = runSidecar({
    project_root: root,
    files: [`../${basename(outside)}`],
    include_dynamic: false,
  });
  assert.equal(completed.status, 1);
  assert.match(completed.stderr, /outside project_root/);
});

test("rejects invalid UTF-8 source", () => {
  const root = temporaryProject();
  writeFileSync(join(root, "entry.js"), Buffer.from([0xff]));
  const completed = runSidecar({ project_root: root, files: ["entry.js"], include_dynamic: false });
  assert.equal(completed.status, 1);
  assert.match(completed.stderr, /not valid UTF-8/);
});

test("rejects unknown request fields", () => {
  const root = temporaryProject();
  const completed = runSidecar({
    project_root: root,
    files: ["entry.js"],
    include_dynamic: false,
    extra: true,
  });
  assert.equal(completed.status, 1);
  assert.match(completed.stderr, /unknown key/);
});
