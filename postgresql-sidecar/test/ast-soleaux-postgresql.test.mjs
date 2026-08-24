import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { spawnSync } from "node:child_process";

const worker = fileURLToPath(new URL("../bin/ast-soleaux-postgresql.mjs", import.meta.url));

function run(request) {
  const completed = spawnSync(process.execPath, [worker], {
    encoding: "utf8",
    input: JSON.stringify(request),
  });
  assert.equal(completed.status, 0, completed.stderr);
  assert.equal(completed.stderr.trim(), "");
  return JSON.parse(completed.stdout);
}

test("reports the resolved PostgreSQL 18 parser and latest deparser versions", () => {
  const completed = spawnSync(process.execPath, [worker, "--version-json"], { encoding: "utf8" });
  assert.equal(completed.status, 0, completed.stderr);
  assert.deepEqual(JSON.parse(completed.stdout), {
    worker: "0.1.0",
    parser: "18.0.0",
    deparser: "18.3.6",
    postgres_major: 18,
  });
});

test("returns PostgreSQL 18 parse and structural facts", () => {
  const sql = [
    "CREATE TABLE app.users (id bigint PRIMARY KEY, name text NOT NULL);",
    "SELECT lower(users.name) FROM app.users WHERE users.id = 7;",
  ].join("\n");
  const response = run({ operation: "parse", sql });
  assert.equal(response.worker_version, "0.1.0");
  assert.equal(response.parser_version, "18.0.0");
  assert.equal(response.deparser_version, "18.3.6");
  assert.equal(response.postgres_major, 18);
  assert.equal(response.operation, "parse");
  assert.match(response.source_digest, /^[0-9a-f]{64}$/);
  assert.equal(response.result.tree.version >= 180000, true);
  assert.equal(response.result.tree.stmts.length, 2);
  assert.ok(
    response.result.declarations.some((item) => item.kind === "table" && item.name === "app.users"),
  );
  assert.ok(
    response.result.references.some(
      (item) => item.kind === "relation" && item.name === "app.users",
    ),
  );
  assert.ok(
    response.result.calls.some((item) => item.kind === "function" && item.name === "lower"),
  );
});

test("supports scan, fingerprint, PL/pgSQL, and deparse equivalence", () => {
  const scan = run({ operation: "scan", sql: "SELECT id FROM app.users" });
  assert.ok(scan.result.tokens.some((token) => token.text.toLowerCase() === "select"));
  const fingerprint = run({
    operation: "fingerprint",
    sql: "SELECT id FROM app.users WHERE id = 7",
  });
  assert.match(fingerprint.result.fingerprint, /^[0-9a-f]{16}$/);
  assert.match(fingerprint.result.normalized, /\$1/);
  const functionSql = [
    "CREATE FUNCTION app.answer(input integer) RETURNS integer",
    "LANGUAGE plpgsql AS $$ BEGIN RETURN input + 1; END $$;",
  ].join("\n");
  const plpgsql = run({ operation: "plpgsql", sql: functionSql });
  assert.ok(plpgsql.result.plpgsql);
  const deparse = run({ operation: "deparse", sql: functionSql });
  assert.equal(deparse.result.equivalent, true);
  assert.match(deparse.result.deparsed_sql, /CREATE FUNCTION/i);
  assert.equal(deparse.result.original_tree_digest, deparse.result.reparsed_tree_digest);
});

test("sanitizes malformed deparse failures", () => {
  const completed = spawnSync(process.execPath, [worker], {
    encoding: "utf8",
    input: JSON.stringify({ operation: "deparse", sql: "SELECT FROM" }),
  });
  assert.equal(completed.status, 1);
  assert.equal(completed.stdout, "");
  const failure = JSON.parse(completed.stderr);
  assert.deepEqual(failure, {
    error: {
      type: "parse_error",
      message: "syntax error at end of input",
      cursor_position: 11,
      cursor_unit: "unicode_code_point",
    },
  });
  for (const forbidden of [
    "/Users/",
    "node_modules",
    "SqlError",
    "scan.l",
    "scanner_yyerror",
    "lineNumber",
  ]) {
    assert.equal(completed.stderr.includes(forbidden), false, forbidden);
  }
});

test("rejects unknown request fields and unsupported operations", () => {
  const unknown = spawnSync(process.execPath, [worker], {
    encoding: "utf8",
    input: JSON.stringify({ operation: "parse", sql: "SELECT 1", extra: true }),
  });
  assert.notEqual(unknown.status, 0);
  assert.match(unknown.stderr, /unknown key/);
  const unsupported = spawnSync(process.execPath, [worker], {
    encoding: "utf8",
    input: JSON.stringify({ operation: "execute", sql: "SELECT 1" }),
  });
  assert.notEqual(unsupported.status, 0);
  assert.match(unsupported.stderr, /unsupported operation/);
});
