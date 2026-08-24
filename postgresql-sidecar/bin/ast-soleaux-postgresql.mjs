#!/usr/bin/env node

import crypto from "node:crypto";
import { readFileSync, realpathSync, statSync } from "node:fs";
import { readFile, realpath, stat } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, isAbsolute, parse, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import parserModule from "@libpg-query/parser";
import { deparseSync } from "pgsql-deparser";

const HELPER_VERSION = "0.1.0";
const PARSER_VERSION = "18.0.0";
const DEPARSER_VERSION = "18.3.6";
const POSTGRESQL_MAJOR = 18;
const PROTOCOL_VERSION = "ast-soleaux.postgresql/v1";
const MAX_FRAME_BYTES = 1024 * 1024;
const MAX_RESPONSE_BYTES = 32 * 1024 * 1024;
const MAX_SQL_BYTES = 4 * 1024 * 1024;
const MAX_FILE_BYTES = 4 * 1024 * 1024;
const MAX_TOTAL_FILE_BYTES = 16 * 1024 * 1024;
const MAX_FILES = 64;
const MAX_FACTS = 10_000;
const MAX_MESSAGE_CHARACTERS = 4000;
const LINE_FEED_BYTE = 10;
const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const packageManifest = JSON.parse(readFileSync(resolve(packageRoot, "package.json"), "utf8"));
const require = createRequire(import.meta.url);
const parserManifest = require("@libpg-query/parser/package.json");
const deparserManifest = require("pgsql-deparser/package.json");
const parser = parserModule;
const versions = Object.freeze({
  worker: packageManifest.version,
  parser: parserManifest.version,
  deparser: deparserManifest.version,
  postgres_major: POSTGRESQL_MAJOR,
});

if (
  packageManifest.version !== HELPER_VERSION ||
  parserManifest.version !== PARSER_VERSION ||
  deparserManifest.version !== DEPARSER_VERSION
) {
  throw new Error(
    `PostgreSQL sidecar identity mismatch: ${packageManifest.version}/${parserManifest.version}/${deparserManifest.version}`,
  );
}

await parser.loadModule();

function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function rejectUnknownKeys(value, allowed, label) {
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) {
      throw new Error(`${label} contains an unknown key: ${key}`);
    }
  }
}

function boundedMessage(value) {
  const text = Error.isError(value) ? value.message : String(value);
  return text.length <= MAX_MESSAGE_CHARACTERS ? text : `${text.slice(0, MAX_MESSAGE_CHARACTERS)}…`;
}

class RequestFailure extends Error {}

function normalizedFailure(error) {
  if (error instanceof RequestFailure) {
    return { type: "invalid_request", message: boundedMessage(error) };
  }
  if (error instanceof parser.SqlError || parser.hasSqlDetails(error)) {
    const details = parser.hasSqlDetails(error) ? error.sqlDetails : null;
    return {
      type: "parse_error",
      message: details?.message ?? boundedMessage(error),
      cursor_position: Number.isSafeInteger(details?.cursorPosition)
        ? details.cursorPosition
        : null,
      cursor_unit: "unicode_code_point",
    };
  }
  return { type: "worker_error", message: "PostgreSQL worker failed" };
}

function writeFailure(error) {
  process.stderr.write(`${JSON.stringify({ error: normalizedFailure(error) })}\n`);
  process.exitCode = 1;
}

function isWithin(path, root) {
  const fromRoot = relative(root, path);
  return (
    fromRoot === "" ||
    (fromRoot !== ".." && !fromRoot.startsWith(`..${sep}`) && !isAbsolute(fromRoot))
  );
}

function validateRequest(value) {
  if (!isRecord(value)) throw new Error("request must be a JSON object");
  rejectUnknownKeys(value, new Set(["operation", "sql"]), "request");
  if (!["parse", "scan", "fingerprint", "plpgsql", "deparse"].includes(value.operation)) {
    throw new Error(`unsupported operation: ${String(value.operation)}`);
  }
  if (typeof value.sql !== "string") throw new Error("sql must be a string");
  if (Buffer.byteLength(value.sql, "utf8") > MAX_SQL_BYTES)
    throw new Error(`sql exceeds ${MAX_SQL_BYTES} bytes`);
  return { operation: value.operation, sql: value.sql };
}

function stableValue(value, key = "") {
  if (Array.isArray(value)) {
    return value.map((item) => stableValue(item));
  }
  if (!isRecord(value)) {
    return value;
  }
  const normalized = {};
  for (const name of Object.keys(value).sort()) {
    if (["location", "stmt_location", "stmt_len"].includes(name)) {
      continue;
    }
    normalized[name] = stableValue(value[name], name);
  }
  return normalized;
}

function stableDigest(value) {
  return crypto
    .createHash("sha256")
    .update(JSON.stringify(stableValue(value)), "utf8")
    .digest("hex");
}

function statementKind(statement) {
  if (!isRecord(statement) || !isRecord(statement.stmt)) {
    return "Unknown";
  }
  return Object.keys(statement.stmt)[0] ?? "Unknown";
}

function stringNode(value) {
  return isRecord(value) && isRecord(value.String) && typeof value.String.sval === "string"
    ? value.String.sval
    : null;
}

function dottedName(value) {
  if (!Array.isArray(value)) {
    return null;
  }
  const parts = value.map(stringNode).filter((item) => item !== null);
  return parts.length > 0 ? parts.join(".") : null;
}

function relationName(value) {
  if (!isRecord(value) || typeof value.relname !== "string") {
    return null;
  }
  return typeof value.schemaname === "string"
    ? `${value.schemaname}.${value.relname}`
    : value.relname;
}

class FactBudget {
  constructor(limit) {
    this.limit = limit;
    this.returned = 0;
    this.truncated = false;
  }

  add(array, value) {
    if (this.returned >= this.limit) {
      this.truncated = true;
      return false;
    }
    array.push(value);
    this.returned += 1;
    return true;
  }
}

function locationOf(value, fallback = null) {
  if (isRecord(value) && Number.isSafeInteger(value.location) && value.location >= 0) {
    return value.location;
  }
  return fallback;
}

function declarationFact(nodeName, payload, statementIndex, statementLocation) {
  const identity = (() => {
    if (["CreateStmt", "AlterTableStmt"].includes(nodeName)) {
      return { kind: "table", name: relationName(payload.relation) };
    }
    if (nodeName === "ViewStmt") {
      return { kind: "view", name: relationName(payload.view) };
    }
    if (nodeName === "CreateTableAsStmt" && isRecord(payload.into)) {
      return { kind: "materialized_view", name: relationName(payload.into.rel) };
    }
    if (["CreateSeqStmt", "AlterSeqStmt"].includes(nodeName)) {
      return { kind: "sequence", name: relationName(payload.sequence) };
    }
    if (nodeName === "CreateSchemaStmt") {
      return {
        kind: "schema",
        name: typeof payload.schemaname === "string" ? payload.schemaname : null,
      };
    }
    if (nodeName === "CreateExtensionStmt") {
      return {
        kind: "extension",
        name: typeof payload.extname === "string" ? payload.extname : null,
      };
    }
    if (nodeName === "CreateEnumStmt") {
      return { kind: "enum", name: dottedName(payload.typeName) };
    }
    if (nodeName === "CreateDomainStmt") {
      return { kind: "domain", name: dottedName(payload.domainname) };
    }
    if (nodeName === "CreateFunctionStmt") {
      return {
        kind: payload.is_procedure ? "procedure" : "function",
        name: dottedName(payload.funcname),
      };
    }
    if (nodeName === "IndexStmt") {
      return {
        kind: "index",
        name:
          typeof payload.idxname === "string" ? payload.idxname : relationName(payload.relation),
      };
    }
    if (nodeName === "CreateTrigStmt") {
      return {
        kind: "trigger",
        name: typeof payload.trigname === "string" ? payload.trigname : null,
      };
    }
    if (nodeName === "CreatePolicyStmt") {
      return {
        kind: "policy",
        name: typeof payload.policy_name === "string" ? payload.policy_name : null,
      };
    }
    return null;
  })();
  if (identity === null || identity.name === null) {
    return null;
  }
  return {
    action: nodeName.startsWith("Alter") ? "alter" : "create",
    statement_index: statementIndex,
    statement_kind: nodeName,
    kind: identity.kind,
    name: identity.name,
    location: locationOf(payload, statementLocation),
  };
}

function extractFacts(tree, maxResults) {
  const budget = new FactBudget(maxResults);
  const statements = [];
  const declarations = [];
  const references = [];
  const calls = [];
  const wrappers = Array.isArray(tree?.stmts) ? tree.stmts : [];
  for (const [statementIndex, wrapper] of wrappers.entries()) {
    const kind = statementKind(wrapper);
    const statementLocation = Number.isSafeInteger(wrapper?.stmt_location)
      ? wrapper.stmt_location
      : 0;
    const statementLength = Number.isSafeInteger(wrapper?.stmt_len) ? wrapper.stmt_len : 0;
    budget.add(statements, {
      index: statementIndex,
      kind,
      byte_start: statementLocation,
      byte_end: statementLength > 0 ? statementLocation + statementLength : null,
    });
    const payload =
      isRecord(wrapper?.stmt) && isRecord(wrapper.stmt[kind]) ? wrapper.stmt[kind] : null;
    if (payload !== null) {
      const declaration = declarationFact(kind, payload, statementIndex, statementLocation);
      if (declaration !== null) {
        budget.add(declarations, declaration);
      }
    }
    const pending = payload === null ? [] : [payload];
    while (pending.length > 0) {
      const value = pending.pop();
      if (Array.isArray(value)) {
        pending.push(...value.toReversed());
        continue;
      }
      if (!isRecord(value)) {
        continue;
      }
      if (isRecord(value.RangeVar)) {
        const name = relationName(value.RangeVar);
        if (name !== null) {
          budget.add(references, {
            statement_index: statementIndex,
            kind: "relation",
            name,
            location: locationOf(value.RangeVar, statementLocation),
          });
        }
      }
      if (isRecord(value.ColumnRef)) {
        const name = dottedName(value.ColumnRef.fields);
        if (name !== null) {
          budget.add(references, {
            statement_index: statementIndex,
            kind: "column",
            name,
            location: locationOf(value.ColumnRef, statementLocation),
          });
        }
      }
      if (isRecord(value.TypeName)) {
        const name = dottedName(value.TypeName.names);
        if (name !== null) {
          budget.add(references, {
            statement_index: statementIndex,
            kind: "type",
            name,
            location: locationOf(value.TypeName, statementLocation),
          });
        }
      }
      if (isRecord(value.FuncCall)) {
        const name = dottedName(value.FuncCall.funcname);
        if (name !== null) {
          const call = {
            statement_index: statementIndex,
            kind: "function",
            name,
            argument_count: Array.isArray(value.FuncCall.args) ? value.FuncCall.args.length : 0,
            location: locationOf(value.FuncCall, statementLocation),
          };
          budget.add(calls, call);
          budget.add(references, {
            statement_index: statementIndex,
            kind: "routine",
            name,
            location: call.location,
          });
        }
      }
      pending.push(...Object.values(value).toReversed());
    }
  }
  return {
    statements,
    declarations,
    references,
    calls,
    returned: budget.returned,
    truncated: budget.truncated,
    limit: budget.limit,
  };
}

function parserDiagnostic(error) {
  const details =
    typeof parser.hasSqlDetails === "function" && parser.hasSqlDetails(error)
      ? error.sqlDetails
      : null;
  return {
    severity: "error",
    message: boundedMessage(error),
    cursor_position:
      details && Number.isSafeInteger(details.cursorPosition) ? details.cursorPosition : null,
    cursor_unit: "unicode_code_point",
  };
}

function analyzeSql(sql, mode, maxResults) {
  const result = {
    parser_version: PARSER_VERSION,
    postgres_major: POSTGRESQL_MAJOR,
    mode,
    tree: null,
    tokens: [],
    fingerprint: null,
    normalized: null,
    plpgsql: null,
    statements: [],
    declarations: [],
    references: [],
    calls: [],
    diagnostics: [],
    returned: 0,
    truncated: false,
    limit: maxResults,
  };
  try {
    if (mode === "all" || mode === "scan") {
      result.tokens = parser.scanSync(sql).tokens;
    }
    if (mode === "all" || mode === "fingerprint") {
      result.fingerprint = parser.fingerprintSync(sql);
      result.normalized = parser.normalizeSync(sql);
    }
    if (mode === "all" || mode === "plpgsql") {
      try {
        result.plpgsql = parser.parsePlPgSQLSync(sql);
      } catch (error) {
        result.diagnostics.push({ ...parserDiagnostic(error), origin: "plpgsql" });
      }
    }
    if (mode === "all" || mode === "parse") {
      result.tree = parser.parseSync(sql);
      const facts = extractFacts(result.tree, maxResults);
      Object.assign(result, facts);
    }
  } catch (error) {
    result.diagnostics.push({ ...parserDiagnostic(error), origin: "parser" });
  }
  return result;
}

function deparseSql(sql) {
  const originalTree = parser.parseSync(sql);
  const deparsedSql = deparseSync(JSON.parse(JSON.stringify(originalTree)));
  const reparsedTree = parser.parseSync(deparsedSql);
  const originalDigest = stableDigest(originalTree);
  const reparsedDigest = stableDigest(reparsedTree);
  return {
    parser_version: PARSER_VERSION,
    postgres_major: POSTGRESQL_MAJOR,
    original_sql: sql,
    deparsed_sql: deparsedSql,
    equivalent: originalDigest === reparsedDigest,
    original_tree_digest: originalDigest,
    reparsed_tree_digest: reparsedDigest,
    diagnostics:
      originalDigest === reparsedDigest
        ? []
        : [
            {
              severity: "warning",
              origin: "deparser",
              message: "deparsed SQL did not reparse to an equivalent normalized tree",
              cursor_position: null,
              cursor_unit: "unicode_code_point",
            },
          ],
  };
}

async function parseFiles(request) {
  const projectRoot = await realpath(request.projectRoot);
  if (!(await stat(projectRoot)).isDirectory()) {
    throw new Error("project_root must identify a directory");
  }
  const files = [];
  let total = 0;
  const seen = new Set();
  for (const raw of request.files) {
    if (isAbsolute(raw)) {
      throw new Error(`file path must be relative: ${raw}`);
    }
    const path = await realpath(resolve(projectRoot, raw));
    if (!isWithin(path, projectRoot)) {
      throw new Error(`file resolves outside project_root: ${raw}`);
    }
    if (seen.has(path)) {
      continue;
    }
    seen.add(path);
    if (parse(path).ext.toLowerCase() !== ".sql") {
      throw new Error(`file must use the .sql extension: ${raw}`);
    }
    const metadata = await stat(path);
    if (!metadata.isFile() || metadata.size > MAX_FILE_BYTES) {
      throw new Error(`file is not regular or exceeds ${MAX_FILE_BYTES} bytes: ${raw}`);
    }
    total += metadata.size;
    if (total > MAX_TOTAL_FILE_BYTES) {
      throw new Error(`selected SQL files exceed ${MAX_TOTAL_FILE_BYTES} aggregate bytes`);
    }
    const sql = await readFile(path, "utf8");
    files.push({
      file: relative(projectRoot, path).split(sep).join("/"),
      ...analyzeSql(sql, request.mode, request.maxResults),
    });
  }
  return { files, returned: files.length, truncated: false, limit: request.files.length };
}

function execute(request) {
  if (request.operation === "deparse") return deparseSql(request.sql);
  return analyzeSql(request.sql, request.operation, MAX_FACTS);
}

const argumentsList = process.argv.slice(2);
if (argumentsList.length === 1 && argumentsList[0] === "--version-json") {
  process.stdout.write(`${JSON.stringify(versions)}\n`);
  process.exit(0);
}
if (argumentsList.length === 1 && argumentsList[0] === "--version") {
  process.stdout.write(
    `ast-soleaux-postgresql ${versions.worker} (parser ${versions.parser}, deparser ${versions.deparser}, PostgreSQL ${versions.postgres_major})\n`,
  );
  process.exit(0);
}
if (argumentsList.length === 1 && argumentsList[0] === "--help") {
  process.stdout.write("Usage: ast-soleaux-postgresql [--version | --version-json | --help]\n");
  process.exit(0);
}
if (argumentsList.length !== 0) {
  writeFailure(new RequestFailure(`unknown arguments: ${argumentsList.join(" ")}`));
} else {
  try {
    const raw = readFileSync(0);
    if (raw.length > MAX_SQL_BYTES + 4096) {
      throw new RequestFailure(`input exceeds ${MAX_SQL_BYTES + 4096} bytes`);
    }
    let decoded;
    try {
      decoded = JSON.parse(raw.toString("utf8"));
    } catch {
      throw new RequestFailure("request must be valid JSON");
    }
    let request;
    try {
      request = validateRequest(decoded);
    } catch (error) {
      throw new RequestFailure(boundedMessage(error));
    }
    const response = {
      worker_version: versions.worker,
      parser_version: versions.parser,
      deparser_version: versions.deparser,
      postgres_major: versions.postgres_major,
      operation: request.operation,
      source_digest: crypto.createHash("sha256").update(request.sql, "utf8").digest("hex"),
      result: execute(request),
    };
    const output = Buffer.from(`${JSON.stringify(response)}\n`, "utf8");
    if (output.length > MAX_RESPONSE_BYTES)
      throw new Error("PostgreSQL worker response exceeds its output limit");
    process.stdout.write(output);
  } catch (error) {
    writeFailure(error);
  }
}
