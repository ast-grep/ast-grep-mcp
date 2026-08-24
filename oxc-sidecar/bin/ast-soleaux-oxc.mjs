#!/usr/bin/env node

import crypto from "node:crypto";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import { readFile, realpath, stat } from "node:fs/promises";
import { createRequire, isBuiltin } from "node:module";
import { dirname, isAbsolute, join, parse, relative, resolve, sep } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { minifySync } from "oxc-minify";
import { parseSync } from "oxc-parser";
import { ResolverFactory } from "oxc-resolver";
import { transformSync } from "oxc-transform";
import ts from "typescript";

const MAX_INPUT_BYTES = 1024 * 1024;
const MAX_FILES = 64;
const MAX_FILE_BYTES = 2 * 1024 * 1024;
const MAX_TOTAL_SOURCE_BYTES = 16 * 1024 * 1024;
const SOURCE_EXTENSIONS = new Set([".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"]);
const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const helperPackage = JSON.parse(readFileSync(join(packageRoot, "package.json"), "utf8"));
const require = createRequire(import.meta.url);

function dependencyVersion(name) {
  let current = dirname(require.resolve(name));
  const filesystemRoot = parse(current).root;
  while (current !== filesystemRoot) {
    const candidate = join(current, "package.json");
    if (existsSync(candidate)) {
      const manifest = JSON.parse(readFileSync(candidate, "utf8"));
      if (manifest.name === name && typeof manifest.version === "string") {
        return manifest.version;
      }
    }
    current = dirname(current);
  }
  throw new Error(`Could not determine the installed ${name} version`);
}

const versions = Object.freeze({
  helper: helperPackage.version,
  parser: dependencyVersion("oxc-parser"),
  resolver: dependencyVersion("oxc-resolver"),
});
const resolverFactory = ResolverFactory.default();
const projectResolvers = new Map();
const moduleCache = new Map();
const MODULE_CACHE_LIMIT = 64;
const RESOLVER_CACHE_LIMIT = 32;
const MODULE_GRAPH_VERSION = 1;

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

function isWithin(path, root) {
  const pathFromRoot = relative(root, path);
  return (
    pathFromRoot === "" ||
    (pathFromRoot !== ".." && !pathFromRoot.startsWith(`..${sep}`) && !isAbsolute(pathFromRoot))
  );
}

function boundedMessage(value) {
  const text = value instanceof Error ? value.message : String(value);
  return text.length <= 2000 ? text : `${text.slice(0, 2000)}…`;
}

async function readInput() {
  const chunks = [];
  let total = 0;
  for await (const chunk of process.stdin) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    total += buffer.length;
    if (total > MAX_INPUT_BYTES) {
      throw new Error(`Input exceeds the ${MAX_INPUT_BYTES}-byte limit`);
    }
    chunks.push(buffer);
  }
  if (total === 0) {
    throw new Error("Expected a JSON request on stdin");
  }
  try {
    const request = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    if (!isRecord(request)) {
      throw new Error("Input must be a JSON object");
    }
    return request;
  } catch (error) {
    throw new Error(`Input is not valid JSON: ${boundedMessage(error)}`);
  }
}

function parseModuleRequest(request) {
  rejectUnknownKeys(
    request,
    new Set(["operation", "project_root", "files", "include_dynamic"]),
    "Input",
  );
  if (typeof request.project_root !== "string" || !isAbsolute(request.project_root)) {
    throw new Error("project_root must be an absolute path");
  }
  if (
    !Array.isArray(request.files) ||
    request.files.length === 0 ||
    request.files.length > MAX_FILES
  ) {
    throw new Error(`files must contain between 1 and ${MAX_FILES} relative paths`);
  }
  if (
    !request.files.every(
      (file) => typeof file === "string" && file.length > 0 && !file.includes("\0"),
    )
  ) {
    throw new Error("files must contain non-empty strings without NUL characters");
  }
  if (request.include_dynamic !== undefined && typeof request.include_dynamic !== "boolean") {
    throw new Error("include_dynamic must be a boolean");
  }
  return {
    projectRoot: request.project_root,
    files: request.files,
    includeDynamic: request.include_dynamic ?? false,
  };
}

function parseComputeRequest(request, operation, allowedOptions) {
  rejectUnknownKeys(request, new Set(["operation", "filename", "code", "options"]), "Input");
  if (request.operation !== operation) {
    throw new Error(`operation must be ${operation}`);
  }
  if (
    typeof request.filename !== "string" ||
    request.filename.length === 0 ||
    request.filename.includes("\0")
  ) {
    throw new Error("filename must be a non-empty string without NUL characters");
  }
  if (typeof request.code !== "string" || Buffer.byteLength(request.code) > MAX_INPUT_BYTES) {
    throw new Error(`code must be a string no larger than ${MAX_INPUT_BYTES} bytes`);
  }
  const options = request.options ?? {};
  if (!isRecord(options)) {
    throw new Error("options must be an object");
  }
  rejectUnknownKeys(options, allowedOptions, "options");
  return {
    filename: request.filename,
    code: request.code,
    options,
  };
}

function normalizeDiagnostic(error) {
  const labels = Array.isArray(error?.labels)
    ? error.labels.map((label) => ({
        message: typeof label?.message === "string" ? label.message : null,
        start: Number.isInteger(label?.start) ? label.start : 0,
        end: Number.isInteger(label?.end) ? label.end : 0,
      }))
    : [];
  return {
    severity: typeof error?.severity === "string" ? error.severity : "Error",
    message: typeof error?.message === "string" ? error.message : boundedMessage(error),
    help: typeof error?.helpMessage === "string" ? error.helpMessage : null,
    codeframe: typeof error?.codeframe === "string" ? error.codeframe : null,
    labels,
  };
}

function unresolvedResolution(message) {
  return {
    resolution: "unresolved",
    resolved_path: null,
    package_json_path: null,
    module_type: null,
    resolution_error: message,
  };
}

function resolverForProject(projectRoot, mode) {
  const key = `${projectRoot}\0${mode}`;
  const cached = projectResolvers.get(key);
  if (cached !== undefined) {
    projectResolvers.delete(key);
    projectResolvers.set(key, cached);
    return cached;
  }
  const resolver = resolverFactory.cloneWithOptions({
    tsconfig: "auto",
    extensions: [".tsx", ".ts", ".mts", ".cts", ".jsx", ".js", ".mjs", ".cjs", ".json", ".node"],
    extensionAlias: {
      ".js": [".ts", ".tsx", ".js", ".jsx"],
      ".mjs": [".mts", ".mjs"],
      ".cjs": [".cts", ".cjs"],
    },
    conditionNames:
      mode === "require"
        ? ["types", "require", "node", "default"]
        : ["types", "import", "node", "default"],
    mainFields: ["types", "module", "main"],
    roots: [projectRoot],
    nodePath: false,
    builtinModules: true,
    moduleType: true,
  });
  projectResolvers.set(key, resolver);
  while (projectResolvers.size > RESOLVER_CACHE_LIMIT)
    projectResolvers.delete(projectResolvers.keys().next().value);
  return resolver;
}

function resolveSpecifier(resolver, importer, specifier, projectRoot) {
  if (isBuiltin(specifier)) {
    return {
      resolution: "external",
      resolved_path: null,
      package_json_path: null,
      module_type: null,
      resolution_error: null,
    };
  }
  let result;
  try {
    result = resolver.resolveFileSync(importer, specifier);
  } catch (error) {
    return unresolvedResolution(boundedMessage(error));
  }
  if (!isRecord(result)) {
    return unresolvedResolution("Resolver returned an invalid result");
  }
  if (result.builtin !== undefined) {
    return {
      resolution: "external",
      resolved_path: null,
      package_json_path: null,
      module_type: "builtin",
      resolution_error: null,
    };
  }
  if (typeof result.error === "string") {
    return unresolvedResolution(result.error);
  }
  if (typeof result.path !== "string") {
    return unresolvedResolution("Resolver did not return a path");
  }
  const resolvedPath = realpathSync(result.path);
  if (!isWithin(resolvedPath, projectRoot)) {
    return {
      resolution: "external",
      resolved_path: null,
      package_json_path: null,
      module_type: typeof result.moduleType === "string" ? result.moduleType : null,
      resolution_error: null,
    };
  }
  let packageJsonPath = null;
  if (typeof result.packageJsonPath === "string") {
    const resolvedPackageJson = realpathSync(result.packageJsonPath);
    if (isWithin(resolvedPackageJson, projectRoot)) {
      packageJsonPath = resolvedPackageJson;
    }
  }
  return {
    resolution: "resolved",
    resolved_path: resolvedPath,
    package_json_path: packageJsonPath,
    module_type: typeof result.moduleType === "string" ? result.moduleType : null,
    resolution_error: null,
  };
}

function staticEdge(kind, moduleRequest, resolver, importer, projectRoot, moduleSystem = "esm") {
  return {
    kind,
    module_system: moduleSystem,
    specifier: moduleRequest.value,
    expression: null,
    start: moduleRequest.start,
    end: moduleRequest.end,
    ...resolveSpecifier(resolver, importer, moduleRequest.value, projectRoot),
  };
}

function moduleEdges(moduleInfo, sourceText, resolver, importer, projectRoot, includeDynamic) {
  const edges = [];
  for (const imported of moduleInfo.staticImports) {
    edges.push(staticEdge("import", imported.moduleRequest, resolver, importer, projectRoot));
  }
  const reexports = new Set();
  for (const exported of moduleInfo.staticExports) {
    for (const entry of exported.entries) {
      const request = entry.moduleRequest;
      if (request === null) {
        continue;
      }
      const key = `${request.start}:${request.end}:${request.value}`;
      if (reexports.has(key)) {
        continue;
      }
      reexports.add(key);
      edges.push(staticEdge("reexport", request, resolver, importer, projectRoot));
    }
  }
  if (includeDynamic) {
    for (const imported of moduleInfo.dynamicImports) {
      const request = imported.moduleRequest;
      edges.push({
        kind: "dynamic",
        module_system: "esm",
        specifier: null,
        expression: sourceText.slice(request.start, request.end),
        start: request.start,
        end: request.end,
        resolution: "dynamic",
        resolved_path: null,
        package_json_path: null,
        module_type: null,
        resolution_error: null,
      });
    }
  }
  edges.sort(
    (left, right) =>
      left.start - right.start || left.end - right.end || left.kind.localeCompare(right.kind),
  );
  return edges;
}

function scriptKindFor(path) {
  const extension = parse(path).ext.toLowerCase();
  if (extension === ".ts" || extension === ".mts" || extension === ".cts") return ts.ScriptKind.TS;
  if (extension === ".tsx") return ts.ScriptKind.TSX;
  if (extension === ".jsx") return ts.ScriptKind.JSX;
  return ts.ScriptKind.JS;
}

function commonJsFacts(sourceText, importer, projectRoot, resolver) {
  const sourceFile = ts.createSourceFile(
    importer,
    sourceText,
    ts.ScriptTarget.Latest,
    true,
    scriptKindFor(importer),
  );
  const edges = [];
  const exports = [];
  const visit = (node) => {
    if (
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === "require" &&
      node.arguments.length === 1 &&
      ts.isStringLiteralLike(node.arguments[0])
    ) {
      const argument = node.arguments[0];
      edges.push(
        staticEdge(
          "import",
          {
            value: argument.text,
            start: argument.getStart(sourceFile) + 1,
            end: argument.getEnd() - 1,
          },
          resolver,
          importer,
          projectRoot,
          "commonjs",
        ),
      );
    }
    if (ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.EqualsToken) {
      const left = node.left;
      if (
        ts.isPropertyAccessExpression(left) &&
        ((ts.isIdentifier(left.expression) &&
          left.expression.text === "module" &&
          left.name.text === "exports") ||
          (ts.isIdentifier(left.expression) && left.expression.text === "exports") ||
          (ts.isPropertyAccessExpression(left.expression) &&
            ts.isIdentifier(left.expression.expression) &&
            left.expression.expression.text === "module" &&
            left.expression.name.text === "exports"))
      ) {
        exports.push({
          start: left.getStart(sourceFile),
          end: left.getEnd(),
          text: left.getText(sourceFile),
        });
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(sourceFile);
  return { edges, exports };
}

function packageMetadata(path, projectRoot) {
  let directory = dirname(path);
  while (isWithin(directory, projectRoot)) {
    const manifestPath = join(directory, "package.json");
    if (existsSync(manifestPath)) {
      try {
        const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
        return {
          path: relative(projectRoot, realpathSync(manifestPath)).split(sep).join("/"),
          name: typeof manifest.name === "string" ? manifest.name : null,
          type: typeof manifest.type === "string" ? manifest.type : null,
        };
      } catch {
        return null;
      }
    }
    const parent = dirname(directory);
    if (parent === directory) break;
    directory = parent;
  }
  return null;
}

async function inspectModules(request) {
  const projectRoot = await realpath(request.projectRoot);
  const projectMetadata = await stat(projectRoot);
  if (!projectMetadata.isDirectory()) {
    throw new Error("project_root must identify a directory");
  }
  const importResolver = resolverForProject(projectRoot, "import");
  const requireResolver = resolverForProject(projectRoot, "require");
  const modules = [];
  const seen = new Set();
  let totalSourceBytes = 0;
  for (const rawFile of request.files) {
    if (isAbsolute(rawFile)) {
      throw new Error(`File paths must be relative to project_root: ${rawFile}`);
    }
    const absoluteFile = await realpath(resolve(projectRoot, rawFile));
    if (!isWithin(absoluteFile, projectRoot)) {
      throw new Error(`File resolves outside project_root: ${rawFile}`);
    }
    if (seen.has(absoluteFile)) {
      continue;
    }
    seen.add(absoluteFile);
    const extension = parse(absoluteFile).ext.toLowerCase();
    if (!SOURCE_EXTENSIONS.has(extension)) {
      throw new Error(`Unsupported JavaScript or TypeScript extension: ${rawFile}`);
    }
    const metadata = await stat(absoluteFile);
    if (!metadata.isFile()) {
      throw new Error(`File is not regular: ${rawFile}`);
    }
    if (metadata.size > MAX_FILE_BYTES) {
      throw new Error(`File exceeds the ${MAX_FILE_BYTES}-byte limit: ${rawFile}`);
    }
    totalSourceBytes += metadata.size;
    if (totalSourceBytes > MAX_TOTAL_SOURCE_BYTES) {
      throw new Error(`Selected files exceed the ${MAX_TOTAL_SOURCE_BYTES}-byte aggregate limit`);
    }
    const sourceBuffer = await readFile(absoluteFile);
    let sourceText;
    try {
      sourceText = new TextDecoder("utf-8", { fatal: true }).decode(sourceBuffer);
    } catch (error) {
      throw new Error(`File is not valid UTF-8: ${rawFile}: ${boundedMessage(error)}`);
    }
    const parsed = parseSync(absoluteFile, sourceText, {
      sourceType: "unambiguous",
      range: true,
      showSemanticErrors: true,
    });
    const esmEdges = moduleEdges(
      parsed.module,
      sourceText,
      importResolver,
      absoluteFile,
      projectRoot,
      request.includeDynamic,
    );
    const commonjs = commonJsFacts(sourceText, absoluteFile, projectRoot, requireResolver);
    const edges = [...esmEdges, ...commonjs.edges].sort(
      (left, right) =>
        left.start - right.start || left.end - right.end || left.kind.localeCompare(right.kind),
    );
    modules.push({
      file: relative(projectRoot, absoluteFile).split(sep).join("/"),
      has_module_syntax: parsed.module.hasModuleSyntax,
      source_type: parsed.program.sourceType ?? null,
      package: packageMetadata(absoluteFile, projectRoot),
      commonjs_exports: commonjs.exports,
      import_meta_spans: parsed.module.importMetas.map((span) => ({
        start: span.start,
        end: span.end,
      })),
      edges,
      diagnostics: parsed.errors.map(normalizeDiagnostic),
    });
  }
  const graph = {
    version: MODULE_GRAPH_VERSION,
    nodes: modules.map((module) => ({
      file: module.file,
      source_type: module.source_type,
      package: module.package,
    })),
    edges: modules.flatMap((module) =>
      module.edges.map((edge) => ({
        importer: module.file,
        kind: edge.kind,
        module_system: edge.module_system,
        specifier: edge.specifier,
        expression: edge.expression,
        start: edge.start,
        end: edge.end,
        resolution: edge.resolution,
        target:
          typeof edge.resolved_path === "string" && isWithin(edge.resolved_path, projectRoot)
            ? relative(projectRoot, edge.resolved_path).split(sep).join("/")
            : null,
        module_type: edge.module_type,
        resolution_error: edge.resolution_error,
      })),
    ),
  };
  return { versions, graph_version: MODULE_GRAPH_VERSION, graph, modules, cache_hit: false };
}

async function inspectModulesCached(request) {
  const projectRoot = await realpath(request.projectRoot);
  const identities = [];
  const configurationPaths = new Set();
  for (const file of request.files) {
    const path = await realpath(resolve(projectRoot, file));
    const payload = await readFile(path);
    identities.push([path, crypto.createHash("sha256").update(payload).digest("hex")]);
    let directory = dirname(path);
    while (isWithin(directory, projectRoot)) {
      for (const name of ["tsconfig.json", "jsconfig.json", "package.json"]) {
        const candidate = join(directory, name);
        if (existsSync(candidate)) configurationPaths.add(await realpath(candidate));
      }
      if (directory === projectRoot) break;
      const parent = dirname(directory);
      if (parent === directory) break;
      directory = parent;
    }
  }
  const configurationIdentities = [];
  for (const path of [...configurationPaths].sort()) {
    const payload = await readFile(path);
    configurationIdentities.push([path, crypto.createHash("sha256").update(payload).digest("hex")]);
  }
  const key = JSON.stringify([
    projectRoot,
    identities,
    configurationIdentities,
    request.includeDynamic,
  ]);
  const cached = moduleCache.get(key);
  if (cached !== undefined) {
    moduleCache.delete(key);
    moduleCache.set(key, cached);
    return { ...cached, cache_hit: true };
  }
  for (const mode of ["import", "require"]) {
    const resolver = projectResolvers.get(`${projectRoot}\0${mode}`);
    if (resolver !== undefined) resolver.clearCache();
  }
  const result = await inspectModules(request);
  moduleCache.set(key, result);
  while (moduleCache.size > MODULE_CACHE_LIMIT) {
    moduleCache.delete(moduleCache.keys().next().value);
  }
  return result;
}

const TRANSFORM_OPTIONS = new Set([
  "lang",
  "sourceType",
  "cwd",
  "sourcemap",
  "target",
  "typescript",
  "jsx",
  "decorator",
  "assumptions",
  "define",
]);
const MINIFY_OPTIONS = new Set(["compress", "mangle", "sourcemap", "codegen"]);

function transformSource(request) {
  const parsed = parseComputeRequest(request, "transform", TRANSFORM_OPTIONS);
  const result = transformSync(parsed.filename, parsed.code, parsed.options);
  return {
    versions,
    operation: "transform",
    result: {
      code: result.code,
      map: result.map ?? null,
      declaration: result.declaration ?? null,
      declaration_map: result.declarationMap ?? null,
      helpers_used: result.helpersUsed,
      diagnostics: result.errors.map(normalizeDiagnostic),
    },
  };
}

function minifySource(request) {
  const parsed = parseComputeRequest(request, "minify", MINIFY_OPTIONS);
  const result = minifySync(parsed.filename, parsed.code, parsed.options);
  return {
    versions,
    operation: "minify",
    result: {
      code: result.code,
      map: result.map ?? null,
      legal_comments: result.legalComments,
      mangle_cache: result.mangleCache ?? null,
      diagnostics: result.errors.map(normalizeDiagnostic),
    },
  };
}

async function executeRequest(request) {
  const operation = request.operation ?? "modules";
  if (operation === "modules") return inspectModulesCached(parseModuleRequest(request));
  if (operation === "transform") return transformSource(request);
  if (operation === "minify") return minifySource(request);
  throw new Error(`Unsupported operation: ${operation}`);
}

async function main() {
  const argumentsList = process.argv.slice(2);
  if (argumentsList.length === 1 && argumentsList[0] === "--version") {
    process.stdout.write(
      `ast-soleaux-oxc ${versions.helper} (oxc-parser ${versions.parser}, oxc-resolver ${versions.resolver})\n`,
    );
    return;
  }
  if (argumentsList.length === 1 && argumentsList[0] === "--version-json") {
    process.stdout.write(`${JSON.stringify(versions)}\n`);
    return;
  }
  if (argumentsList.length === 1 && argumentsList[0] === "--help") {
    process.stdout.write(
      "Usage: ast-soleaux-oxc [--version | --version-json | --serve | --help]\n",
    );
    return;
  }
  if (argumentsList.length === 1 && argumentsList[0] === "--serve") {
    const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
    for await (const line of lines) {
      if (line.length === 0) continue;
      try {
        process.stdout.write(`${JSON.stringify(await executeRequest(JSON.parse(line)))}\n`);
      } catch (error) {
        process.stdout.write(`${JSON.stringify({ error: boundedMessage(error) })}\n`);
      }
    }
    return;
  }
  if (argumentsList.length !== 0) throw new Error(`Unknown arguments: ${argumentsList.join(" ")}`);
  process.stdout.write(`${JSON.stringify(await executeRequest(await readInput()))}\n`);
}

main().catch((error) => {
  process.stderr.write(`${boundedMessage(error)}\n`);
  process.exitCode = 1;
});
