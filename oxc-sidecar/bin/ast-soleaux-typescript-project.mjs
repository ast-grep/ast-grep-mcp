#!/usr/bin/env node

import crypto from "node:crypto";
import { existsSync, readFileSync, realpathSync, statSync } from "node:fs";
import { dirname, isAbsolute, parse, relative, resolve, sep } from "node:path";
import { createInterface } from "node:readline";
import ts from "typescript";

const WORKER_VERSION = "0.1.0";
const TYPESCRIPT_VERSION = "6.0.2";
const MAX_INPUT_BYTES = 1024 * 1024;
const MAX_OUTPUT_BYTES = 32 * 1024 * 1024;
const MAX_FILE_BYTES = 2 * 1024 * 1024;
const MAX_DEPENDENCY_FILE_BYTES = 4 * 1024 * 1024;
const MAX_TOTAL_SOURCE_BYTES = 16 * 1024 * 1024;
const MAX_SOURCE_FILES = 2048;
const MAX_ROOT_PATHS = 256;
const MAX_FACTS = 10_000;
const MAX_CACHE_ENTRIES = 32;
const SOURCE_EXTENSIONS = new Set([".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"]);
const projectCache = new Map();

if (ts.version !== TYPESCRIPT_VERSION) {
  throw new Error(`TypeScript worker requires ${TYPESCRIPT_VERSION}, found ${ts.version}`);
}

function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function rejectUnknownKeys(value, allowed, label) {
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) throw new Error(`${label} contains an unknown key: ${key}`);
  }
}

function isWithin(path, root) {
  const value = relative(root, path);
  return value === "" || (value !== ".." && !value.startsWith(`..${sep}`) && !isAbsolute(value));
}

function repositoryPath(path, root) {
  const normalized = existsSync(path) ? realpathSync(path) : resolve(path);
  return isWithin(normalized, root) ? relative(root, normalized).split(sep).join("/") : null;
}

function checkedRelativeFile(root, value, label) {
  if (
    typeof value !== "string" ||
    value.length === 0 ||
    value.includes("\0") ||
    isAbsolute(value)
  ) {
    throw new Error(`${label} must be a non-empty NUL-free relative path`);
  }
  const path = realpathSync(resolve(root, value));
  if (!isWithin(path, root)) throw new Error(`${label} resolves outside project_root: ${value}`);
  if (!statSync(path).isFile()) throw new Error(`${label} is not a regular file: ${value}`);
  return path;
}

function validateRequest(value) {
  if (!isRecord(value)) throw new Error("request must be a JSON object");
  rejectUnknownKeys(
    value,
    new Set([
      "project_root",
      "tsconfig",
      "paths",
      "include_emit",
      "include_code_actions",
      "max_results",
    ]),
    "request",
  );
  if (typeof value.project_root !== "string" || !isAbsolute(value.project_root)) {
    throw new Error("project_root must be an absolute path");
  }
  if (value.tsconfig !== undefined && typeof value.tsconfig !== "string")
    throw new Error("tsconfig must be a string");
  if (value.paths !== undefined && value.paths !== null) {
    if (
      !Array.isArray(value.paths) ||
      value.paths.length === 0 ||
      value.paths.length > MAX_ROOT_PATHS
    ) {
      throw new Error(`paths must contain between 1 and ${MAX_ROOT_PATHS} relative paths`);
    }
    if (!value.paths.every((path) => typeof path === "string"))
      throw new Error("paths must contain strings");
  }
  for (const key of ["include_emit", "include_code_actions"]) {
    if (value[key] !== undefined && typeof value[key] !== "boolean")
      throw new Error(`${key} must be a boolean`);
  }
  if (
    value.max_results !== undefined &&
    (!Number.isSafeInteger(value.max_results) ||
      value.max_results < 1 ||
      value.max_results > MAX_FACTS)
  ) {
    throw new Error(`max_results must be between 1 and ${MAX_FACTS}`);
  }
  return {
    projectRoot: value.project_root,
    tsconfig: value.tsconfig ?? "tsconfig.json",
    paths: value.paths ?? null,
    includeEmit: value.include_emit ?? true,
    includeCodeActions: value.include_code_actions ?? true,
    maxResults: value.max_results ?? 5000,
  };
}

function safeReadFile(path, projectRoot, typescriptRoot) {
  try {
    const resolved = realpathSync(path);
    if (!isWithin(resolved, projectRoot) && !isWithin(resolved, typescriptRoot)) return undefined;
    const metadata = statSync(resolved);
    const dependencyRoot = resolve(projectRoot, "node_modules");
    const limit = isWithin(resolved, typescriptRoot)
      ? MAX_TOTAL_SOURCE_BYTES
      : isWithin(resolved, dependencyRoot)
        ? MAX_DEPENDENCY_FILE_BYTES
        : MAX_FILE_BYTES;
    if (!metadata.isFile() || metadata.size > limit) return undefined;
    return readFileSync(resolved, "utf8");
  } catch {
    return undefined;
  }
}

function safeFileExists(path, projectRoot, typescriptRoot) {
  try {
    const resolved = realpathSync(path);
    return (
      (isWithin(resolved, projectRoot) || isWithin(resolved, typescriptRoot)) &&
      statSync(resolved).isFile()
    );
  } catch {
    return false;
  }
}

function safeDirectoryExists(path, projectRoot, typescriptRoot) {
  try {
    const resolved = realpathSync(path);
    return (
      (isWithin(resolved, projectRoot) || isWithin(resolved, typescriptRoot)) &&
      statSync(resolved).isDirectory()
    );
  } catch {
    return false;
  }
}

function diagnosticValue(diagnostic, projectRoot) {
  const file = diagnostic.file;
  const start = diagnostic.start ?? null;
  const length = diagnostic.length ?? null;
  const position = file && start !== null ? file.getLineAndCharacterOfPosition(start) : null;
  return {
    code: diagnostic.code,
    category: ts.DiagnosticCategory[diagnostic.category] ?? "Unknown",
    message: ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n"),
    file: file ? repositoryPath(file.fileName, projectRoot) : null,
    start,
    length,
    line: position?.line ?? null,
    column: position?.character ?? null,
    byte_start:
      file && start !== null ? Buffer.byteLength(file.text.slice(0, start), "utf8") : null,
    byte_end:
      file && start !== null
        ? Buffer.byteLength(file.text.slice(0, start + (length ?? 0)), "utf8")
        : null,
  };
}

function declarationName(node) {
  if (node.name && ts.isIdentifier(node.name)) return node.name;
  if (ts.isVariableStatement(node)) {
    const declaration = node.declarationList.declarations[0];
    if (declaration && ts.isIdentifier(declaration.name)) return declaration.name;
  }
  return null;
}

function resolveModule(specifier, sourceFile, options, host, projectRoot) {
  const resolved = ts.resolveModuleName(
    specifier,
    sourceFile.fileName,
    options,
    host,
  ).resolvedModule;
  if (!resolved)
    return { resolution: "unresolved", resolved_path: null, extension: null, external: false };
  const resolvedPath = repositoryPath(resolved.resolvedFileName, projectRoot);
  return {
    resolution: resolvedPath === null ? "external" : "resolved",
    resolved_path: resolvedPath,
    extension: String(resolved.extension),
    external: resolved.isExternalLibraryImport ?? false,
  };
}

function moduleFacts(sourceFile, options, host, projectRoot, add) {
  const facts = [];
  const capture = (kind, specifier, node, typeOnly) => {
    const start = node.getStart(sourceFile);
    const end = node.getEnd();
    add(facts, {
      file: repositoryPath(sourceFile.fileName, projectRoot),
      kind,
      specifier,
      type_only: typeOnly,
      start,
      end,
      byte_start: Buffer.byteLength(sourceFile.text.slice(0, start), "utf8"),
      byte_end: Buffer.byteLength(sourceFile.text.slice(0, end), "utf8"),
      ...resolveModule(specifier, sourceFile, options, host, projectRoot),
    });
  };
  const visit = (node) => {
    if (ts.isImportDeclaration(node) && ts.isStringLiteralLike(node.moduleSpecifier)) {
      capture(
        "import",
        node.moduleSpecifier.text,
        node.moduleSpecifier,
        node.importClause?.isTypeOnly ?? false,
      );
    } else if (
      ts.isExportDeclaration(node) &&
      node.moduleSpecifier &&
      ts.isStringLiteralLike(node.moduleSpecifier)
    ) {
      capture("reexport", node.moduleSpecifier.text, node.moduleSpecifier, node.isTypeOnly);
    } else if (ts.isCallExpression(node)) {
      const [argument] = node.arguments;
      const dynamicImport = node.expression.kind === ts.SyntaxKind.ImportKeyword;
      const requireCall = ts.isIdentifier(node.expression) && node.expression.text === "require";
      if ((dynamicImport || requireCall) && argument && ts.isStringLiteralLike(argument)) {
        capture(dynamicImport ? "dynamic_import" : "require", argument.text, argument, false);
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(sourceFile);
  return facts;
}

function normalizeChanges(changes, projectRoot) {
  const result = [];
  for (const change of changes) {
    const path = repositoryPath(change.fileName, projectRoot);
    if (path === null) continue;
    result.push({
      file: path,
      is_new_file: change.isNewFile ?? false,
      changes: change.textChanges.map((item) => ({
        start: item.span.start,
        length: item.span.length,
        new_text: item.newText,
      })),
    });
  }
  return result;
}

function inspect(rawRequest) {
  const request = validateRequest(rawRequest);
  const projectRoot = realpathSync(request.projectRoot);
  if (!statSync(projectRoot).isDirectory())
    throw new Error("project_root must identify a directory");
  const typescriptRoot = realpathSync(
    dirname(new URL(import.meta.resolve("typescript/package.json")).pathname),
  );
  const configPath = checkedRelativeFile(projectRoot, request.tsconfig, "tsconfig");
  const readConfig = ts.readConfigFile(configPath, (path) =>
    safeReadFile(path, projectRoot, typescriptRoot),
  );
  if (readConfig.error)
    throw new Error(ts.flattenDiagnosticMessageText(readConfig.error.messageText, "\n"));
  const parseHost = {
    useCaseSensitiveFileNames: ts.sys.useCaseSensitiveFileNames,
    fileExists: (path) => safeFileExists(path, projectRoot, typescriptRoot),
    readFile: (path) => safeReadFile(path, projectRoot, typescriptRoot),
    readDirectory: (root, extensions, excludes, includes, depth) =>
      isWithin(resolve(root), projectRoot)
        ? ts.sys.readDirectory(root, extensions, excludes, includes, depth)
        : [],
    trace: () => undefined,
  };
  const parsed = ts.parseJsonConfigFileContent(
    readConfig.config,
    parseHost,
    dirname(configPath),
    {},
    configPath,
  );
  const roots = request.paths
    ? request.paths.map((path) => checkedRelativeFile(projectRoot, path, "path"))
    : parsed.fileNames;
  const uniqueRoots = [...new Set(roots.map((path) => realpathSync(path)))];
  if (uniqueRoots.length === 0 || uniqueRoots.length > MAX_SOURCE_FILES) {
    throw new Error(`project roots must contain between 1 and ${MAX_SOURCE_FILES} files`);
  }
  const options = { ...parsed.options, noEmit: !request.includeEmit, noEmitOnError: false };
  const baseHost = ts.createCompilerHost(options, true);
  const host = {
    ...baseHost,
    fileExists: (path) => safeFileExists(path, projectRoot, typescriptRoot),
    readFile: (path) => safeReadFile(path, projectRoot, typescriptRoot),
    directoryExists: (path) => safeDirectoryExists(path, projectRoot, typescriptRoot),
    getSourceFile: (path, languageVersion, onError) => {
      const text = safeReadFile(path, projectRoot, typescriptRoot);
      if (text === undefined) {
        onError?.(`read denied or unavailable: ${path}`);
        return undefined;
      }
      return ts.createSourceFile(path, text, languageVersion, true);
    },
  };
  const program = ts.createProgram({ rootNames: uniqueRoots, options, host });
  const checker = program.getTypeChecker();
  const sourceFiles = program.getSourceFiles().filter((sourceFile) => {
    const path = repositoryPath(sourceFile.fileName, projectRoot);
    return (
      path !== null &&
      !sourceFile.isDeclarationFile &&
      SOURCE_EXTENSIONS.has(parse(sourceFile.fileName).ext.toLowerCase())
    );
  });
  let totalBytes = 0;
  for (const sourceFile of sourceFiles) {
    const size = Buffer.byteLength(sourceFile.text, "utf8");
    if (size > MAX_FILE_BYTES)
      throw new Error(`source exceeds ${MAX_FILE_BYTES} bytes: ${sourceFile.fileName}`);
    totalBytes += size;
    if (totalBytes > MAX_TOTAL_SOURCE_BYTES)
      throw new Error(`sources exceed ${MAX_TOTAL_SOURCE_BYTES} aggregate bytes`);
  }

  let returned = 0;
  let truncated = false;
  const add = (array, value) => {
    if (returned >= request.maxResults) {
      truncated = true;
      return false;
    }
    array.push(value);
    returned += 1;
    return true;
  };
  const diagnostics = [];
  const modules = [];
  const symbols = [];
  const inferredTypes = [];
  const emitted = [];
  const codeActions = [];
  const allDiagnostics = [...parsed.errors, ...ts.getPreEmitDiagnostics(program)];
  for (const diagnostic of allDiagnostics)
    add(diagnostics, diagnosticValue(diagnostic, projectRoot));
  for (const sourceFile of sourceFiles) {
    modules.push({
      file: repositoryPath(sourceFile.fileName, projectRoot),
      imports: moduleFacts(sourceFile, options, host, projectRoot, add),
      exports: (() => {
        const moduleSymbol = checker.getSymbolAtLocation(sourceFile);
        return moduleSymbol
          ? checker
              .getExportsOfModule(moduleSymbol)
              .map((symbol) => ({ name: symbol.getName(), flags: symbol.flags }))
          : [];
      })(),
    });
    for (const statement of sourceFile.statements) {
      const name = declarationName(statement);
      if (!name) continue;
      const symbol = checker.getSymbolAtLocation(name);
      const type = checker.getTypeAtLocation(name);
      const record = {
        file: repositoryPath(sourceFile.fileName, projectRoot),
        name: name.text,
        start: name.getStart(sourceFile),
        end: name.getEnd(),
        symbol_flags: symbol?.flags ?? null,
        type: checker.typeToString(type, name, ts.TypeFormatFlags.NoTruncation),
        documentation: symbol
          ? ts.displayPartsToString(symbol.getDocumentationComment(checker)) || null
          : null,
      };
      add(symbols, record);
      add(inferredTypes, record);
    }
  }
  if (request.includeEmit) {
    program.emit(undefined, (fileName, data, _bom, _error, sources) => {
      const path = repositoryPath(fileName, projectRoot);
      if (path === null || Buffer.byteLength(data, "utf8") > MAX_FILE_BYTES) return;
      add(emitted, {
        file: path,
        source_files: (sources ?? [])
          .map((source) => repositoryPath(source.fileName, projectRoot))
          .filter((item) => item !== null),
        code: data,
      });
    });
  }
  if (request.includeCodeActions && !truncated) {
    const languageHost = {
      getCompilationSettings: () => options,
      getScriptFileNames: () => sourceFiles.map((sourceFile) => sourceFile.fileName),
      getScriptVersion: () => "0",
      getScriptSnapshot: (path) => {
        const text = safeReadFile(path, projectRoot, typescriptRoot);
        return text === undefined ? undefined : ts.ScriptSnapshot.fromString(text);
      },
      getCurrentDirectory: () => projectRoot,
      getDefaultLibFileName: (compilerOptions) => ts.getDefaultLibFilePath(compilerOptions),
      fileExists: (path) => safeFileExists(path, projectRoot, typescriptRoot),
      readFile: (path) => safeReadFile(path, projectRoot, typescriptRoot),
      readDirectory: parseHost.readDirectory,
      directoryExists: (path) => safeDirectoryExists(path, projectRoot, typescriptRoot),
    };
    const languageService = ts.createLanguageService(languageHost, ts.createDocumentRegistry());
    try {
      for (const sourceFile of sourceFiles) {
        const changes = languageService.organizeImports(
          { type: "file", fileName: sourceFile.fileName, mode: ts.OrganizeImportsMode.All },
          {},
          {},
        );
        if (changes.length > 0)
          add(codeActions, {
            kind: "organize_imports",
            description: "Organize imports",
            changes: normalizeChanges(changes, projectRoot),
          });
      }
      for (const diagnostic of allDiagnostics) {
        if (!diagnostic.file || diagnostic.start === undefined) continue;
        for (const fix of languageService.getCodeFixesAtPosition(
          diagnostic.file.fileName,
          diagnostic.start,
          diagnostic.start + (diagnostic.length ?? 0),
          [diagnostic.code],
          {},
          {},
        )) {
          add(codeActions, {
            kind: "code_fix",
            description: fix.description,
            fix_name: fix.fixName,
            changes: normalizeChanges(fix.changes, projectRoot),
          });
        }
      }
    } finally {
      languageService.dispose();
    }
  }

  const digest = crypto.createHash("sha256");
  for (const sourceFile of [...sourceFiles].sort((left, right) =>
    left.fileName.localeCompare(right.fileName),
  )) {
    digest.update(repositoryPath(sourceFile.fileName, projectRoot) ?? "");
    digest.update("\0");
    digest.update(sourceFile.text);
    digest.update("\0");
  }
  return {
    typescript_version: ts.version,
    tsconfig: repositoryPath(configPath, projectRoot),
    root_files: uniqueRoots
      .map((file) => repositoryPath(file, projectRoot))
      .filter((item) => item !== null),
    options,
    diagnostics,
    modules,
    symbols,
    inferred_types: inferredTypes,
    emit: emitted,
    code_actions: codeActions,
    source_digest: digest.digest("hex"),
    returned,
    truncated,
    limit: request.maxResults,
  };
}

function requestIdentity(rawRequest) {
  const request = validateRequest(rawRequest);
  const projectRoot = realpathSync(request.projectRoot);
  const configPath = checkedRelativeFile(projectRoot, request.tsconfig, "tsconfig");
  const candidates = request.paths
    ? request.paths.map((path) => checkedRelativeFile(projectRoot, path, "path"))
    : ts.sys.readDirectory(projectRoot, [...SOURCE_EXTENSIONS], ["node_modules", "dist"], ["**/*"]);
  const files = [...new Set(candidates.map((path) => realpathSync(path)))].sort();
  if (files.length === 0 || files.length > MAX_SOURCE_FILES) {
    throw new Error(`cache identity requires between 1 and ${MAX_SOURCE_FILES} source files`);
  }
  const digest = crypto.createHash("sha256");
  digest.update(
    JSON.stringify({
      tsconfig: request.tsconfig,
      paths: request.paths,
      includeEmit: request.includeEmit,
      includeCodeActions: request.includeCodeActions,
      maxResults: request.maxResults,
    }),
  );
  let total = 0;
  for (const path of [configPath, ...files]) {
    const content = readFileSync(path);
    if (content.length > MAX_FILE_BYTES && path !== configPath)
      throw new Error(`cache source exceeds ${MAX_FILE_BYTES} bytes: ${path}`);
    total += content.length;
    if (total > MAX_TOTAL_SOURCE_BYTES + MAX_INPUT_BYTES)
      throw new Error("cache sources exceed aggregate byte limit");
    digest.update(repositoryPath(path, projectRoot) ?? path);
    digest.update("\0");
    digest.update(content);
    digest.update("\0");
  }
  return digest.digest("hex");
}

function cachedInspect(rawRequest) {
  const key = requestIdentity(rawRequest);
  const cached = projectCache.get(key);
  if (cached !== undefined) {
    projectCache.delete(key);
    projectCache.set(key, cached);
    return { ...cached, cache_hit: true };
  }
  const result = { ...inspect(rawRequest), cache_hit: false };
  projectCache.set(key, result);
  while (projectCache.size > MAX_CACHE_ENTRIES)
    projectCache.delete(projectCache.keys().next().value);
  return result;
}

function executeInput(raw) {
  if (Buffer.byteLength(raw, "utf8") > MAX_INPUT_BYTES)
    throw new Error(`input exceeds ${MAX_INPUT_BYTES} bytes`);
  const output = JSON.stringify(cachedInspect(JSON.parse(raw)));
  if (Buffer.byteLength(output, "utf8") > MAX_OUTPUT_BYTES)
    throw new Error(`output exceeds ${MAX_OUTPUT_BYTES} bytes`);
  return output;
}

if (process.argv[2] === "--version-json") {
  process.stdout.write(`${JSON.stringify({ worker: WORKER_VERSION, typescript: ts.version })}\n`);
} else if (process.argv[2] === "--version") {
  process.stdout.write(
    `ast-soleaux-typescript-project ${WORKER_VERSION} (typescript ${ts.version})\n`,
  );
} else if (process.argv[2] === "--serve") {
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lines) {
    if (line.length === 0) continue;
    try {
      process.stdout.write(`${executeInput(line)}\n`);
    } catch (error) {
      process.stdout.write(
        `${JSON.stringify({ error: error instanceof Error ? error.message : String(error) })}\n`,
      );
    }
  }
} else {
  if (process.argv.length > 2)
    throw new Error(`unknown arguments: ${process.argv.slice(2).join(" ")}`);
  process.stdout.write(`${executeInput(readFileSync(0).toString("utf8"))}\n`);
}
