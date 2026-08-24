use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, BufRead, Read, Write};
use std::path::{Component, Path};

use oxc_allocator::Allocator;
use oxc_cfg::graph::visit::EdgeRef;
use oxc_parser::Parser;
use oxc_semantic::{Semantic, SemanticBuilder};
use oxc_span::{GetSpan, SourceType};
use oxc_syntax::reference::ReferenceId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const WORKER_VERSION: &str = "0.1.0";
const OXC_VERSION: &str = "0.147.0";
const RESOLVER_VERSION: &str = "11.24.2";
const MAX_REQUEST_BYTES: u64 = 16 * 1024 * 1024 + 64 * 1024;
const MAX_SOURCE_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 32 * 1024 * 1024;
const CACHE_ENTRIES: usize = 64;
const CACHE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    operation: String,
    filename: String,
    source: String,
    #[serde(default)]
    position: Option<u32>,
    #[serde(default)]
    source_digest: Option<String>,
}

#[derive(Clone, Serialize)]
struct DiagnosticDto {
    message: String,
}

#[derive(Clone, Serialize)]
struct StatsDto {
    nodes: u32,
    scopes: u32,
    symbols: u32,
    references: u32,
}

#[derive(Clone, Copy, Serialize)]
struct SpanDto {
    start: u32,
    end: u32,
}

#[derive(Clone, Serialize)]
struct BindingDto {
    name: String,
    symbol_id: usize,
}

#[derive(Clone, Serialize)]
struct ScopeDto {
    scope_id: usize,
    parent_scope_id: Option<usize>,
    flags: String,
    node_id: usize,
    span: SpanDto,
    bindings: Vec<BindingDto>,
}

#[derive(Clone, Serialize)]
struct RedeclarationDto {
    span: SpanDto,
    node_id: usize,
}

#[derive(Clone, Serialize)]
struct SymbolDto {
    symbol_id: usize,
    name: String,
    span: SpanDto,
    flags: String,
    scope_id: usize,
    declaration_node_id: usize,
    reference_count: usize,
    redeclarations: Vec<RedeclarationDto>,
}

#[derive(Clone, Serialize)]
struct ReferenceDto {
    reference_id: usize,
    name: String,
    span: SpanDto,
    scope_id: usize,
    symbol_id: Option<usize>,
    flags: String,
    read: bool,
    write: bool,
    r#type: bool,
}

#[derive(Clone, Serialize)]
struct UnresolvedDto {
    name: String,
    reference_id: usize,
    span: SpanDto,
    scope_id: usize,
    flags: String,
}

#[derive(Clone, Serialize)]
struct CfgInstructionDto {
    kind: String,
    node_id: Option<usize>,
    span: Option<SpanDto>,
}

#[derive(Clone, Serialize)]
struct CfgBlockDto {
    block_id: usize,
    unreachable: bool,
    instructions: Vec<CfgInstructionDto>,
}

#[derive(Clone, Serialize)]
struct CfgEdgeDto {
    from: usize,
    to: usize,
    kind: String,
}

#[derive(Clone, Serialize)]
struct CfgDto {
    entry_block_id: Option<usize>,
    exit_block_ids: Vec<usize>,
    selected_block_id: Option<usize>,
    blocks: Vec<CfgBlockDto>,
    edges: Vec<CfgEdgeDto>,
}

#[derive(Clone, Serialize)]
struct TargetDto {
    kind: &'static str,
    name: String,
    span: SpanDto,
    symbol_id: Option<usize>,
    reference_id: Option<usize>,
}

#[derive(Clone, Serialize)]
struct CachedAnalysis {
    diagnostics: Vec<DiagnosticDto>,
    stats: StatsDto,
    scopes: Vec<ScopeDto>,
    symbols: Vec<SymbolDto>,
    references: Vec<ReferenceDto>,
    unresolved: Vec<UnresolvedDto>,
    cfg: Vec<CfgDto>,
}

#[derive(Serialize)]
struct Response {
    worker_version: &'static str,
    oxc_version: &'static str,
    operation: String,
    source_digest: String,
    cache_hit: bool,
    cache_entries: usize,
    diagnostics: Vec<DiagnosticDto>,
    stats: StatsDto,
    scopes: Vec<ScopeDto>,
    symbols: Vec<SymbolDto>,
    references: Vec<ReferenceDto>,
    unresolved: Vec<UnresolvedDto>,
    cfg: Vec<CfgDto>,
    target: Option<TargetDto>,
}

struct CacheEntry {
    analysis: CachedAnalysis,
    bytes: usize,
}

struct SemanticCache {
    entries: HashMap<String, CacheEntry>,
    order: VecDeque<String>,
    bytes: usize,
    max_entries: usize,
    max_bytes: usize,
}

impl SemanticCache {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            max_entries,
            max_bytes,
        }
    }

    fn get(&mut self, key: &str) -> Option<CachedAnalysis> {
        let analysis = self.entries.get(key)?.analysis.clone();
        self.order.retain(|item| item != key);
        self.order.push_back(key.to_string());
        Some(analysis)
    }

    fn insert(&mut self, key: String, analysis: CachedAnalysis) -> Result<(), serde_json::Error> {
        let bytes = serde_json::to_vec(&analysis)?.len();
        if bytes > self.max_bytes {
            return Ok(());
        }
        if let Some(previous) = self.entries.remove(&key) {
            self.bytes -= previous.bytes;
            self.order.retain(|item| item != &key);
        }
        self.bytes += bytes;
        self.order.push_back(key.clone());
        self.entries.insert(key, CacheEntry { analysis, bytes });
        while self.entries.len() > self.max_entries || self.bytes > self.max_bytes {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest) {
                self.bytes -= removed.bytes;
            }
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

fn span_dto(span: oxc_span::Span) -> SpanDto {
    SpanDto {
        start: span.start,
        end: span.end,
    }
}

fn semantic_payload(semantic: &Semantic<'_>) -> CachedAnalysis {
    let scoping = semantic.scoping();
    let mut scope_ids = vec![scoping.root_scope_id()];
    scope_ids.extend(scoping.scope_descendants_from_root());
    let mut seen_scopes = HashSet::new();
    scope_ids.retain(|scope_id| seen_scopes.insert(scope_id.index()));
    let scopes = scope_ids
        .into_iter()
        .map(|scope_id| {
            let node_id = scoping.get_node_id(scope_id);
            let span = semantic.nodes().get_node(node_id).kind().span();
            let bindings = scoping
                .get_bindings(scope_id)
                .iter()
                .map(|(name, symbol_id)| BindingDto {
                    name: name.to_string(),
                    symbol_id: symbol_id.index(),
                })
                .collect();
            ScopeDto {
                scope_id: scope_id.index(),
                parent_scope_id: scoping.scope_parent_id(scope_id).map(|id| id.index()),
                flags: format!("{:?}", scoping.scope_flags(scope_id)),
                node_id: node_id.index(),
                span: span_dto(span),
                bindings,
            }
        })
        .collect();

    let symbols = scoping
        .symbol_ids()
        .map(|symbol_id| SymbolDto {
            symbol_id: symbol_id.index(),
            name: scoping.symbol_name(symbol_id).to_string(),
            span: span_dto(scoping.symbol_span(symbol_id)),
            flags: format!("{:?}", scoping.symbol_flags(symbol_id)),
            scope_id: scoping.symbol_scope_id(symbol_id).index(),
            declaration_node_id: scoping.symbol_declaration(symbol_id).index(),
            reference_count: scoping.get_resolved_reference_ids(symbol_id).len(),
            redeclarations: scoping
                .symbol_redeclarations(symbol_id)
                .iter()
                .map(|item| RedeclarationDto {
                    span: span_dto(item.span),
                    node_id: item.declaration.index(),
                })
                .collect(),
        })
        .collect();

    let references = (0..scoping.references_len())
        .map(ReferenceId::from_usize)
        .map(|reference_id| {
            let reference = scoping.get_reference(reference_id);
            ReferenceDto {
                reference_id: reference_id.index(),
                name: semantic.reference_name(reference).to_string(),
                span: span_dto(semantic.reference_span(reference)),
                scope_id: reference.scope_id().index(),
                symbol_id: reference.symbol_id().map(|id| id.index()),
                flags: format!("{:?}", reference.flags()),
                read: reference.is_read(),
                write: reference.is_write(),
                r#type: reference.is_type(),
            }
        })
        .collect();

    let unresolved = scoping
        .root_unresolved_references()
        .iter()
        .flat_map(|(name, reference_ids)| {
            reference_ids.iter().map(move |reference_id| {
                let reference = scoping.get_reference(*reference_id);
                UnresolvedDto {
                    name: name.to_string(),
                    reference_id: reference_id.index(),
                    span: span_dto(semantic.reference_span(reference)),
                    scope_id: reference.scope_id().index(),
                    flags: format!("{:?}", reference.flags()),
                }
            })
        })
        .collect();

    let cfg = semantic
        .cfg()
        .map(|cfg| {
            let blocks = cfg
                .graph()
                .node_indices()
                .map(|node| {
                    let block = cfg.basic_block(node);
                    CfgBlockDto {
                        block_id: node.index(),
                        unreachable: block.is_unreachable(),
                        instructions: block
                            .instructions()
                            .iter()
                            .map(|instruction| CfgInstructionDto {
                                kind: format!("{:?}", instruction.kind),
                                node_id: instruction.node_id.map(|id| id.index()),
                                span: instruction.node_id.map(|id| {
                                    span_dto(semantic.nodes().get_node(id).kind().span())
                                }),
                            })
                            .collect(),
                    }
                })
                .collect();
            let graph = cfg.graph();
            let entry_block_id = graph.node_indices().next().map(|node| node.index());
            let exit_block_ids = graph
                .node_indices()
                .filter(|node| graph.edges(*node).next().is_none())
                .map(|node| node.index())
                .collect();
            let edges = graph
                .edge_references()
                .map(|edge| CfgEdgeDto {
                    from: edge.source().index(),
                    to: edge.target().index(),
                    kind: format!("{:?}", edge.weight()),
                })
                .collect();
            vec![CfgDto {
                entry_block_id,
                exit_block_ids,
                selected_block_id: None,
                blocks,
                edges,
            }]
        })
        .unwrap_or_default();

    let stats = semantic.stats();
    CachedAnalysis {
        diagnostics: Vec::new(),
        stats: StatsDto {
            nodes: stats.nodes,
            scopes: stats.scopes,
            symbols: stats.symbols,
            references: stats.references,
        },
        scopes,
        symbols,
        references,
        unresolved,
        cfg,
    }
}

fn validate_request(request: &Request) -> Result<SourceType, String> {
    if request.source.len() > MAX_SOURCE_BYTES {
        return Err(format!("source exceeds {MAX_SOURCE_BYTES} bytes"));
    }
    if request.filename.contains('\0') || Path::new(&request.filename).is_absolute() {
        return Err("filename must be a relative path without NUL bytes".to_string());
    }
    if Path::new(&request.filename).components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err("filename must remain inside the logical project".to_string());
    }
    if !matches!(
        request.operation.as_str(),
        "analyze" | "scopes" | "symbols" | "references" | "cfg"
    ) {
        return Err(format!("unsupported operation: {}", request.operation));
    }
    SourceType::from_path(Path::new(&request.filename)).map_err(|error| error.to_string())
}

fn build_analysis(request: &Request, source_type: SourceType) -> CachedAnalysis {
    let allocator = Allocator::default();
    let parsed = Parser::new(&allocator, &request.source, source_type).parse();
    let mut parse_diagnostics = parsed
        .diagnostics
        .iter()
        .map(|diagnostic| DiagnosticDto {
            message: diagnostic.to_string(),
        })
        .collect::<Vec<_>>();
    let program = allocator.alloc(parsed.program);
    let built = SemanticBuilder::new()
        .with_build_nodes(true)
        .with_cfg(true)
        .build(program);
    parse_diagnostics.extend(built.diagnostics.iter().map(|diagnostic| DiagnosticDto {
        message: diagnostic.to_string(),
    }));
    let mut analysis = semantic_payload(&built.semantic);
    analysis.diagnostics = parse_diagnostics;
    analysis
}

fn select_target(analysis: &CachedAnalysis, position: Option<u32>) -> Option<TargetDto> {
    let offset = position?;
    if let Some(reference) = analysis
        .references
        .iter()
        .find(|reference| reference.span.start <= offset && offset < reference.span.end)
    {
        return Some(TargetDto {
            kind: "reference",
            name: reference.name.clone(),
            span: reference.span,
            symbol_id: reference.symbol_id,
            reference_id: Some(reference.reference_id),
        });
    }
    analysis
        .symbols
        .iter()
        .find(|symbol| symbol.span.start <= offset && offset < symbol.span.end)
        .map(|symbol| TargetDto {
            kind: "symbol",
            name: symbol.name.clone(),
            span: symbol.span,
            symbol_id: Some(symbol.symbol_id),
            reference_id: None,
        })
}

fn process_request(
    request: Request,
    cache: &mut SemanticCache,
) -> Result<Response, Box<dyn std::error::Error>> {
    let source_type = validate_request(&request)?;
    let digest = hex::encode(Sha256::digest(request.source.as_bytes()));
    if request
        .source_digest
        .as_ref()
        .is_some_and(|expected| expected != &digest)
    {
        return Err("source digest does not match current source".into());
    }
    let cache_key = format!("{}:{:?}", digest, source_type);
    let (analysis, cache_hit) = match cache.get(&cache_key) {
        Some(value) => (value, true),
        None => {
            let value = build_analysis(&request, source_type);
            cache.insert(cache_key, value.clone())?;
            (value, false)
        }
    };
    let target = select_target(&analysis, request.position);
    let mut response = Response {
        worker_version: WORKER_VERSION,
        oxc_version: OXC_VERSION,
        operation: request.operation.clone(),
        source_digest: digest,
        cache_hit,
        cache_entries: cache.len(),
        diagnostics: analysis.diagnostics.clone(),
        stats: analysis.stats.clone(),
        scopes: analysis.scopes.clone(),
        symbols: analysis.symbols.clone(),
        references: analysis.references.clone(),
        unresolved: analysis.unresolved.clone(),
        cfg: analysis.cfg.clone(),
        target,
    };
    if request.operation == "cfg" {
        for graph in &mut response.cfg {
            graph.selected_block_id = request.position.and_then(|offset| {
                graph.blocks.iter().find_map(|block| {
                    block
                        .instructions
                        .iter()
                        .filter_map(|instruction| instruction.span)
                        .any(|span| span.start <= offset && offset < span.end)
                        .then_some(block.block_id)
                })
            });
        }
    }
    match request.operation.as_str() {
        "scopes" => {
            response.symbols.clear();
            response.references.clear();
            response.unresolved.clear();
            response.cfg.clear();
        }
        "symbols" => {
            response.scopes.clear();
            response.references.clear();
            response.unresolved.clear();
            response.cfg.clear();
        }
        "references" => {
            response.scopes.clear();
            response.cfg.clear();
            match response.target.as_ref() {
                Some(target) => {
                    if let Some(symbol_id) = target.symbol_id {
                        response
                            .references
                            .retain(|reference| reference.symbol_id == Some(symbol_id));
                        response
                            .symbols
                            .retain(|symbol| symbol.symbol_id == symbol_id);
                    } else if let Some(reference_id) = target.reference_id {
                        response
                            .references
                            .retain(|reference| reference.reference_id == reference_id);
                        response.symbols.clear();
                    }
                }
                None => {
                    response.symbols.clear();
                    response.references.clear();
                }
            }
        }
        "cfg" => {
            response.scopes.clear();
            response.symbols.clear();
            response.references.clear();
            response.unresolved.clear();
        }
        _ => {}
    }
    Ok(response)
}

fn encode_response(response: &Response) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut encoded = serde_json::to_vec(response)?;
    if encoded.len() > MAX_RESPONSE_BYTES {
        return Err(format!("response exceeds {MAX_RESPONSE_BYTES} bytes").into());
    }
    encoded.push(b'\n');
    Ok(encoded)
}

fn read_bounded_line(reader: &mut impl BufRead) -> io::Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Ok(Some(line))
            };
        }
        let consumed = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if line.len() as u64 + consumed as u64 > MAX_REQUEST_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("request exceeds {MAX_REQUEST_BYTES} bytes"),
            ));
        }
        line.extend_from_slice(&available[..consumed]);
        let complete = available[consumed - 1] == b'\n';
        reader.consume(consumed);
        if complete {
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            return Ok(Some(line));
        }
    }
}

fn serve() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut input = io::BufReader::new(stdin.lock());
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    let mut cache = SemanticCache::new(CACHE_ENTRIES, CACHE_BYTES);
    while let Some(line) = read_bounded_line(&mut input)? {
        if line.is_empty() {
            continue;
        }
        let response = match serde_json::from_slice::<Request>(&line)
            .map_err(|error| error.into())
            .and_then(|request| process_request(request, &mut cache))
        {
            Ok(response) => serde_json::to_vec(&response)?,
            Err(error) => serde_json::to_vec(&serde_json::json!({"error": error.to_string()}))?,
        };
        if response.len() > MAX_RESPONSE_BYTES {
            return Err(format!("response exceeds {MAX_RESPONSE_BYTES} bytes").into());
        }
        stdout.write_all(&response)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match std::env::args().nth(1).as_deref() {
        Some("--version-json") => {
            println!(
                "{{\"worker\":\"{WORKER_VERSION}\",\"oxc\":\"{OXC_VERSION}\",\"resolver\":\"{RESOLVER_VERSION}\"}}"
            );
            return Ok(());
        }
        Some("--serve") => return serve(),
        Some(argument) => return Err(format!("unknown argument: {argument}").into()),
        None => {}
    }
    let mut input = Vec::new();
    io::stdin()
        .take(MAX_REQUEST_BYTES + 1)
        .read_to_end(&mut input)?;
    if input.len() as u64 > MAX_REQUEST_BYTES {
        return Err(format!("request exceeds {MAX_REQUEST_BYTES} bytes").into());
    }
    let request: Request = serde_json::from_slice(&input)?;
    let mut cache = SemanticCache::new(CACHE_ENTRIES, CACHE_BYTES);
    let response = process_request(request, &mut cache)?;
    io::stdout().write_all(&encode_response(&response)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(source: &str) -> Request {
        Request {
            operation: "analyze".to_string(),
            filename: "src/sample.ts".to_string(),
            source: source.to_string(),
            position: Some(6),
            source_digest: None,
        }
    }

    #[test]
    fn builds_hoisted_and_shadowed_symbols_with_cfg() {
        let source = "f(); const x = 1; function f(y) { let x = y; while (x) { if (x > 1) break; x--; } return x; }";
        let mut cache = SemanticCache::new(8, 1024 * 1024);
        let response = process_request(request(source), &mut cache).unwrap();
        let names = response
            .symbols
            .iter()
            .map(|symbol| symbol.name.as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"f"));
        assert!(names.iter().filter(|name| **name == "x").count() >= 2);
        assert!(response.scopes.len() >= 3);
        assert!(response.cfg[0].entry_block_id.is_some());
        assert!(!response.cfg[0].exit_block_ids.is_empty());
        assert!(response.cfg[0].blocks.len() >= 3);
        assert!(!response.cfg[0].edges.is_empty());
    }

    #[test]
    fn references_are_selected_by_position_and_include_unresolved_names() {
        let mut selected = request("const x = 1; x; x;");
        selected.operation = "references".to_string();
        selected.position = Some(6);
        let mut cache = SemanticCache::new(8, 1024 * 1024);
        let response = process_request(selected, &mut cache).unwrap();
        assert_eq!(
            response.target.as_ref().map(|target| target.name.as_str()),
            Some("x")
        );
        assert_eq!(response.references.len(), 2);
        assert!(
            response
                .references
                .iter()
                .all(|reference| reference.name == "x")
        );

        let mut unresolved = request("missing();");
        unresolved.operation = "references".to_string();
        unresolved.position = Some(2);
        let response = process_request(unresolved, &mut cache).unwrap();
        assert_eq!(
            response.target.as_ref().map(|target| target.name.as_str()),
            Some("missing")
        );
        assert!(
            response
                .unresolved
                .iter()
                .any(|reference| reference.name == "missing")
        );
    }

    #[test]
    fn rejects_stale_digest() {
        let mut request = request("const x = 1;");
        request.source_digest = Some("stale".to_string());
        let mut cache = SemanticCache::new(8, 1024 * 1024);
        let error = process_request(request, &mut cache)
            .err()
            .expect("stale digest must fail");
        assert!(error.to_string().contains("source digest"));
    }

    #[test]
    fn cache_is_bounded_and_reports_hits() {
        let mut cache = SemanticCache::new(2, 1024 * 1024);
        let first = process_request(request("const a = 1;"), &mut cache).unwrap();
        let repeated = process_request(request("const a = 1;"), &mut cache).unwrap();
        process_request(request("const b = 2;"), &mut cache).unwrap();
        process_request(request("const c = 3;"), &mut cache).unwrap();
        let evicted = process_request(request("const a = 1;"), &mut cache).unwrap();
        assert!(!first.cache_hit);
        assert!(repeated.cache_hit);
        assert!(!evicted.cache_hit);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn rejects_path_escape_and_oversized_source() {
        let mut escaped = request("const x = 1;");
        escaped.filename = "../escape.ts".to_string();
        assert!(validate_request(&escaped).is_err());
        let mut oversized = request("");
        oversized.source = "x".repeat(MAX_SOURCE_BYTES + 1);
        assert!(validate_request(&oversized).is_err());
    }
}
