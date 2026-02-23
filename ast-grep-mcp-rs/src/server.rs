use crate::command::run_ast_grep;
use crate::config::Config;
use crate::format::format_matches_as_text;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router,
    ErrorData as McpError,
    ServerHandler,
};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DumpSyntaxTreeParams {
    /// The code you need
    pub code: String,
    /// The language of the code. Supported: bash, c, cpp, csharp, css, elixir, go, haskell, html, java, javascript, json, jsx, kotlin, lua, nix, php, python, ruby, rust, scala, solidity, swift, tsx, typescript, yaml
    pub language: String,
    /// Code dump format. Available values: pattern, ast, cst
    #[serde(default = "default_cst")]
    pub format: String,
}

fn default_cst() -> String {
    "cst".to_string()
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TestMatchCodeRuleParams {
    /// The code to test against the rule
    pub code: String,
    /// The ast-grep YAML rule to search. It must have id, language, rule fields.
    pub yaml: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FindCodeParams {
    /// The absolute path to the project folder. It must be absolute path.
    pub project_folder: String,
    /// The ast-grep pattern to search for. Note, the pattern must have valid AST structure.
    pub pattern: String,
    /// The language of the code. Supported: bash, c, cpp, csharp, css, elixir, go, haskell, html, java, javascript, json, jsx, kotlin, lua, nix, php, python, ruby, rust, scala, solidity, swift, tsx, typescript, yaml. If not specified, will be auto-detected based on file extensions.
    #[serde(default)]
    pub language: String,
    /// Maximum results to return
    #[serde(default)]
    pub max_results: i32,
    /// 'text' or 'json'
    #[serde(default = "default_text")]
    pub output_format: String,
}

fn default_text() -> String {
    "text".to_string()
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FindCodeByRuleParams {
    /// The absolute path to the project folder. It must be absolute path.
    pub project_folder: String,
    /// The ast-grep YAML rule to search. It must have id, language, rule fields.
    pub yaml: String,
    /// Maximum results to return
    #[serde(default)]
    pub max_results: i32,
    /// 'text' or 'json'
    #[serde(default = "default_text")]
    pub output_format: String,
}

#[derive(Clone)]
pub struct AstGrepServer {
    config: Config,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl AstGrepServer {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "
Dump code's syntax structure or dump a query's pattern structure.
This is useful to discover correct syntax kind and syntax tree structure. Call it when debugging a rule.
The tool requires three arguments: code, language and format. The first two are self-explanatory.
`format` is the output format of the syntax tree.
use `format=cst` to inspect the code's concrete syntax tree structure, useful to debug target code.
use `format=pattern` to inspect how ast-grep interprets a pattern, useful to debug pattern rule.

Internally calls: ast-grep run --pattern <code> --lang <language> --debug-query=<format>
")]
    async fn dump_syntax_tree(
        &self,
        Parameters(params): Parameters<DumpSyntaxTreeParams>,
    ) -> Result<CallToolResult, McpError> {
        let result = run_ast_grep(
            "run",
            &[
                "--pattern".to_string(),
                params.code,
                "--lang".to_string(),
                params.language,
                format!("--debug-query={}", params.format),
            ],
            None,
            self.config.config_path.as_ref(),
        )
        .await
        .map_err(|e| McpError {
            code: ErrorCode(0),
            message: e.to_string().into(),
            data: None,
        })?;

        Ok(CallToolResult::success(vec![Content::text(
            result.stderr.trim().to_string(),
        )]))
    }

    #[tool(description = "
Test a code against an ast-grep YAML rule.
This is useful to test a rule before using it in a project.

Internally calls: ast-grep scan --inline-rules <yaml> --json --stdin
")]
    async fn test_match_code_rule(
        &self,
        Parameters(params): Parameters<TestMatchCodeRuleParams>,
    ) -> Result<CallToolResult, McpError> {
        let result = run_ast_grep(
            "scan",
            &[
                "--inline-rules".to_string(),
                params.yaml,
                "--json".to_string(),
                "--stdin".to_string(),
            ],
            Some(&params.code),
            self.config.config_path.as_ref(),
        )
        .await
        .map_err(|e| McpError {
            code: ErrorCode(0),
            message: e.to_string().into(),
            data: None,
        })?;

        let matches: Vec<Value> = serde_json::from_str(&result.stdout).unwrap_or_else(|_| vec![]);
        if matches.is_empty() {
             return Err(McpError {
                 code: ErrorCode(-32603), // Internal error or similar
                 message: "No matches found for the given code and rule. Try adding `stopBy: end` to your inside/has rule.".to_string().into(),
                 data: None,
             });
        }

        let json_str = serde_json::to_string_pretty(&matches).unwrap_or_default();
        Ok(CallToolResult::success(vec![Content::text(json_str)]))
    }

    #[tool(description = "
Find code in a project folder that matches the given ast-grep pattern.
Pattern is good for simple and single-AST node result.
For more complex usage, please use YAML by `find_code_by_rule`.

Internally calls: ast-grep run --pattern <pattern> [--json] <project_folder>

Output formats:
- text (default): Compact text format with file:line-range headers and complete match text
  Example:
    Found 2 matches:

    path/to/file.py:10-15
    def example_function():
        # function body
        return result

    path/to/file.py:20-22
    def another_function():
        pass

- json: Full match objects with metadata including ranges, meta-variables, etc.

The max_results parameter limits the number of complete matches returned (not individual lines).
When limited, the header shows \"Found X matches (showing first Y of Z)\".

Example usage:
  find_code(pattern=\"class $NAME\", max_results=20)  # Returns text format
  find_code(pattern=\"class $NAME\", output_format=\"json\")  # Returns JSON with metadata
")]
    async fn find_code(
        &self,
        Parameters(params): Parameters<FindCodeParams>,
    ) -> Result<CallToolResult, McpError> {
        if params.output_format != "text" && params.output_format != "json" {
             return Err(McpError {
                 code: ErrorCode(-32602), // Invalid params
                 message: format!("Invalid output_format: {}. Must be 'text' or 'json'.", params.output_format).into(),
                 data: None,
             });
        }

        let mut args = vec!["--pattern".to_string(), params.pattern];
        if !params.language.is_empty() {
            args.push("--lang".to_string());
            args.push(params.language);
        }
        args.push("--json".to_string());
        args.push(params.project_folder);

        let result = run_ast_grep(
            "run",
            &args,
            None,
            self.config.config_path.as_ref(),
        )
        .await
        .map_err(|e| McpError {
            code: ErrorCode(0),
            message: e.to_string().into(),
            data: None,
        })?;

        let stdout = result.stdout.trim();
        let matches: Vec<Value> = if stdout.is_empty() {
            vec![]
        } else {
            serde_json::from_str(stdout).unwrap_or_else(|_| vec![])
        };

        let total_matches = matches.len();
        let matches = if params.max_results > 0 && total_matches > params.max_results as usize {
            matches[..params.max_results as usize].to_vec()
        } else {
            matches
        };

        if params.output_format == "text" {
            if matches.is_empty() {
                return Ok(CallToolResult::success(vec![Content::text("No matches found")]));
            }
            let text_output = format_matches_as_text(&matches);
            let mut header = format!("Found {} matches", matches.len());
            if params.max_results > 0 && total_matches > params.max_results as usize {
                header = format!("Found {} matches (showing first {} of {})", total_matches, params.max_results, total_matches);
            }
            Ok(CallToolResult::success(vec![Content::text(format!("{}:\n\n{}", header, text_output))]))
        } else {
             let json_str = serde_json::to_string_pretty(&matches).unwrap_or_default();
             Ok(CallToolResult::success(vec![Content::text(json_str)]))
        }
    }

    #[tool(description = "
Find code using ast-grep's YAML rule in a project folder.
YAML rule is more powerful than simple pattern and can perform complex search like find AST inside/having another AST.
It is a more advanced search tool than the simple `find_code`.

Tip: When using relational rules (inside/has), add `stopBy: end` to ensure complete traversal.

Internally calls: ast-grep scan --inline-rules <yaml> [--json] <project_folder>

Output formats:
- text (default): Compact text format with file:line-range headers and complete match text
  Example:
    Found 2 matches:

    src/models.py:45-52
    class UserModel:
        def __init__(self):
            self.id = None
            self.name = None

    src/views.py:12
    class SimpleView: pass

- json: Full match objects with metadata including ranges, meta-variables, etc.

The max_results parameter limits the number of complete matches returned (not individual lines).
When limited, the header shows \"Found X matches (showing first Y of Z)\".

Example usage:
  find_code_by_rule(yaml=\"id: x\\nlanguage: python\\nrule: {pattern: 'class $NAME'}\", max_results=20)
  find_code_by_rule(yaml=\"...\", output_format=\"json\")  # For full metadata
")]
    async fn find_code_by_rule(
        &self,
        Parameters(params): Parameters<FindCodeByRuleParams>,
    ) -> Result<CallToolResult, McpError> {
         if params.output_format != "text" && params.output_format != "json" {
             return Err(McpError {
                 code: ErrorCode(-32602), // Invalid params
                 message: format!("Invalid output_format: {}. Must be 'text' or 'json'.", params.output_format).into(),
                 data: None,
             });
        }

        let args = vec!["--inline-rules".to_string(), params.yaml, "--json".to_string(), params.project_folder];

        let result = run_ast_grep(
            "scan",
            &args,
            None,
            self.config.config_path.as_ref(),
        )
        .await
        .map_err(|e| McpError {
            code: ErrorCode(0),
            message: e.to_string().into(),
            data: None,
        })?;

        let stdout = result.stdout.trim();
        let matches: Vec<Value> = if stdout.is_empty() {
            vec![]
        } else {
            serde_json::from_str(stdout).unwrap_or_else(|_| vec![])
        };

        let total_matches = matches.len();
        let matches = if params.max_results > 0 && total_matches > params.max_results as usize {
            matches[..params.max_results as usize].to_vec()
        } else {
            matches
        };

        if params.output_format == "text" {
            if matches.is_empty() {
                return Ok(CallToolResult::success(vec![Content::text("No matches found")]));
            }
            let text_output = format_matches_as_text(&matches);
            let mut header = format!("Found {} matches", matches.len());
            if params.max_results > 0 && total_matches > params.max_results as usize {
                header = format!("Found {} matches (showing first {} of {})", total_matches, params.max_results, total_matches);
            }
            Ok(CallToolResult::success(vec![Content::text(format!("{}:\n\n{}", header, text_output))]))
        } else {
             let json_str = serde_json::to_string_pretty(&matches).unwrap_or_default();
             Ok(CallToolResult::success(vec![Content::text(json_str)]))
        }
    }
}

#[tool_handler]
impl ServerHandler for AstGrepServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            server_info: Implementation {
                name: "ast-grep".into(),
                version: "0.1.0".into(),
                ..Default::default()
            },
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}
