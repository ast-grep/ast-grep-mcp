use anyhow::Result;
use clap::{Parser, ValueEnum};
use std::env;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "ast-grep-mcp-server")]
#[command(author, version, about, long_about = None)]
#[command(after_help = "environment variables:
  AST_GREP_CONFIG    Path to sgconfig.yaml file (overridden by --config flag)

For more information, see: https://github.com/ast-grep/ast-grep-mcp")]
pub struct Cli {
    /// Path to sgconfig.yaml file for customizing ast-grep behavior (language mappings, rule directories, etc.)
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,

    /// Transport type for MCP server (default: stdio)
    #[arg(long, default_value_t = TransportType::Stdio, value_enum)]
    pub transport: TransportType,

    /// Port for SSE transport (default: 3101)
    #[arg(long, default_value_t = 3101)]
    pub port: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TransportType {
    Stdio,
    Sse,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub config_path: Option<PathBuf>,
    pub transport: TransportType,
    #[allow(dead_code)]
    pub port: u16,
}

impl Config {
    pub fn from_cli() -> Result<Self> {
        let cli = Cli::parse();
        let mut config_path = cli.config;

        // If config path not provided via CLI, check env var
        if config_path.is_none() {
            if let Ok(env_config) = env::var("AST_GREP_CONFIG") {
                if !env_config.is_empty() {
                    let path = PathBuf::from(env_config);
                    if !path.exists() {
                         anyhow::bail!("Config file '{}' specified in AST_GREP_CONFIG does not exist", path.display());
                    }
                    config_path = Some(path);
                }
            }
        } else if let Some(ref path) = config_path {
             if !path.exists() {
                anyhow::bail!("Config file '{}' does not exist", path.display());
            }
        }

        Ok(Self {
            config_path,
            transport: cli.transport,
            port: cli.port,
        })
    }
}
