use ast_grep_mcp::config::{Config, TransportType};
use ast_grep_mcp::server::AstGrepServer;
use rmcp::transport::stdio;
use rmcp::ServiceExt;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Setup tracing (log to stderr only, never stdout — stdout is for MCP protocol)
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    // 2. Setup signal handlers
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        // Handle SIGINT (ignore)
        tokio::spawn(async move {
            let mut sigint = signal(SignalKind::interrupt()).unwrap();
            loop {
                sigint.recv().await;
                eprintln!("Received SIGINT - ignoring for multi-session stability");
            }
        });

        // Handle SIGTERM (shutdown)
        tokio::spawn(async move {
            let mut sigterm = signal(SignalKind::terminate()).unwrap();
            sigterm.recv().await;
            eprintln!("Received SIGTERM - shutting down gracefully");
            std::process::exit(0);
        });
    }

    #[cfg(not(unix))]
    {
        // On Windows, just handle Ctrl-C as SIGINT and ignore it
        tokio::spawn(async move {
            loop {
                if let Ok(_) = tokio::signal::ctrl_c().await {
                    eprintln!("Received SIGINT - ignoring for multi-session stability");
                }
            }
        });
    }

    // 3. Parse CLI args and build Config
    let config = Config::from_cli()?;

    // 4. Create server instance
    let server = AstGrepServer::new(config.clone());

    // 5. Start the server based on transport type
    match config.transport {
        TransportType::Stdio => {
            let service = server.serve(stdio()).await.map_err(|e| anyhow::anyhow!("Error starting server: {}", e))?;
            eprintln!("Server started on stdio");
            service.waiting().await.map_err(|e| anyhow::anyhow!("Error waiting for service: {}", e))?;
        }
        TransportType::Sse => {
            // SSE transport requires an HTTP server (e.g., axum) which is not currently in dependencies.
            // Documenting this limitation as per prompt instructions.
            anyhow::bail!("SSE transport is not yet implemented in this Rust port. Please use --transport stdio (default).");
        }
    }

    Ok(())
}
