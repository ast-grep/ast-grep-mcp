use anyhow::Result;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

#[derive(Debug, thiserror::Error)]
pub enum CommandError {
    #[error("Command {cmd:?} failed with exit code {code}: {stderr}")]
    Failed { cmd: Vec<String>, code: i32, stderr: String },

    #[error("Command '{name}' not found. Please ensure {name} is installed and in PATH.")]
    NotFound { name: String, source: std::io::Error },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub struct CommandResult {
    pub stdout: String,
    pub stderr: String,
}

pub async fn run_command(args: &[String], input_text: Option<&str>) -> Result<CommandResult, CommandError> {
    // Windows handling: if command is "ast-grep", use shell=True equivalent
    // But here we are passed "args" where args[0] is likely "ast-grep".

    let mut cmd_args = args.to_vec();
    if cmd_args.is_empty() {
        return Err(CommandError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Empty command args")));
    }
    let program = cmd_args.remove(0);

    let mut command = if cfg!(target_os = "windows") && program == "ast-grep" {
        let mut cmd = Command::new("cmd");
        cmd.arg("/C");
        cmd.arg(&program);
        cmd.args(&cmd_args);
        cmd
    } else {
        let mut cmd = Command::new(&program);
        cmd.args(&cmd_args);
        cmd
    };

    command.stdin(Stdio::piped());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());

    // Spawn the child process
    let mut child = command.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            CommandError::NotFound { name: program.clone(), source: e }
        } else {
            CommandError::Io(e)
        }
    })?;

    // Write input to stdin if provided
    if let Some(input) = input_text {
        if let Some(mut stdin) = child.stdin.take() {
            if let Err(e) = stdin.write_all(input.as_bytes()).await {
                 // Ignore broken pipe errors as the process might have closed stdin
                 if e.kind() != std::io::ErrorKind::BrokenPipe {
                     return Err(CommandError::Io(e));
                 }
            }
        }
    }

    // Wait for output
    let output = child.wait_with_output().await.map_err(CommandError::Io)?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code().unwrap_or(1); // Default to 1 if no code (signal)

    if output.status.success() {
        return Ok(CommandResult { stdout, stderr });
    }

    // Handle exit code 1 logic
    if exit_code == 1 {
        let stdout_stripped = stdout.trim();
        // Valid "no matches" cases: empty JSON array or valid JSON with matches (starts with [)
        // or empty string
        if stdout_stripped.is_empty() || stdout_stripped == "[]" || stdout_stripped.starts_with('[') {
             return Ok(CommandResult { stdout, stderr });
        }

        // If --json flag is not present, empty stdout is also valid "no matches"
        // Check if --json is in args. Note: args here includes program name at index 0.
        if !args.contains(&"--json".to_string()) && stdout_stripped.is_empty() {
            return Ok(CommandResult { stdout, stderr });
        }
    }

    Err(CommandError::Failed {
        cmd: args.to_vec(),
        code: exit_code,
        stderr: if stderr.trim().is_empty() { "(no error output)".to_string() } else { stderr.trim().to_string() },
    })
}

pub async fn run_ast_grep(
    command: &str,
    args: &[String],
    input_text: Option<&str>,
    config_path: Option<&std::path::PathBuf>,
) -> Result<CommandResult> {
    let mut final_args = vec!["ast-grep".to_string(), command.to_string()];

    if let Some(path) = config_path {
        final_args.push("--config".to_string());
        final_args.push(path.to_string_lossy().to_string());
    }

    final_args.extend_from_slice(args);

    Ok(run_command(&final_args, input_text).await?)
}
