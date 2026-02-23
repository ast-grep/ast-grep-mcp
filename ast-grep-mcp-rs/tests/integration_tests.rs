use std::path::PathBuf;
use std::process::Command;

fn ast_grep_available() -> bool {
    Command::new("ast-grep")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[tokio::test]
async fn test_find_code_integration() {
    if !ast_grep_available() {
        eprintln!("ast-grep not found, skipping integration test");
        return;
    }

    // Assuming we can run the binary or just test the library logic invoking the command
    // Since we refactored to a library, we can call run_ast_grep directly.

    use ast_grep_mcp::command::run_ast_grep;

    let fixture_path = PathBuf::from("tests/fixtures/example.py");
    let absolute_path = std::fs::canonicalize(&fixture_path).expect("Failed to get absolute path");
    let project_folder = absolute_path.parent().unwrap().to_string_lossy().to_string();

    // Test find_code logic via run_ast_grep directly
    // internal call: ast-grep run --pattern <pattern> --json <project_folder>

    let result = run_ast_grep(
        "run",
        &[
            "--pattern".to_string(),
            "def $NAME".to_string(),
            "--json".to_string(),
            project_folder,
        ],
        None,
        None
    ).await;

    assert!(result.is_ok(), "ast-grep command failed");
    let output = result.unwrap();
    // Verify JSON output
    assert!(output.stdout.contains("example_function") || output.stdout.contains("hello") || output.stdout.contains("add"));
}
