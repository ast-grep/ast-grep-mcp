use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

fn request() -> Value {
    json!({
        "operation": "analyze",
        "filename": "src/sample.ts",
        "source": "const value: number = 1; console.log(value);",
        "position": 6,
        "source_digest": null
    })
}

#[test]
fn serves_multiple_json_lines_and_reuses_snapshots() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_ast-soleaux-analysis"))
        .arg("--serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let encoded = serde_json::to_string(&request()).unwrap();
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{{not-json").unwrap();
    writeln!(stdin, "{encoded}").unwrap();
    writeln!(stdin, "{encoded}").unwrap();
    drop(stdin);

    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let responses = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(responses.len(), 3);
    assert!(responses[0]["error"].as_str().is_some());
    assert_eq!(responses[1]["cache_hit"], false);
    assert_eq!(responses[2]["cache_hit"], true);
    assert_eq!(responses[2]["worker_version"], "0.1.0");
}

#[test]
fn preserves_one_shot_stdin_compatibility() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_ast-soleaux-analysis"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(&request()).unwrap().as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let response: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["operation"], "analyze");
    assert_eq!(response["cache_hit"], false);
}
