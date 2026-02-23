use ast_grep_mcp::format::{format_matches_as_text, get_supported_languages};
use serde_json::json;

#[test]
fn test_format_matches_as_text_empty() {
    let matches = vec![];
    assert_eq!(format_matches_as_text(&matches), "");
}

#[test]
fn test_format_matches_as_text_single_line() {
    let matches = vec![json!({
        "file": "test.py",
        "range": {
            "start": { "line": 0, "column": 0 },
            "end": { "line": 0, "column": 10 }
        },
        "text": "def foo():"
    })];

    let result = format_matches_as_text(&matches);
    assert_eq!(result, "test.py:1\ndef foo():");
}

#[test]
fn test_format_matches_as_text_multi_line() {
    let matches = vec![json!({
        "file": "test.py",
        "range": {
            "start": { "line": 0, "column": 0 },
            "end": { "line": 2, "column": 10 }
        },
        "text": "def foo():\n    pass\n    return"
    })];

    let result = format_matches_as_text(&matches);
    assert_eq!(result, "test.py:1-3\ndef foo():\n    pass\n    return");
}

#[test]
fn test_format_matches_as_text_multiple_matches() {
    let matches = vec![
        json!({
            "file": "test.py",
            "range": {
                "start": { "line": 0, "column": 0 },
                "end": { "line": 0, "column": 10 }
            },
            "text": "match1"
        }),
        json!({
            "file": "test.py",
            "range": {
                "start": { "line": 10, "column": 0 },
                "end": { "line": 10, "column": 10 }
            },
            "text": "match2"
        })
    ];

    let result = format_matches_as_text(&matches);
    assert_eq!(result, "test.py:1\nmatch1\n\ntest.py:11\nmatch2");
}

#[test]
fn test_get_supported_languages_default() {
    let langs = get_supported_languages(None);
    assert!(langs.contains(&"python".to_string()));
    assert!(langs.contains(&"rust".to_string()));
    // Check sorted
    let mut sorted_langs = langs.clone();
    sorted_langs.sort();
    assert_eq!(langs, sorted_langs);
}

#[test]
fn test_get_supported_languages_with_config() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("sgconfig.yaml");

    let mut file = std::fs::File::create(&config_path).unwrap();
    writeln!(file, "customLanguages:\n  my-lang:\n    extensions: [ml]").unwrap();

    let langs = get_supported_languages(Some(&config_path));
    assert!(langs.contains(&"my-lang".to_string()));
    assert!(langs.contains(&"python".to_string()));
}
