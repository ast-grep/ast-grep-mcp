use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub fn format_matches_as_text(matches: &[Value]) -> String {
    if matches.is_empty() {
        return String::new();
    }

    let mut output_blocks = Vec::new();

    for m in matches {
        let file_path = m.get("file").and_then(|v| v.as_str()).unwrap_or("");

        // lines are 0-indexed in JSON, convert to 1-indexed
        let start_line = m.pointer("/range/start/line")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) + 1;

        let end_line = m.pointer("/range/end/line")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) + 1;

        let match_text = m.get("text").and_then(|v| v.as_str()).unwrap_or("").trim_end();

        let header = if start_line == end_line {
            format!("{}:{}", file_path, start_line)
        } else {
            format!("{}:{}-{}", file_path, start_line, end_line)
        };

        output_blocks.push(format!("{}\n{}", header, match_text));
    }

    output_blocks.join("\n\n")
}

#[allow(dead_code)]
pub fn get_supported_languages(config_path: Option<&Path>) -> Vec<String> {
    let mut languages = vec![
        "bash", "c", "cpp", "csharp", "css", "elixir", "go", "haskell", "html", "java",
        "javascript", "json", "jsx", "kotlin", "lua", "nix", "php", "python", "ruby", "rust",
        "scala", "solidity", "swift", "tsx", "typescript", "yaml",
    ]
    .iter()
    .map(|&s| s.to_string())
    .collect::<Vec<String>>();

    if let Some(path) = config_path {
        if path.exists() {
            if let Ok(content) = fs::read_to_string(path) {
                // Parse as generic JSON Value to be flexible with YAML structure
                // serde_yaml::from_str returns Result<T>
                if let Ok(config) = serde_yaml::from_str::<serde_json::Value>(&content) {
                    if let Some(custom_langs) = config.get("customLanguages").and_then(|v| v.as_object()) {
                        for key in custom_langs.keys() {
                            languages.push(key.clone());
                        }
                    }
                }
            }
        }
    }

    // Sort and deduplicate using BTreeSet
    let unique_langs: BTreeSet<String> = languages.into_iter().collect();
    unique_langs.into_iter().collect()
}
