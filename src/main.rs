use clap::{Parser, Subcommand};
use std::fs;

use webnn_graph::ast::GraphJson;
use webnn_graph::emit_js::{emit_builder_js, emit_weights_loader_js};
use webnn_graph::parser::parse_wg_text;
use webnn_graph::serialize::serialize_graph_to_wg_text;
use webnn_graph::validate::{validate_graph, validate_weights};
use webnn_graph::weights::WeightsManifest;
use webnn_graph::weights_io::{
    create_manifest, extract_weights, inline_weights, pack_weights, unpack_weights,
};

#[derive(Parser)]
#[command(name = "webnn-graph")]
#[command(about = "WebNN Graph DSL tools", long_about = None)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    Parse {
        path: String,
    },
    Validate {
        path: String,
        #[arg(long)]
        weights_manifest: Option<String>,
    },
    EmitJs {
        path: String,
    },
    Serialize {
        path: String,
    },
    PackWeights {
        #[arg(long)]
        manifest: String,
        #[arg(long)]
        input_dir: String,
        #[arg(long)]
        output: String,
    },
    UnpackWeights {
        #[arg(long)]
        weights: String,
        #[arg(long)]
        manifest: String,
        #[arg(long)]
        output_dir: String,
    },
    CreateManifest {
        #[arg(long)]
        input_dir: String,
        #[arg(long)]
        output: String,
        #[arg(long, default_value = "little")]
        endianness: String,
    },
    ExtractWeights {
        #[arg(long)]
        input: String,
        #[arg(long)]
        output_dir: String,
        #[arg(long)]
        weights: String,
        #[arg(long)]
        manifest: String,
        #[arg(long)]
        output_graph: String,
    },
    InlineWeights {
        #[arg(long)]
        input: String,
        #[arg(long)]
        weights: String,
        #[arg(long)]
        manifest: String,
        #[arg(long)]
        output: String,
    },
}

/// Load a graph from either .webnn text format or JSON format (auto-detect)
fn load_graph(path: &str) -> anyhow::Result<GraphJson> {
    let content = fs::read_to_string(path)?;
    let trimmed = content.trim_start();

    // Auto-detect format: .webnn starts with "webnn_graph", JSON starts with "{"
    if trimmed.starts_with("webnn_graph") {
        Ok(parse_wg_text(&content)?)
    } else if trimmed.starts_with('{') {
        Ok(serde_json::from_str(&content)?)
    } else {
        Err(anyhow::anyhow!(
            "Unknown format: file must be .webnn text (starts with 'webnn_graph') or JSON (starts with '{{')"
        ))
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Command::Parse { path } => {
            let txt = fs::read_to_string(path)?;
            let g = parse_wg_text(&txt)?;
            println!("{}", serde_json::to_string_pretty(&g)?);
        }
        Command::Validate {
            path,
            weights_manifest,
        } => {
            let g = load_graph(&path)?;
            validate_graph(&g)?;

            if let Some(mpath) = weights_manifest {
                let mtxt = fs::read_to_string(mpath)?;
                let m: WeightsManifest = serde_json::from_str(&mtxt)?;
                validate_weights(&g, &m)?;
            }
            eprintln!("OK");
        }
        Command::EmitJs { path } => {
            let g = load_graph(&path)?;
            validate_graph(&g)?;

            // Emit WeightsFile helper class
            print!("{}", emit_weights_loader_js());
            println!();

            // Emit buildGraph function
            let js = emit_builder_js(&g);
            print!("{js}");
        }
        Command::Serialize { path } => {
            let txt = fs::read_to_string(path)?;
            let g: GraphJson = serde_json::from_str(&txt)?;
            let wg_text = serialize_graph_to_wg_text(&g)?;
            print!("{}", wg_text);
        }
        Command::PackWeights {
            manifest,
            input_dir,
            output,
        } => {
            pack_weights(&manifest, &input_dir, &output)?;
        }
        Command::UnpackWeights {
            weights,
            manifest,
            output_dir,
        } => {
            unpack_weights(&weights, &manifest, &output_dir)?;
        }
        Command::CreateManifest {
            input_dir,
            output,
            endianness,
        } => {
            create_manifest(&input_dir, &output, &endianness)?;
        }
        Command::ExtractWeights {
            input,
            output_dir,
            weights,
            manifest,
            output_graph,
        } => {
            let g = load_graph(&input)?;
            let new_graph = extract_weights(&g, &output_dir, &weights, &manifest)?;
            let json = serde_json::to_string_pretty(&new_graph)?;
            fs::write(&output_graph, json)?;
            eprintln!("Wrote graph with weight references to: {}", output_graph);
        }
        Command::InlineWeights {
            input,
            weights,
            manifest,
            output,
        } => {
            let g = load_graph(&input)?;
            let new_graph = inline_weights(&g, &weights, &manifest)?;
            let json = serde_json::to_string_pretty(&new_graph)?;
            fs::write(&output, json)?;
            eprintln!("Wrote graph with inline weights to: {}", output);
        }
    }

    Ok(())
}
