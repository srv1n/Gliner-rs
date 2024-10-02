use anyhow::Result;
use gliner::InitConfig;

mod decoder;
mod gliner;
mod model;
mod onnxwrapper;
mod processor;
mod types;

pub use gliner::Gliner;
use types::InferenceResultSingle;
pub use types::{EntityResult, InferenceResultMultiple, RawInferenceResult};

#[tokio::main]
async fn main() -> Result<()> {
    let mut gliner = Gliner::new(InitConfig {
        tokenizer_path: "./tokenizer.json".to_string(),
        // onnx_settings: settings,
        max_width: Some(12),
        model_path: "./model_quantized.onnx".to_string(),
        num_threads: Some(4),
    });
    // let mut gliner = Gliner::new(InitConfig {
    //     tokenizer_path:
    //         "/Users/sarav/Downloads/Gliner/gliner-model-merge-large-v1.0/tokenizer.json".to_string(),
    //     // onnx_settings: settings,
    //     max_width: Some(12),
    //     model_path: "/Users/sarav/Downloads/Gliner/gliner-model-merge-large-v1.0/onnx/model.onnx"
    //         .to_string(),
    //     num_threads: Some(4),
    // });
    //     let mut gliner = Gliner::new(InitConfig {
    //     tokenizer_path: "/Users/sarav/Downloads/Gliner/gliclass-base-v1.0/knowledgator:gliclass-base-v1.0_tokenizer.json"
    //         .to_string(),
    //     // onnx_settings: settings,
    //     max_width: Some(12),
    //     model_path: "/Users/sarav/Downloads/Gliner/gliclass-base-v1.0/knowledgator:gliclass-base-v1.0_model-int8-quantized.onnx".to_string(),
    //     num_threads: Some(4),
    // });

    let _ = gliner.initialize().await?;

    let input_texts = vec![

        "Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team.".to_owned(),
        "John Smith is the CEO of OneTrust headquarters in Atlanta. He was made CEO last week. He is 55 years old. He is married to Jane Smith. They have two children, Emily and Michael. Emily is a lawyer and Michael is a software engineer. Emily will be getting married in the last week of December".to_string(),
    ];

    let entities = vec![
        "person",
        "award",
        "when",
        "date",
        "competitions",
        "teams",
        "location",
        "organization",
        "age",
        "amount",
    ];

    let result: Vec<InferenceResultSingle> = gliner
        .inference(&input_texts, &entities, false, 0.25)
        .await?;

    for each in result {
        println!("\nEntities: ");
        for entity in each.entities {
            println!("{}", entity);
        }
    }
    Ok(())
}
