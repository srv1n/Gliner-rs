use anyhow::Result;

use glinerrust::{gliner::InitConfig, types::InferenceResultSingle, Gliner};
#[tokio::main]
async fn main() -> Result<()> {
    let mut gliner = Gliner::new(InitConfig {
        tokenizer_path: "./tokenizer.json".to_string(),
        max_width: Some(12),
        model_path: "./model_quantized.onnx".to_string(),
        num_threads: Some(4),
    });

    let _ = gliner.initialize().await?;

    let input_texts = vec![

        "Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team.".to_owned(),
      
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


