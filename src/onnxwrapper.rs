use anyhow::anyhow;
use anyhow::{Context, Result};
use ort::{
    inputs, Environment, Error, GraphOptimizationLevel, Session, SessionBuilder, SessionInputValue,
};
use std::borrow::Cow;
use std::{path::Path, thread::available_parallelism};
use tokenizers::{PaddingParams, Tokenizer};
use tracing::{error, info};

pub struct ONNXWrapper {
    pub session: Session,
    // pub settings: ONNXSettings,
}

impl ONNXWrapper {
    pub fn new(model_path: String, num_threads: Option<usize>) -> Self {
        // let model_dir = "model_quantized.onnx";
        let threads = num_threads.unwrap_or(available_parallelism().unwrap().get() as usize);
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(threads)
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();

        ONNXWrapper {
            session: session,
            // settings,
        }
    }

    // pub fn init(&mut self) -> Result<()> {
    //     if self.session.is_none() {
    //         // info!("Threads for reranker: {}", threads);

    //         let session = match Session::builder() {
    //             Ok(builder) => {
    //                 info!("Session builder initialized successfully");
    //                 let execution_providers = vec![];

    //                 match builder.with_execution_providers(execution_providers) {
    //                     Ok(builder) => {
    //                         info!("Execution providers set successfully");
    //                         match builder.with_optimization_level(GraphOptimizationLevel::Level3) {
    //                             Ok(builder) => {
    //                                 info!("Optimization level set successfully");
    //                                 match builder.with_intra_threads(threads) {
    //                                     Ok(builder) => {
    //                                         info!("Intra threads set successfully: {}", threads);
    //                                         match builder.commit_from_file(model_dir) {
    //                                             Ok(session) => {
    //                                                 info!("Model loaded and session committed successfully");
    //                                                 session
    //                                             }
    //                                             Err(e) => {
    //                                                 error!(
    //                                                     "Error committing session from file: {:?}",
    //                                                     e
    //                                                 );
    //                                                 return Err(anyhow!(format!("Error: {:?}", e)));
    //                                                 // Err(e);
    //                                             }
    //                                         }
    //                                     }
    //                                     Err(e) => {
    //                                         error!("Error setting intra threads: {:?}", e);
    //                                         return Err(anyhow!(format!("Error: {:?}", e)));
    //                                     }
    //                                 }
    //                             }
    //                             Err(e) => {
    //                                 error!("Error setting optimization level: {:?}", e);
    //                                 return Err(anyhow!(format!("Error: {:?}", e)));
    //                             }
    //                         }
    //                     }
    //                     Err(e) => {
    //                         error!("Error setting execution providers: {:?}", e);
    //                         return Err(anyhow!(format!("Error: {:?}", e)));
    //                     }
    //                 }
    //             }
    //             Err(e) => {
    //                 error!("Error initializing session builder: {:?}", e);
    //                 return Err(anyhow!(format!("Error: {:?}", e)));
    //             }
    //         };
    //         let padding_params = PaddingParams::default();
    //         let mut tokenizer = Tokenizer::from_file(
    //             "tokenizer.json", // .join("resources/shared/reranker/")
    //                               // .join("mxbai-rerank-xsmall-v1-tokenizer.json"),
    //                               // .join("jina-reranker-v1-turbo-en-tokenizer.json"),
    //                               // .join("mxbai-rerank-base-v1-tokenizer.json"),

    //                               // .join("jina-reranker-v2-base-multilingual-tokenizer.json"),
    //                               // .join("reranker_tokenizer.json"),
    //         )
    //         .unwrap();
    //         tokenizer.with_padding(Some(padding_params));

    //         self.session = Some(session);
    //     }

    //     Ok(())
    // }

    // pub fn run(
    //     &self,
    //     inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)>,
    // ) -> OrtResult<ort::Outputs> {
    //     let session = self.session.as_ref().ok_or_else(|| {
    //         ort::Error::InvalidArgument(
    //             "ONNXWrapper: Session not initialized. Please call init() first.".into(),
    //         )
    //     })?;

    //     let input_values = inputs!(inputs)?;
    //     let outputs = session.run(input_values)?;
    //     Ok(outputs)
    // }
}
